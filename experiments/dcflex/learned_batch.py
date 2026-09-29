"""Vectorized training of many small baseline networks at once (one per building x seed).

The models of `learned.py` are tiny, so training them one by one is dominated by Python overhead. Here every
parameter carries a leading model axis M (buildings x seeds) and one tensor program trains all of them: the loss
is a sum of per-model losses, so each model's gradient is independent and Adam treats them separately.

Two model families, both driven only by exogenous inputs (calendar, dry-bulb temperature, dew point, PV):
  * `PGNB`  physics-guided:  y = b(c)·(1 + r(θ)) + h_T(θ) + h_D(θ_dew) + c0 − β·pv
            with r, h_T, h_D monotone increasing convex hinge sums (structure, not penalty);
  * `MLP`   unconstrained black box over the same inputs.
Optional building embeddings turn PGNB into a shared, pretrained model (`SharedPGNB`) for transfer.
"""
import math
import numpy as np
import torch
from torch import nn

from .learned import calendar_features, N_CAL, KNOTS, HINGE_WIDTH, T_SCALE

DEW_KNOTS = np.arange(-10.0, 32.5, 2.5)
HINGE_INIT = -8.0


def _hinge_basis(x, knots, width=HINGE_WIDTH):
    """[N, K] softplus hinge features w·softplus((x − τ_k)/w)/T_SCALE; non-negative, increasing and convex in x."""
    tau = torch.tensor(knots, dtype=torch.float32)
    return nn.functional.softplus((x[:, None] - tau[None, :]) / width) * width / T_SCALE


def _init(shape, fan_in, gen):
    return (torch.rand(shape, generator=gen) * 2 - 1) / math.sqrt(fan_in)


class PGNB(nn.Module):
    """M independent physics-guided networks. constrained=False: free MLPs of weather replace the hinge sums."""

    def __init__(self, M, n_cal=N_CAL, hidden=32, seed=0, constrained=True, use_dew=True, use_temp=True):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.M, self.constrained, self.use_dew, self.use_temp = M, constrained, use_dew, use_temp
        P = lambda *s, fan: nn.Parameter(_init(s, fan, g))
        self.W1, self.b1 = P(M, n_cal, hidden, fan=n_cal), P(M, hidden, fan=n_cal)
        self.W2, self.b2 = P(M, hidden, hidden, fan=hidden), P(M, hidden, fan=hidden)
        self.W3, self.b3 = P(M, hidden, fan=hidden), nn.Parameter(torch.zeros(M) + 0.5)
        K, KD = len(KNOTS), len(DEW_KNOTS)
        if constrained:
            # hinge weights start near zero (softplus(-8) = 3e-4): no weather response until the data asks for one
            self.ar, self.r0 = nn.Parameter(torch.full((M, K), HINGE_INIT)), nn.Parameter(torch.full((M,), -1.0))
            self.ah = nn.Parameter(torch.full((M, K), HINGE_INIT))
            self.ad = nn.Parameter(torch.full((M, KD), HINGE_INIT))
        else:
            self.fW1, self.fb1 = P(M, 2, 16, fan=2), P(M, 16, fan=2)
            self.fW2, self.fb2 = P(M, 16, 2, fan=16), nn.Parameter(torch.zeros(M, 2))
        self.c0 = nn.Parameter(torch.full((M,), -3.0))
        self.beta = nn.Parameter(torch.zeros(M))

    def base(self, cal):
        h = nn.functional.silu(torch.einsum("nf,mfh->mnh", cal, self.W1) + self.b1[:, None, :])
        h = nn.functional.silu(torch.einsum("mnh,mhk->mnk", h, self.W2) + self.b2[:, None, :])
        return nn.functional.softplus(torch.einsum("mnh,mh->mn", h, self.W3) + self.b3[:, None])

    def weather(self, BT, BD, theta, dew):
        """r (multiplicative cooling term) and the additive weather gain, each [M, N]."""
        if not self.use_temp:
            z = torch.zeros(self.M, BT.shape[0])
            return z, z
        if self.constrained:
            sp = nn.functional.softplus
            r = sp(self.r0)[:, None] + sp(self.ar) @ BT.T
            add = sp(self.ah) @ BT.T + (sp(self.ad) @ BD.T if self.use_dew else 0.0)
            return r, add
        x = torch.stack([(theta - 30) / 10, (dew - 15) / 10 if self.use_dew else torch.zeros_like(dew)], 1)
        f = nn.functional.silu(torch.einsum("nf,mfh->mnh", x, self.fW1) + self.fb1[:, None, :])
        f = torch.einsum("mnh,mhk->mnk", f, self.fW2) + self.fb2[:, None, :]
        return f[..., 0], f[..., 1]

    def forward(self, cal, BT, BD, theta, dew, pv):
        r, add = self.weather(BT, BD, theta, dew)
        return (self.base(cal) * (1 + r) + add + nn.functional.softplus(self.c0)[:, None]
                - torch.sigmoid(self.beta)[:, None] * pv[None, :])

    def shape_penalty(self):
        if not self.constrained:
            return torch.zeros(self.c0.shape[0])
        sp = nn.functional.softplus
        return sp(self.ar).pow(2).sum(1) + sp(self.ah).pow(2).sum(1) + sp(self.ad).pow(2).sum(1)


class MLP(nn.Module):
    """M independent black-box MLPs over calendar features, temperature, dew point and PV."""

    def __init__(self, M, n_cal=N_CAL, hidden=64, seed=0, use_dew=True):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        P = lambda *s, fan: nn.Parameter(_init(s, fan, g))
        n_in = n_cal + 3
        self.use_dew = use_dew
        self.W1, self.b1 = P(M, n_in, hidden, fan=n_in), P(M, hidden, fan=n_in)
        self.W2, self.b2 = P(M, hidden, hidden, fan=hidden), P(M, hidden, fan=hidden)
        self.W3, self.b3 = P(M, hidden, fan=hidden), nn.Parameter(torch.ones(M))

    def forward(self, cal, BT, BD, theta, dew, pv):
        dew_ = (dew - 15) / 10 if self.use_dew else torch.zeros_like(dew)
        x = torch.cat([cal, ((theta - 30) / 10)[:, None], dew_[:, None], pv[:, None]], 1)
        h = nn.functional.silu(torch.einsum("nf,mfh->mnh", x, self.W1) + self.b1[:, None, :])
        h = nn.functional.silu(torch.einsum("mnh,mhk->mnk", h, self.W2) + self.b2[:, None, :])
        return torch.einsum("mnh,mh->mn", h, self.W3) + self.b3[:, None]

    def shape_penalty(self):
        return torch.zeros(self.b3.shape[0])


def fit_many(kind, hour, dow, theta, dew, pv, Y, train, seeds=(0, 1, 2, 3, 4), epochs=1000, lr=3e-3,
             val_frac=0.15, weight_decay=1e-4, shape_l2=1e-3, **kw):
    """Train one model per column of Y ([N, B], NaN = missing) on the rows in `train` (bool [N]), for each seed.
    Inputs are shared by all columns (buildings of one site). Returns (predict, info): predict(hour, dow, theta,
    dew, pv) -> [N', B] ensemble-mean predictions in the units of Y; info holds training curves and scales.
    Early stopping per model on the last val_frac of the training rows (in time)."""
    Y = np.asarray(Y, float)
    B, S = Y.shape[1], len(seeds)
    scale = np.nanmean(np.where(train[:, None], Y, np.nan), axis=0)
    rows = np.where(train)[0]
    n_val = max(24, int(round(val_frac * len(rows))))
    tr_rows, va_rows = rows[:-n_val], rows[-n_val:]
    t = lambda a: torch.tensor(np.asarray(a, np.float32))
    cal = t(calendar_features(hour, dow)); th, dw, p = t(theta), t(dew), t(pv)
    BT, BD = _hinge_basis(th, KNOTS), _hinge_basis(dw, DEW_KNOTS)
    Yn = torch.tensor(np.tile((Y / scale).T, (S, 1)), dtype=torch.float32)          # [M, N], M = S*B
    mask = ~torch.isnan(Yn)
    Yn = torch.nan_to_num(Yn)
    pvn = p / float(np.nanmean(scale))
    M = S * B
    nets = [((PGNB if kind.startswith("pgnb") else MLP)(B, seed=s, **kw)) for s in seeds]
    # stack the per-seed modules into one: each seed initialized from its own generator
    net = nets[0]
    with torch.no_grad():
        for name, par in net.named_parameters():
            par.data = torch.cat([dict(n.named_parameters())[name].data for n in nets], 0)
    if isinstance(net, PGNB):
        net.M = M
    shape_names = ("ar", "r0", "ah", "ad", "c0", "beta")
    groups = [{"params": [q for n_, q in net.named_parameters() if n_ not in shape_names], "weight_decay": weight_decay},
              {"params": [q for n_, q in net.named_parameters() if n_ in shape_names], "lr": 5 * lr, "weight_decay": 0.0}]
    opt = torch.optim.Adam(groups, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tri, vai = torch.tensor(tr_rows), torch.tensor(va_rows)
    best = torch.full((M,), float("inf")); best_state = {k: v.clone() for k, v in net.state_dict().items()}
    curve = []
    for ep in range(epochs):
        opt.zero_grad()
        out = net(cal[tri], BT[tri], BD[tri], th[tri], dw[tri], pvn[tri])
        err = (out - Yn[:, tri]) ** 2 * mask[:, tri]
        per = err.sum(1) / mask[:, tri].sum(1).clamp(min=1)
        loss = per.sum() + (shape_l2 * net.shape_penalty()).sum()
        loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            ov = net(cal[vai], BT[vai], BD[vai], th[vai], dw[vai], pvn[vai])
            v = (((ov - Yn[:, vai]) ** 2) * mask[:, vai]).sum(1) / mask[:, vai].sum(1).clamp(min=1)
            imp = v < best
            best = torch.where(imp, v, best)
            for k, val in net.state_dict().items():
                if val.dim() >= 1 and val.shape[0] == M:
                    sel = imp.view(-1, *([1] * (val.dim() - 1)))
                    best_state[k] = torch.where(sel, val, best_state[k])
        curve.append((float(per.detach().mean()), float(v.mean())))
    net.load_state_dict(best_state)

    def predict(hour_, dow_, theta_, dew_, pv_):
        c_ = t(calendar_features(hour_, dow_)); th_, dw_ = t(theta_), t(dew_)
        with torch.no_grad():
            out = net(c_, _hinge_basis(th_, KNOTS), _hinge_basis(dw_, DEW_KNOTS), th_, dw_,
                      t(pv_) / float(np.nanmean(scale))).numpy()
        return out.reshape(S, B, -1).mean(0).T * scale

    return predict, dict(curve=np.array(curve), scale=scale, net=net, seeds=seeds)


def committed(hour, dow, theta, dew, pv, train, seeds=(0, 1, 2, 3, 4), epochs=1500, threads=1, **kw):
    """B5 as the settlement pipeline uses it: a function load -> hourly predictions of a PGNB ensemble trained
    only on the hours in `train` (the pre-season), plus the trained network for the commitment. Training runs
    on `threads` CPU threads so that the result does not depend on the machine's core count."""
    cache = {}

    def fit(load):
        key = np.asarray(load, float)[train].tobytes()
        if key in cache:                       # same pre-season data, same committed model
            return cache[key]
        n0 = torch.get_num_threads()
        torch.set_num_threads(threads)
        try:
            pred, info = fit_many("pgnb", hour, dow, theta, dew, pv, np.asarray(load, float)[:, None], train,
                                  seeds=seeds, epochs=epochs, use_dew=True, **kw)
            fit.info = info
            cache[key] = pred(hour, dow, theta, dew, pv)[:, 0]
            return cache[key]
        finally:
            torch.set_num_threads(n0)
    return fit


def model_bytes(net):
    """Deterministic serialization of a trained ensemble for the commitment: every parameter as float64
    (name, then values), in sorted order."""
    out = bytearray()
    for name, t in sorted(net.state_dict().items()):
        out += name.encode() + t.detach().cpu().numpy().astype(np.float64).tobytes()
    return bytes(out)
