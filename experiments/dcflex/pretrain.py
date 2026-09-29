"""Cross-building pretraining of the physics-guided baseline, and few-shot adaptation to a new facility.

One network serves every building. Each building i has a learned embedding e_i; a hypernetwork maps e_i to the
non-negative hinge weights of that building's cooling response r_i(theta) and weather gains h_i(theta), h_i(dew),
so every building's temperature response stays monotone increasing and convex, as in learned_batch.PGNB:

    y_it / s_i = b(c_t, e_i) * (1 + r_i(theta_t)) + h_i(theta_t) + h_i^D(dew_t) + c0_i

s_i is the building's mean load, so buildings of any size share one network. The base b is an MLP over the
calendar features and e_i. `kind="mlp"` is the unconstrained counterpart (an MLP over calendar, weather and e_i),
so that pretraining and physics structure can be compared as two separate factors.

A new facility gets a fresh embedding fitted on its pre-season days with every shared weight frozen: 16 numbers
per building, which is what a committed pre-season baseline can afford to learn from 60 days of data.
"""
import numpy as np
import torch
from torch import nn

from .learned import calendar_features, N_CAL, KNOTS, HINGE_WIDTH, T_SCALE
from .learned_batch import DEW_KNOTS, HINGE_INIT, _hinge_basis


class Shared(nn.Module):
    def __init__(self, n_build, kind="pgnb", dim=16, hidden=64, n_cal=N_CAL, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.kind, self.dim = kind, dim
        self.emb = nn.Parameter(0.1 * torch.randn(n_build, dim))
        n_in = n_cal + dim + (0 if kind == "pgnb" else 2)
        self.base = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 1))
        if kind == "pgnb":
            K, KD = len(KNOTS), len(DEW_KNOTS)
            self.hyper = nn.Linear(dim, 2 * K + KD + 2)          # hinge weights (r, h, dew), r0, c0
            with torch.no_grad():
                self.hyper.weight.mul_(0.1)
                self.hyper.bias.copy_(torch.cat([torch.full((2 * K + KD,), HINGE_INIT), torch.tensor([-1.0, -3.0])]))
            self.K, self.KD = K, KD

    def heads(self, e):
        """Per-building physics parameters from embeddings e [B, dim]."""
        z = self.hyper(e)
        sp = nn.functional.softplus
        K, KD = self.K, self.KD
        return sp(z[:, :K]), sp(z[:, K:2 * K]), sp(z[:, 2 * K:2 * K + KD]), sp(z[:, -2]), sp(z[:, -1])

    def forward(self, e, cal, BT, BD, theta, dew):
        """Row-wise: e [n, dim] (the embedding of each row's building) and inputs [n, ...] -> [n]."""
        if self.kind == "pgnb":
            b = nn.functional.softplus(self.base(torch.cat([cal, e], 1))).squeeze(-1)
            ar, ah, ad, r0, c0 = self.heads(e)
            r = r0 + (BT * ar).sum(1)
            return b * (1 + r) + (BT * ah).sum(1) + (BD * ad).sum(1) + c0
        x = torch.cat([cal, e, ((theta - 30) / 10)[:, None], ((dew - 15) / 10)[:, None]], 1)
        return self.base(x).squeeze(-1)

    def grid(self, E, cal, BT, BD, theta, dew):
        """Predictions [B, N] for every building embedding in E [B, dim] at every row of the inputs [N, ...].
        Same function as forward(); the first layer and the hypernetwork are evaluated once per building."""
        lin1, rest = self.base[0], self.base[1:]
        nc = cal.shape[1]
        W = lin1.weight                                                    # [hidden, n_in]
        xa = cal @ W[:, :nc].T + lin1.bias                                 # [N, hidden]
        if self.kind != "pgnb":
            x = torch.stack([(theta - 30) / 10, (dew - 15) / 10], 1)
            xa = xa + x @ W[:, nc + self.dim:].T
        h = xa[None, :, :] + (E @ W[:, nc:nc + self.dim].T)[:, None, :]    # [B, N, hidden]
        out = rest(h).squeeze(-1)
        if self.kind != "pgnb":
            return out
        b = nn.functional.softplus(out)
        ar, ah, ad, r0, c0 = self.heads(E)
        return b * (1 + r0[:, None] + ar @ BT.T) + ah @ BT.T + ad @ BD.T + c0[:, None]

    def shape_penalty(self, e):
        if self.kind != "pgnb":
            return torch.zeros(())
        ar, ah, ad, _, _ = self.heads(e)
        return (ar ** 2).sum(1).mean() + (ah ** 2).sum(1).mean() + (ad ** 2).sum(1).mean()


def _inputs(hour, dow, theta, dew):
    t = lambda a: torch.tensor(np.asarray(a, np.float32))
    th, dw = t(theta), t(dew)
    return t(calendar_features(hour, dow)), _hinge_basis(th, KNOTS), _hinge_basis(dw, DEW_KNOTS), th, dw


def pool_rows(panels, max_ratio=8.0):
    """Stack a list of panels (dicts with Y [N, B], hour, dow, temp, dew and a boolean `use` [N]) into row arrays
    for pretraining. Each building is scaled by its mean over the rows used; readings above max_ratio times the
    mean (meter spikes) are dropped."""
    cols = {k: [] for k in ("b", "hour", "dow", "temp", "dew", "y")}
    names, nb = [], 0
    for p in panels:
        ok_w = np.isfinite(p["temp"]) & np.isfinite(p["dew"]) & p["use"]
        for j in range(p["Y"].shape[1]):
            y = p["Y"][:, j]
            m = ok_w & np.isfinite(y)
            if m.sum() < 24 * 60:
                continue
            s = y[m].mean()
            if not s > 0:
                continue
            m &= y <= max_ratio * s
            cols["b"].append(np.full(m.sum(), nb, np.int32)); cols["y"].append((y[m] / s).astype(np.float32))
            for k, v in (("hour", p["hour"]), ("dow", p["dow"]), ("temp", p["temp"]), ("dew", p["dew"])):
                cols[k].append(v[m])
            names.append(p["names"][j]); nb += 1
    return {k: np.concatenate(v) for k, v in cols.items()}, names


def pretrain(rows, n_build, kind="pgnb", epochs=8, batch=16384, lr=2e-3, shape_l2=1e-3, huber=0.5, seed=0, log=None):
    """Minibatch Adam over all rows (Huber loss, robust to the meter noise of a large pool), one-cycle schedule.
    Returns the trained Shared model and the loss curve."""
    net = Shared(n_build, kind=kind, seed=seed)
    cal, BT, BD, th, dw = _inputs(rows["hour"], rows["dow"], rows["temp"], rows["dew"])
    y, b = torch.tensor(rows["y"]), torch.tensor(rows["b"]).long()
    n = len(y)
    steps = epochs * ((n + batch - 1) // batch)
    # the hypernetwork's hinge biases start at -8 (no weather response) and need large steps to reach the data;
    # embeddings are touched by few rows per batch, so they get a larger rate too
    fast = [q for n_, q in net.named_parameters() if n_.startswith(("hyper", "emb"))]
    rest = [q for n_, q in net.named_parameters() if not n_.startswith(("hyper", "emb"))]
    opt = torch.optim.Adam([{"params": rest, "lr": lr}, {"params": fast, "lr": 10 * lr}])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, 10 * lr], total_steps=steps, pct_start=0.05)
    g = torch.Generator().manual_seed(seed)
    curve = []
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for a in range(0, n, batch):
            i = perm[a:a + batch]
            e = net.emb[b[i]]
            out = net(e, cal[i], BT[i], BD[i], th[i], dw[i])
            loss = nn.functional.huber_loss(out, y[i], delta=huber) + shape_l2 * net.shape_penalty(e)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss.detach()) * len(i)
        curve.append(tot / n)
        if log:
            log(f"  epoch {ep + 1}/{epochs}  loss {curve[-1]:.5f}")
    return net, np.array(curve)


def adapt(net, hour, dow, theta, dew, Y, train, epochs=600, lr=2e-2, val_frac=0.15, restarts=4, seed=0):
    """Few-shot adaptation: a new embedding per column of Y ([N, B], NaN = missing), fitted on rows `train` with
    the shared network frozen; early stopping on the last val_frac of the training rows. Several restarts from
    embeddings of pretraining buildings; the best on validation is kept. Returns predict(...) -> [N', B]."""
    for q in net.parameters():
        q.requires_grad_(False)
    Y = np.asarray(Y, float)
    B = Y.shape[1]
    scale = np.nanmean(np.where(train[:, None], Y, np.nan), axis=0)
    rows = np.where(train)[0]
    n_val = max(24, int(round(val_frac * len(rows))))
    tri, vai = torch.tensor(rows[:-n_val]), torch.tensor(rows[-n_val:])
    cal, BT, BD, th, dw = _inputs(hour, dow, theta, dew)
    Yn = torch.tensor((Y / scale).T, dtype=torch.float32)
    mask = ~torch.isnan(Yn)
    Yn = torch.nan_to_num(Yn)
    g = torch.Generator().manual_seed(seed)
    R = restarts
    start = net.emb[torch.randint(0, net.emb.shape[0], (R * B,), generator=g)].clone()
    E = nn.Parameter(start)                                           # [R*B, dim]
    opt = torch.optim.Adam([E], lr=lr)
    Yr, Mr = Yn.repeat(R, 1), mask.repeat(R, 1)

    def grid(Eb, sel):
        """Predictions [len(Eb), len(sel)] for every embedding at rows sel."""
        return net.grid(Eb, cal[sel], BT[sel], BD[sel], th[sel], dw[sel])

    best = torch.full((R * B,), float("inf")); best_E = E.detach().clone()
    for ep in range(epochs):
        opt.zero_grad()
        err = ((grid(E, tri) - Yr[:, tri]) ** 2 * Mr[:, tri]).sum(1) / Mr[:, tri].sum(1).clamp(min=1)
        err.sum().backward(); opt.step()
        with torch.no_grad():
            v = ((grid(E, vai) - Yr[:, vai]) ** 2 * Mr[:, vai]).sum(1) / Mr[:, vai].sum(1).clamp(min=1)
            imp = v < best
            best = torch.where(imp, v, best)
            best_E = torch.where(imp[:, None], E.detach(), best_E)
    pick = best.view(R, B).argmin(0)
    E_final = best_E.view(R, B, -1)[pick, torch.arange(B)]

    def predict(hour_, dow_, theta_, dew_, chunk=4096):
        c_, bt_, bd_, th_, dw_ = _inputs(hour_, dow_, theta_, dew_)
        out = []
        with torch.no_grad():
            for a in range(0, len(th_), chunk):
                s_ = slice(a, a + chunk)
                out.append(net.grid(E_final, c_[s_], bt_[s_], bd_[s_], th_[s_], dw_[s_]))
        return torch.cat(out, 1).numpy().T * scale

    return predict, dict(embedding=E_final, val=best.view(R, B).min(0).values.numpy())
