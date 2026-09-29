"""Learned DR baselines (paper Section III.G): B5, the physics-guided neural baseline, and its ablations.

B5 predicts hourly load from exogenous inputs only (calendar, outdoor temperature, on-site PV potential):

    y_t = b(c_t) * (1 + r(theta_t)) + h(theta_t) + c0 - beta * pv_t

  b(c)      base load (IT load in a data center, internal load in a building): an MLP over calendar features,
            kept positive by a softplus output;
  r(theta)  cooling electricity per unit of base load, the role of (1 + overhead) / COP(theta): monotone
            increasing and convex in temperature;
  h(theta)  weather-driven gains (envelope, ventilation): monotone increasing and convex;
  c0 >= 0   constant auxiliary load;  beta in [0, 1]  the share of on-site PV that offsets import.

r and h are sums of softplus hinges with non-negative weights at fixed temperature knots, so they are
monotone and convex for every parameter value: the physics enters as structure (a physics-guided or grey-box
network), not as a loss penalty. Beyond the training range they continue with the slope of the last active
hinge, which is how a linear COP decline extrapolates; a tree ensemble would stay flat, and a free MLP may bend.
The model is trained only on pre-season data and its weights are then committed (protocol.commit), so later
load cannot move it.
"""
import hashlib
import io
import numpy as np
import torch
from torch import nn

torch.set_num_threads(max(1, torch.get_num_threads()))

KNOTS = np.arange(0.0, 57.5, 2.5)        # temperature knots (deg C) for the monotone hinge sums
HINGE_WIDTH = 1.5                        # softness of each hinge (deg C)
T_SCALE = 10.0                           # hinge outputs are in units of 10 deg C, so weights are O(0.01-1)


N_CAL = 31


def calendar_features(hour, dow):
    """Hour of day and day of week, both one-hot: 31 features. The base network combines them, so it can
    represent any hour-of-week profile (the resolution of a TOWT regression); smooth harmonics of the hour
    could not follow sharp schedule changes."""
    hour, dow = np.asarray(hour, int), np.asarray(dow, int)
    return np.column_stack([np.eye(24)[hour], np.eye(7)[dow]]).astype(np.float32)


class MonotoneHinge(nn.Module):
    """g(theta) = softplus(g0) + sum_k softplus(a_k) * w * softplus((theta - tau_k) / w) / T_SCALE:
    positive, increasing and convex in theta for every parameter value."""

    def __init__(self, knots=KNOTS, width=HINGE_WIDTH, init=-8.0, init0=-1.0):
        super().__init__()
        self.register_buffer("tau", torch.tensor(knots, dtype=torch.float32))
        self.w = width
        self.a = nn.Parameter(torch.full((len(knots),), init))
        self.g0 = nn.Parameter(torch.tensor(init0))

    def forward(self, theta):
        z = nn.functional.softplus((theta[:, None] - self.tau[None, :]) / self.w) * self.w / T_SCALE
        return nn.functional.softplus(self.g0) + z @ nn.functional.softplus(self.a)

    def weights(self):
        return nn.functional.softplus(self.a)


class PhysicsGuided(nn.Module):
    """B5. constrained=False replaces r and h by free MLPs of temperature (ablation: no shape constraints)."""

    def __init__(self, n_cal=N_CAL, hidden=32, constrained=True, use_temp=True):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(n_cal, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 1))
        self.constrained, self.use_temp = constrained, use_temp
        if constrained:
            self.r, self.h = MonotoneHinge(), MonotoneHinge()
        else:
            mk = lambda: nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 1))
            self.r_net, self.h_net = mk(), mk()
        self.c0 = nn.Parameter(torch.tensor(-3.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def parts(self, cal, theta, pv):
        b = nn.functional.softplus(self.base(cal)).squeeze(-1)
        if not self.use_temp:
            r = h = torch.zeros_like(theta)
        elif self.constrained:
            r, h = self.r(theta), self.h(theta)
        else:
            t = (theta[:, None] - 30.0) / 10.0
            r, h = self.r_net(t).squeeze(-1), self.h_net(t).squeeze(-1)
        return b, r, h

    def forward(self, cal, theta, pv):
        b, r, h = self.parts(cal, theta, pv)
        return b * (1 + r) + h + nn.functional.softplus(self.c0) - torch.sigmoid(self.beta) * pv


class BlackBox(nn.Module):
    """Ablation: an unconstrained MLP over the same inputs (calendar, temperature, PV)."""

    def __init__(self, n_cal=N_CAL, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_cal + 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, cal, theta, pv):
        x = torch.cat([cal, ((theta - 30.0) / 10.0)[:, None], pv[:, None]], dim=1)
        return self.net(x).squeeze(-1)


class Learned:
    """A trained baseline model: an ensemble of networks (or one LightGBM model) with its input scaling."""

    def __init__(self, kind, nets=None, gbm=None, scale=1.0, history=None):
        self.kind, self.nets, self.gbm, self.scale, self.history = kind, nets or [], gbm, scale, history or []

    def predict(self, hour, dow, theta, pv):
        if self.kind == "gbm":
            X = np.column_stack([hour, dow, theta, pv])
            return self.gbm.predict(X) * self.scale
        cal = torch.tensor(calendar_features(hour, dow))
        th = torch.tensor(np.asarray(theta, np.float32))
        p = torch.tensor(np.asarray(pv, np.float32) / self.scale)
        with torch.no_grad():
            return np.mean([net(cal, th, p).numpy() for net in self.nets], axis=0) * self.scale

    def model_bytes(self):
        """Deterministic serialization for the commitment: every parameter as float64, in a fixed order."""
        if self.kind == "gbm":
            return self.gbm.booster_.model_to_string().encode()
        buf = io.BytesIO()
        for net in self.nets:
            for name, t in sorted(net.state_dict().items()):
                buf.write(name.encode()); buf.write(t.detach().cpu().numpy().astype(np.float64).tobytes())
        return buf.getvalue()

    def sha256(self):
        return hashlib.sha256(self.model_bytes()).hexdigest()


def fit(kind, hour, dow, theta, pv, y, seeds=(0, 1, 2, 3, 4), epochs=1500, lr=3e-3, val_frac=0.15,
        weight_decay=1e-4, hinge_l2=1e-4):
    """Train one baseline on (pre-season) hourly data. kind: 'pgnb' (B5), 'pgnb_free' (no shape constraints),
    'pgnb_notemp' (calendar only), 'mlp' (black box), 'gbm' (LightGBM). Neural models are ensembles over seeds,
    trained full-batch with Adam and a cosine schedule; the last val_frac of the period (in time) is held out
    to record the validation curve and pick each seed's best epoch."""
    y = np.asarray(y, np.float64)
    scale = float(np.mean(y))
    if kind == "gbm":
        import lightgbm as lgb
        g = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                              subsample=0.9, subsample_freq=1, colsample_bytree=1.0, random_state=seeds[0], verbose=-1)
        g.fit(np.column_stack([hour, dow, theta, pv]), y / scale)
        return Learned("gbm", gbm=g, scale=scale)
    n = len(y); n_val = max(24, int(round(val_frac * n)))
    cal = torch.tensor(calendar_features(hour, dow))
    th = torch.tensor(np.asarray(theta, np.float32))
    p = torch.tensor(np.asarray(pv, np.float32) / scale)
    yt = torch.tensor((y / scale).astype(np.float32))
    tr, va = slice(0, n - n_val), slice(n - n_val, n)
    nets, hist = [], []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        if kind == "mlp":
            net = BlackBox()
        else:
            net = PhysicsGuided(constrained=(kind != "pgnb_free"), use_temp=(kind != "pgnb_notemp"))
        shape = [q for n_, q in net.named_parameters() if n_.startswith(("r.", "h.", "c0", "beta"))]
        rest = [q for n_, q in net.named_parameters() if not n_.startswith(("r.", "h.", "c0", "beta"))]
        opt = torch.optim.Adam([{"params": rest, "weight_decay": weight_decay},
                                {"params": shape, "lr": 5 * lr, "weight_decay": 0.0}], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, best_state, curve = np.inf, None, []
        for ep in range(epochs):
            opt.zero_grad()
            pred = net(cal[tr], th[tr], p[tr])
            loss = torch.mean((pred - yt[tr]) ** 2)
            if isinstance(net, PhysicsGuided) and net.constrained:
                loss = loss + hinge_l2 * (net.r.weights().pow(2).sum() + net.h.weights().pow(2).sum())
            loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                v = torch.mean((net(cal[va], th[va], p[va]) - yt[va]) ** 2).item()
            curve.append((float(loss.item()), v))
            if v < best:
                best, best_state = v, {k: t.clone() for k, t in net.state_dict().items()}
        net.load_state_dict(best_state)
        net.eval()
        nets.append(net); hist.append(np.array(curve))
    return Learned(kind, nets=nets, scale=scale, history=hist)
