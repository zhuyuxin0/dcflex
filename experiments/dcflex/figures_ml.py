"""Figures of the ML extension (learned baselines on real buildings, forecast-driven control). Style: dcflex.style;
violet marks learned or forecast-driven components, blue the perfect-foresight reference, green the rule-based
strategy R, grey business as usual and naive references."""
import numpy as np
import matplotlib.pyplot as plt

from .style import PALETTE as P, TEXT_WIDTH_IN as W, finalize_figure

BAR = dict(edgecolor="black", linewidth=0.7)


def mpc(path, retained, reliability, r0, levels=(0.8, 0.9, 1.0), reliability_wl=None):
    """(a) Share of the perfect-foresight system value retained by each controller over the summer;
    (b) share of DR events delivered in full against the commitment level, by planning margin."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.4), gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    rows = [("Rule-based R", "R", P["green_3"]),
            ("MPC, raw ECMWF forecast", "ecmwf", P["red_1"]),
            ("MPC, persistence", "persistence", P["neutral"]),
            ("MPC, bias-corrected ECMWF", "ecmwf_mos", "#C9A3C2"),
            ("MPC, expected price", "ecmwf_expected", P["violet"]),
            ("MPC, perfect forecasts", "perfect", P["blue_main"])]
    y = np.arange(len(rows))
    vals = [100 * retained[k] for _, k, _ in rows]
    a.barh(y, vals, color=[c for *_, c in rows], height=0.62, **BAR)
    for yi, v in zip(y, vals):
        a.text(v + 2.5, yi, f"{v:.0f}%", va="center", fontsize=7.5)
    a.set_yticks(y); a.set_yticklabels([r[0] for r in rows], fontsize=8)
    a.set_xlim(0, 115); a.set_xticks(range(0, 101, 25))
    a.axvline(100, color="black", lw=0.8, ls="--", zorder=0)
    a.set_xlabel("System value retained (% of perfect foresight)")
    a.set_title("(a) Summer value under day-ahead forecasts")
    modes = [("none", "No margin", P["grey_mid"], "o", "--"),
             ("conf0.2", "Conformal margin, $\\alpha=0.2$", "#C9A3C2", "s", "-"),
             ("conf0.1", "Conformal margin, $\\alpha=0.1$", P["violet"], "D", "-")]
    for m, lab, col, mk, ls in modes:
        ev = [100 * reliability[(lv, m)]["events"] for lv in levels]
        b.plot([lv * r0 for lv in levels], ev, color=col, marker=mk, ls=ls, lw=1.4, ms=5, label=lab)
    if reliability_wl is not None:
        ev = [100 * reliability_wl[(lv, "conf0.1")]["events"] for lv in levels if (lv, "conf0.1") in reliability_wl]
        b.plot([lv * r0 for lv in levels][:len(ev)], ev, color=P["violet"], marker="D", mfc="white", ls=":", lw=1.4,
               ms=5, label="$\\alpha=0.1$, real IT-load forecast errors")
    b.axhline(80, color="black", lw=0.8, ls=":")
    b.text(levels[-1] * r0, 81.5, "80% reliability", fontsize=7, va="bottom", ha="right")
    b.set_xlabel("Committed reduction (MW)"); b.set_ylabel("Events held in every hour (%)")
    b.set_ylim(0, 105); b.set_title("(b) S3 delivery under forecasts")
    b.legend(loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, fontsize=7, handlelength=2.2,
             columnspacing=0.8)
    finalize_figure(fig, path)


RD_STYLE = {"b1": ("B1 High-5-of-10", P["red_strong"], "^", ":"), "b2": ("B2 Rolling regression", P["red_2"], "v", ":"),
            "towt": ("B3 Committed regression", P["grey_dark"], "s", "--"),
            "gbm": ("LightGBM", P["teal"], "X", "-."), "mlp": ("MLP", P["grey_mid"], "o", "-"),
            "pgnb": ("B5 physics-guided", P["violet"], "D", "-"),
            "pt_pgnb": ("B5 pretrained", "#C9A3C2", "P", "--")}
KIND = {"chilledwater": "chilled water", "electricity": "electricity"}
SITE = {"Fox": "Tempe", "Bull": "Austin", "Panther": "Orlando"}


def realdata(path, df, show, show_name):
    """(a) Median event-window nMAE per site and meter for each baseline (new facility, 60 training days);
    (b) temperature response at 15:00 on a weekday for one building, against its summer and training readings."""
    new = df[df.regime == "new"]
    groups = [(s, k) for k in ("chilledwater", "electricity") for s in ("Fox", "Bull", "Panther")
              if len(new[(new.site == s) & (new.kind == k)])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 1.5 + 0.42 * max(len(groups), 4)),
                               gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    for gi, (s, k) in enumerate(groups):
        g = new[(new.site == s) & (new.kind == k)]
        for mi, (m, (lab, col, mk, _)) in enumerate(RD_STYLE.items()):
            v = g[g.model == m].event_nmae.median()
            if np.isfinite(v):
                a.scatter(v, gi + (mi - 3) * 0.1, color=col, marker=mk, s=26, edgecolor="black", linewidth=0.4,
                          zorder=3)
        nb = g.building.nunique()
        a.axhline(gi + 0.5, color=P["neutral"], lw=0.6, zorder=0)
    a.set_yticks(range(len(groups)))
    a.set_yticklabels([f"{SITE[s]}, {KIND[k]}\n({new[(new.site == s) & (new.kind == k)].building.nunique()} meters)"
                       for s, k in groups], fontsize=7.5)
    a.set_ylim(len(groups) - 0.5, -0.5)
    a.set_xlabel("Event-window nMAE, median (%)"); a.set_title("(a) Hottest weekdays, 14:00–18:00")
    a.set_xlim(left=0)
    t = show["resp_temp"]
    t_hi = float(np.nanmax(show["train_temp"]))
    b.axvspan(t_hi, t[-1], color=P["highlight"], alpha=0.18, lw=0)
    b.text(t_hi + 0.3, 0.02, "beyond training\ntemperatures", transform=b.get_xaxis_transform(), fontsize=6.8,
           va="bottom", color=P["grey_dark"])
    j = show["j"]
    b.scatter(show["train_temp"], show["train_y"][:, j], s=6, color=P["blue_secondary"], alpha=0.5, lw=0,
              label="Training days (Apr–May)")
    b.scatter(show["obs_temp"], show["obs_y"][:, j], s=6, color=P["grey_mid"], alpha=0.5, lw=0,
              label="Season (Jun–Sep)")
    for m in ("towt", "gbm", "mlp", "pgnb", "pt_pgnb"):
        if "resp_" + m in show:
            lab, col, _, ls = RD_STYLE[m]
            b.plot(t, show["resp_" + m][:, j], color=col, ls=ls, lw=1.5, label=lab)
    lo = min(np.nanmin(show["train_temp"]), np.nanmin(show["obs_temp"])) - 2
    b.set_xlim(lo, t[-1]); b.set_xlabel("Outdoor temperature ($^\\circ$C)"); b.set_ylabel("Hourly load at 15:00 (kWh)")
    b.set_title(f"(b) Learned response, {show_name}")
    # one legend entry per model: its marker in (a) and, for the models drawn in (b), its line style
    from matplotlib.lines import Line2D
    drawn = {m for m in RD_STYLE if "resp_" + m in show}
    h1 = [Line2D([], [], color=col, marker=mk, ms=5, mec="black", mew=0.4, ls=ls if m in drawn else "none", lw=1.3)
          for m, (lab, col, mk, ls) in RD_STYLE.items()]
    l1 = [lab for lab, *_ in RD_STYLE.values()]
    h2, l2 = b.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(h2, l2) if l.startswith(("Training", "Season"))]
    fig.legend(h1 + [h for h, _ in keep], l1 + [l for _, l in keep], loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.4, handlelength=2.6, columnspacing=0.9)
    fig.tight_layout(pad=0.6, rect=(0, 0.12, 1, 1))
    finalize_figure(fig, path, tight=False)


def training(path, syn_curve, rd_curves, pt_curves):
    """(a) B5 on the synthetic facility: training and validation loss per epoch (mean over the ensemble);
    (b) validation loss of the building models on the hot-desert chilled-water panel; (c) pretraining loss
    per epoch for each pool (the tested site left out)."""
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.5))
    a, b, c = axs
    if syn_curve is not None:
        ep = np.arange(1, len(syn_curve) + 1)
        a.plot(ep, syn_curve[:, 0], color=P["violet"], lw=1.2, label="Training")
        a.plot(ep, syn_curve[:, 1], color=P["violet"], lw=1.2, ls="--", label="Validation")
        a.set_yscale("log"); a.legend(fontsize=7)
    a.set_xlabel("Epoch"); a.set_ylabel("Mean squared error (scaled)"); a.set_title("(a) B5, synthetic facility")
    for m, (lab, col, ls) in {"pgnb": ("B5", P["violet"], "-"), "pgnb_free": ("B5, no constraints", "#C9A3C2", "--"),
                              "mlp": ("MLP", P["grey_mid"], ":")}.items():
        if m in rd_curves:
            cv = rd_curves[m]
            b.plot(np.arange(1, len(cv) + 1), cv[:, 1], color=col, ls=ls, lw=1.2, label=lab)
    if rd_curves:
        b.set_yscale("log"); b.legend(fontsize=7)
    b.set_xlabel("Epoch"); b.set_title("(b) Validation, Tempe chilled water")
    for (kind, site, mk), cv in sorted(pt_curves.items()):
        col = P["violet"] if mk == "pgnb" else P["grey_mid"]
        ls = "-" if kind == "chilledwater" else "--"
        c.plot(np.arange(1, len(cv) + 1), cv, color=col, ls=ls, lw=1.0, marker="o", ms=2.5,
               label=f"{'B5' if mk == 'pgnb' else 'MLP'}, {'chilled water' if kind == 'chilledwater' else 'electricity'}")
    h, l = c.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    c.legend(uniq.values(), uniq.keys(), fontsize=6.5)
    c.set_xlabel("Epoch"); c.set_ylabel("Huber loss"); c.set_title("(c) Pretraining, per pool")
    finalize_figure(fig, path)
