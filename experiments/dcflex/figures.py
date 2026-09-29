"""Data figures for the paper (written to figures/). Style: dcflex.style (figures4papers)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

from .style import PALETTE as P, DIVERGING, TEXT_WIDTH_IN as W, finalize_figure

SCEN = {"S0": ("S0 business as usual", P["grey_mid"], "--"),
        "S1c": ("S1 flat tariff", P["red_strong"], "-"),
        "S2": ("S2 system signal", P["blue_main"], "-")}
# lever ablation: one hue, completeness by alpha (figures4papers ablation pattern)
LEVALPHA = {"Compute deferral only": 0.30, "TES only": 0.55, "Battery only": 0.80, "All levers": 1.0}
BAR = dict(edgecolor="black", linewidth=0.7)


def _prof(x, idx, months=(6, 7, 8, 9)):
    m = np.isin(idx.month, months)
    h = idx.hour.to_numpy()[m]
    return np.array([np.asarray(x)[m][h == k].mean() for k in range(24)])


def _hours(ax):
    ax.set_xlim(-0.5, 23.5); ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("Hour of day (local time)")


def duck(path, S, sig, idx):
    """(a) average-day system net load, summer and winter; (b) summer campus import by scenario."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.2))
    # system quantities in black (summer solid, winter dashed), so that grey keeps its meaning of business as usual
    for months, lab, ls in (((6, 7, 8, 9), "Summer (Jun–Sep)", "-"), ((12, 1, 2), "Winter (Dec–Feb)", "--")):
        a.plot(range(24), _prof(sig["net"], idx, months), color=P["ink"], ls=ls, label=lab)
    a.axhline(0, color=P["neutral"], lw=1.0, zorder=0)
    a.set_ylabel("System net load (GW)"); _hours(a); a.set_title("(a) Synthetic 2030 net load")
    for key in ("S0", "S1c", "S2"):
        lab, col, ls = SCEN[key]
        b.plot(range(24), _prof(S[key]["pg"], idx), color=col, ls=ls, label=lab)
    b.set_ylabel("Campus grid import (MW)"); _hours(b); b.set_title("(b) Campus import, summer")
    # one legend under each panel, at the same height
    for ax, nc in ((a, 2), (b, 3)):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=nc, fontsize=7.8, handlelength=2.0,
                  columnspacing=1.0)
    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(bottom=0.3)
    finalize_figure(fig, path, tight=False)


def flex(path, F):
    durs = sorted(next(iter(F.values())).keys())
    fig, ax = plt.subplots(figsize=(W, 2.6))
    w = 0.8 / len(F)
    for i, (name, vals) in enumerate(F.items()):
        x = np.arange(len(durs)) + (i - (len(F) - 1) / 2) * w
        bars = ax.bar(x, [vals[d] for d in durs], w, label=name, color=P["blue_main"],
                      alpha=LEVALPHA.get(name, 0.6), **BAR)
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(), f"{r.get_height():.0f}", ha="center",
                    va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(durs))); ax.set_xticklabels([f"{d}-hour hold" for d in durs])
    ax.set_ylabel("Firm reduction $F(d)$ (MW)")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    finalize_figure(fig, path)


def heatmap(path, S, idx):
    d = S["S2"]["pg"] - S["S0"]["pg"]
    M = np.array([[d[(idx.month == m) & (idx.hour == h)].mean() for m in range(1, 13)] for h in range(24)])
    fig, ax = plt.subplots(figsize=(W, 3.0))
    lim = max(np.nanmax(np.abs(M)), 1e-6)
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim))
    ax.set_xticks(range(12)); ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_yticks(range(0, 24, 3)); ax.set_xlabel("Month"); ax.set_ylabel("Hour of day")
    for s in ax.spines.values():
        s.set_visible(True)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Change in grid import, S2 − S0 (MW)"); cb.outline.set_linewidth(0.8)
    finalize_figure(fig, path)


def value(path, vals):
    """Value per kW-yr of the four-hour firm reduction, by perspective."""
    fig, a = plt.subplots(figsize=(0.72 * W, 2.9))
    names = [k for k, v in vals.items() if v is not None]
    # private values in the red family (lightest = flat tariffs), system blue, ERCOT benchmark teal; DEWA hatched
    style = {"Private (ADDC flat)": (P["red_1"], ""), "Private (ADDC transmission)": (P["red_2"], ""),
             "Private (ADDC TOU)": (P["red_strong"], ""), "Private (DEWA)": (P["red_2"], "///"),
             "System": (P["blue_main"], ""), "ERCOT": (P["teal"], "")}
    bars = a.bar(range(len(names)), [vals[k] for k in names], color=[style.get(n, (P["neutral"], ""))[0] for n in names],
                 hatch=None, **BAR)
    for r, n in zip(bars, names):
        r.set_hatch(style.get(n, (None, ""))[1])
        a.text(r.get_x() + r.get_width() / 2, r.get_height(), f"{r.get_height():,.0f}", ha="center", va="bottom", fontsize=7.5)
    a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    a.set_ylabel("AED per kW-yr of $F(4)$")
    finalize_figure(fig, path)


def baselines(path, acc, names):
    keys = list(names)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.9))
    r1 = a.bar(range(len(keys)), [acc[k]["nmae"] for k in keys], color=P["neutral"], **BAR)
    oc = [acc[k]["overcredit"] for k in keys]
    r2 = b.bar(range(len(keys)), oc, color=[P["red_strong"] if v > 0 else P["blue_main"] for v in oc], **BAR)
    for ax, rs, lab, t in ((a, r1, "nMAE (%)", "(a) Placebo accuracy"), (b, r2, "Over-crediting (%)", "(b) Under 10% inflation")):
        for r in rs:
            h = r.get_height()     # label above a positive bar, below a negative one
            ax.text(r.get_x() + r.get_width() / 2, h, f"{h:.1f}".replace("-", "\u2212"), ha="center",
                    va="bottom" if h >= 0 else "top", fontsize=7.5)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([names[k] for k in keys], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(lab); ax.set_title(t)
    b.axhline(0, color="black", lw=0.8)
    if min(oc) < 0:                # room for the label under a negative bar
        b.set_ylim(min(oc) - 0.15 * (max(max(oc), 0) - min(oc)), b.get_ylim()[1])
    finalize_figure(fig, path)


def tornado(path, base, rows):
    rows = sorted(rows, key=lambda r: abs(r[4] - r[3]))
    fig, ax = plt.subplots(figsize=(0.9 * W, 2.9))
    lo_c, hi_c = P["blue_secondary"], P["blue_main"]
    for i, (lab, lo, hi, flo, fhi, *_) in enumerate(rows):
        # draw the end farther from the base first, so both stay visible when they fall on the same side
        for val, col, al in sorted(((flo, lo_c, 0.45), (fhi, hi_c, 1.0)), key=lambda t: -abs(t[0] - base)):
            ax.barh(i, val - base, left=base, color=col, alpha=al, height=0.62, **BAR)
    ax.axvline(base, color="black", lw=1.0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r[0]} ({r[5]})" if len(r) > 5 else r[0] for r in rows])
    ax.set_xlabel("Four-hour firm reduction $F(4)$ (MW)")
    ax.legend(handles=[Patch(facecolor=lo_c, alpha=0.45, **BAR, label="Low end of range"),
                       Patch(facecolor=hi_c, **BAR, label="High end of range")], loc="lower right", fontsize=8)
    finalize_figure(fig, path)


def cop(path, fac, temp=None, slopes=(0.05, 0.08), points=None):
    """Fitted COP curve with the slope range; optional hourly temperature histogram and rating points."""
    t = np.linspace(15, 50, 200)
    fig, ax = plt.subplots(figsize=(0.8 * W, 2.8))
    if temp is not None:
        ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
        n, *_ = ax2.hist(temp, bins=35, color=P["neutral"], alpha=0.6, zorder=0); ax2.set_ylabel("Hours in 2025")
        ax2.set_ylim(0, 1.7 * max(n))   # keep the histogram low so the legend sits on clear space
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    curve = lambda k: np.maximum(fac.cop_min, fac.cop_ref - k * (t - fac.t_ref))
    ax.fill_between(t, curve(slopes[0]), curve(slopes[1]), color=P["blue_secondary"], alpha=0.18, lw=0,
                    label=f"slope {slopes[0]:.2f}–{slopes[1]:.2f} K$^{{-1}}$")
    ax.plot(t, curve(fac.cop_slope), color=P["blue_main"], label=f"fitted, $k$ = {fac.cop_slope:.3f} K$^{{-1}}$")
    for (x, y, lab, mk) in (points or []):
        ax.plot(x, y, mk, color="black", mfc="white", ms=5, mew=1.0, ls="none", label=lab)
    ax.set_xlabel("Outdoor air temperature (°C)"); ax.set_ylabel("Chiller COP")
    ax.legend(loc="upper right", fontsize=7.5)
    finalize_figure(fig, path)


def _lever_parts(s0, s3, cop, fac, h):
    """Change in grid import (S3 - S0) at hours h, split by lever; the parts sum to the change exactly
    (up to PV self-consumption, reported separately). Compute deferral carries its own cooling."""
    ex = 1 + fac.loss_frac + fac.fan_frac
    dpit = s3["pit"][h] - s0["pit"][h]
    return {"Compute deferral": ex * dpit * (1 + 1 / cop[h]),
            "Thermal storage": (s3["qin"][h] - s3["qout"][h]) / cop[h],
            "Battery": s3["bc"][h] - s3["bd"][h],
            "On-site PV": -(s3["pv_used"][h] - s0["pv_used"][h])}


def dispatch(path, s0, s3, net, idx, ev_days, peak_hour, fac, cop, H, r):
    """Event-day dispatch under the DR contract (S3). Left: the event with the highest net-load peak,
    from noon the day before: (a) import, (b) change in import by lever, (c) storage levels and system
    net load. Right: (d) import on every event day, 12:00-24:00."""
    w0 = s3["window"].start
    sl = lambda x, a, b: np.asarray(x)[a - w0:b - w0]          # S3 arrays are indexed from the window start
    dmax = np.asarray(net).reshape(-1, 24).max(1)
    e = max(ev_days, key=lambda d: dmax[d])
    a, b = e * 24 - 12, e * 24 + 24
    hh = np.arange(b - a)
    s0w = {k: np.asarray(v)[a:b] for k, v in s0.items() if isinstance(v, np.ndarray)}
    s3w = {k: sl(v, a, b) for k, v in s3.items() if isinstance(v, np.ndarray)}
    ev = (np.arange(a, b) >= e * 24 + peak_hour[e]) & (np.arange(a, b) < e * 24 + peak_hour[e] + H)
    fig = plt.figure(figsize=(W, 7.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.75, 1], height_ratios=[1.0, 1.0, 0.85], hspace=0.38, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_a)
    for ax in (ax_a, ax_b, ax_c):
        ax.axvspan(hh[ev][0] - 0.5, hh[ev][-1] + 0.5, color=P["highlight"], alpha=0.2, lw=0, zorder=0)
        ax.axvline(11.5, color=P["neutral"], lw=0.8, zorder=0)
    h_bau, = ax_a.plot(hh, s0w["pg"], color=P["grey_mid"], ls="--", label="S0 business as usual")
    h_s3, = ax_a.plot(hh, s3w["pg"], color=P["blue_main"], label="S3 DR contract")
    h_cm = ax_a.hlines(s0w["pg"][ev] - r, hh[ev] - 0.5, hh[ev] + 0.5, color=P["red_strong"], lw=1.8,
                       label=f"Baseline $-$ committed {r:.1f} MW")
    ax_a.set_ylabel("Import (MW)"); ax_a.set_title("(a) Campus grid import around the peak event")
    ax_a.text(hh[ev].mean(), ax_a.get_ylim()[1], "event", ha="center", va="top", fontsize=7.5, color=P["grey_dark"])
    parts = _lever_parts(s0w, s3w, np.asarray(cop)[a:b], fac, slice(None))
    alphas = {"Compute deferral": 0.35, "Thermal storage": 0.65, "Battery": 1.0}
    pos, neg = np.zeros(len(hh)), np.zeros(len(hh))
    hb = []
    if np.max(np.abs(parts["On-site PV"])) < 0.05:     # PV is self-consumed in both cases: nothing to show
        parts.pop("On-site PV")
    for name, v in parts.items():
        col = P["neutral"] if name == "On-site PV" else P["blue_main"]
        al = 1.0 if name == "On-site PV" else alphas[name]
        base = np.where(v >= 0, pos, neg)
        hb.append(ax_b.bar(hh, v, 0.82, bottom=base, color=col, alpha=al, edgecolor="black", linewidth=0.35, label=name))
        pos, neg = pos + np.maximum(v, 0), neg + np.minimum(v, 0)
    h_net, = ax_b.plot(hh, s3w["pg"] - s0w["pg"], color="black", lw=1.0, marker="o", ms=2.2, label="Net change in import")
    ax_b.axhline(0, color="black", lw=0.7)
    ax_b.set_ylabel("S3 $-$ S0 (MW)"); ax_b.set_title("(b) Change in import, split by lever")
    cap_tes = fac.tes_hours * fac.peak_heat_mw
    # storage levels in the shades of their bars in (b): thermal storage mid blue, battery dark blue (dash-dot)
    tes_col = "#6A8FBE"
    h_tes, = ax_c.plot(hh, 100 * s3w["soc"] / cap_tes, color=tes_col, lw=2.0, label="Thermal storage level (%)")
    h_bat, = ax_c.plot(hh, 100 * s3w["soe"] / fac.bess_mwh, color=P["blue_main"], ls="-.", lw=1.4,
                       label="Battery level (%)")
    ax_c.set_ylabel("Level (%)"); ax_c.set_ylim(0, 108)
    ax_n = ax_c.twinx(); ax_n.spines["right"].set_visible(True)
    h_nl, = ax_n.plot(hh, np.asarray(net)[a:b], color=P["ink"], lw=1.0, ls=":", label="System net load (GW, right)")
    ax_n.set_ylabel("Net load (GW)")
    ax_c.set_title("(c) Storage levels and system net load")
    ticks = np.arange(0, len(hh) + 1, 6)
    ax_c.set_xticks(ticks); ax_c.set_xticklabels([f"{(12 + t) % 24:02d}:00" for t in ticks])
    ax_c.set_xlim(-0.5, len(hh) - 0.5)
    ax_c.set_xlabel(f"Local time, {idx[e * 24 - 24].strftime('%d %b')}–{idx[e * 24].strftime('%d %b %Y')}")
    for ax in (ax_a, ax_b):
        plt.setp(ax.get_xticklabels(), visible=False)
    # (d) small multiples: every event day, 12:00-24:00, shared scale
    sub = gs[:, 1].subgridspec(5, 2, hspace=0.62, wspace=0.12)
    seg = lambda d: (np.asarray(s0["pg"])[d * 24 + 12:d * 24 + 24], sl(s3["pg"], d * 24 + 12, d * 24 + 24))
    lo = min(min(x.min() for x in seg(d)) for d in ev_days)
    hi = max(max(x.max() for x in seg(d)) for d in ev_days)
    for i, d in enumerate(sorted(ev_days)):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        x = np.arange(12, 24)
        y0, y3 = seg(d)
        ax.axvspan(peak_hour[d] - 0.5, peak_hour[d] + H - 0.5, color=P["highlight"], alpha=0.2, lw=0)
        h_fill = ax.fill_between(x, y3, y0, where=(x >= peak_hour[d]) & (x < peak_hour[d] + H), facecolor="none",
                                 edgecolor=P["blue_main"], hatch="//////", lw=0, step=None,
                                 label="Delivered reduction (d)")
        ax.plot(x, y0, color=P["grey_mid"], ls="--", lw=1.0)
        ax.plot(x, y3, color=P["blue_main"], lw=1.2)
        ax.set_ylim(lo - 5, hi + 5); ax.set_xlim(11.5, 23.5)
        ax.set_xticks([12, 18, 23]); ax.tick_params(labelsize=6.2, length=2)
        if i % 2:
            ax.set_yticklabels([])
        if i < 8:
            ax.set_xticklabels([])
        ax.set_title(idx[d * 24].strftime("%d %b"), fontsize=7, pad=2, loc="center", fontweight="normal")
        if d == e:
            for s_ in ax.spines.values():
                s_.set_visible(True); s_.set_color(P["ink"]); s_.set_linewidth(1.4)
    pos_d = sub[0, 0].get_position(fig)
    fig.text(pos_d.x0 - 0.03, 0.952, "(d) Import on all ten event days (MW)", fontsize=9.5, fontweight="bold", ha="left")
    handles = [h_bau, h_s3, h_cm, h_fill, *hb, h_net, h_tes, h_bat, h_nl]
    fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.0), fontsize=7.2, handlelength=1.8, columnspacing=1.2)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.915, bottom=0.17)
    finalize_figure(fig, path, tight=False)


def inflation(path, ex, names):
    """Baseline inflation on one event: (a) true and inflated load over the ten days before the event
    and the event day; (b) the event window with each baseline under inflation; (c) credited reduction
    with and without inflation, against the true reduction."""
    fig = plt.figure(figsize=(W, 3.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.25, 1.3], wspace=0.4)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    n = len(ex["load"]); t = np.arange(n) / 24
    infl = ex["load_inflated"] > ex["load"] + 1e-9
    h_hon, = a.plot(t, ex["load"], color=P["grey_mid"], lw=0.9, label="Metered load, honest")
    h_inf, = a.plot(t, np.where(infl, ex["load_inflated"], np.nan), color=P["red_strong"], lw=1.0,
                    label=f"Metered load, inflated by {100 * ex['g']:.0f}%")
    ev0 = (n - 24 + ex["hrs"][0]) / 24
    a.axvspan(ev0, ev0 + len(ex["hrs"]) / 24, color=P["highlight"], alpha=0.5, lw=0)
    a.annotate("event", (ev0, a.get_ylim()[1]), xytext=(-18, -2), textcoords="offset points", fontsize=7,
               ha="right", va="top", color=P["grey_dark"], arrowprops=dict(arrowstyle="-|>", color=P["grey_dark"], lw=0.8))
    a.set_xlim(0, n / 24); a.set_xticks(range(0, ex["G"] + 1, 2))
    a.set_xticklabels([f"$-${ex['G'] - k}" if k < ex["G"] else "0" for k in range(0, ex["G"] + 1, 2)])
    a.set_xlabel("Days before the event"); a.set_ylabel("Grid import (MW)")
    a.set_title("(a) Load before the event")
    x = np.array(ex["hrs"])
    style = {"b1": (P["red_strong"], "-"), "b2": (P["red_2"], "-"), "b3": (P["blue_main"], "--"), "b4": (P["grey_dark"], ":"),
             "b5": (P["violet"], "-.")}
    h_tr, = b.plot(x, ex["truth"], color="black", lw=1.6, label="True counterfactual")
    h_me, = b.plot(x, ex["metered"], color=P["blue_secondary"], lw=1.6, label="Metered during the event")
    hb = []
    for k, v in ex["gamed"].items():
        col, ls = style[k]
        hb.append(b.plot(x, v, color=col, ls=ls, lw=1.3, label=f"B{list(names).index(k) + 1} {names[k]}")[0])
    b.fill_between(x, ex["metered"], ex["truth"], color=P["blue_secondary"], alpha=0.15, lw=0)
    b.set_xticks(x); b.set_xticklabels([f"{h % 24:02d}:00" for h in x], fontsize=7.5)
    b.set_xlabel("Event hour"); b.set_ylabel("MW"); b.set_title("(b) Baselines under inflation")
    keys = list(ex["gamed"])
    clean = [np.mean(ex["clean"][k] - ex["metered"]) for k in keys]
    gamed = [np.mean(ex["gamed"][k] - ex["metered"]) for k in keys]
    xx = np.arange(len(keys)); wdt = 0.38
    h_cl = c.bar(xx - wdt / 2, clean, wdt, color=P["neutral"], label="Credit, honest load", **BAR)
    h_ga = c.bar(xx + wdt / 2, gamed, wdt, label="Credit, inflated load",
                 color=[P["red_strong"] if g_ > c_ + 1e-6 else P["blue_main"] for g_, c_ in zip(gamed, clean)], **BAR)
    for xi, v in zip(xx + wdt / 2, gamed):
        c.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6.8)
    h_true = c.axhline(ex["r_true"], color="black", lw=1.0, ls="--", label=f"True reduction ({ex['r_true']:.1f} MW)")
    c.set_xticks(xx); c.set_xticklabels([f"B{i + 1}" for i in range(len(keys))])
    c.set_ylabel("Credited reduction (MW)"); c.set_title("(c) Credit with and without inflation")
    from matplotlib.patches import Patch
    h_up = Patch(facecolor=P["red_strong"], edgecolor="black", linewidth=0.7, label="Credit, inflated: rises")
    h_eq = Patch(facecolor=P["blue_main"], edgecolor="black", linewidth=0.7, label="Credit, inflated: unchanged")
    handles = [h_hon, h_inf, h_tr, h_me, *hb, h_cl, h_up, h_eq, h_true]
    fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.0), fontsize=6.9, handlelength=1.8, columnspacing=1.0)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.9, bottom=0.36)
    finalize_figure(fig, path, tight=False)
