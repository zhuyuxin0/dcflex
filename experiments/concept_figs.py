#!/usr/bin/env python3
"""Conceptual figures (context, framework, commit-reveal, B5 architecture) for the paper.

The diagrams are drawn on a canvas measured in inches: one axes fills the figure with equal scale on both axes, so
box padding, gaps and arrow positions are exact at print size. Box sizes are computed from the rendered text, every
arrow's tail and tip land exactly on the points given, and check() fails if any text comes closer than PAD to its box.
"""
import sys, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from dcflex.style import PALETTE as P, TEXT_WIDTH_IN as W, finalize_figure

BLUE, GREY, INK, TINT, RULE = P["blue_main"], P["grey_mid"], P["ink"], P["blue_tint"], "#E5E7EB"
VIO, VTINT = P["violet"], P["violet_tint"]
FS = 8.5            # diagram text (pt)
PAD = 0.09          # minimum clearance between text and box edge (in)
LW = 1.1
ARROW = dict(arrowstyle="-|>", lw=LW, shrinkA=0, shrinkB=0, mutation_scale=9)


def canvas(h):
    fig = plt.figure(figsize=(W, h))
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, W); ax.set_ylim(0, h); ax.axis("off")
    ax.boxes = []
    return fig, ax


def measure(ax, text, fs=FS, bold=False):
    """Width and height (in) of a rendered text block."""
    t = ax.text(0, 0, text, fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25)
    bb = t.get_window_extent(ax.figure.canvas.get_renderer())
    t.remove()
    return bb.width / ax.figure.dpi, bb.height / ax.figure.dpi


def box(ax, x, y, w, h, text, ec=GREY, fc="white", fs=FS, bold=False, pad=PAD):
    """Rounded box whose visible edge is exactly (x, y, w, h), with centered text at least `pad` from the edge."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", ec=ec, fc=fc, lw=LW,
                                zorder=3, clip_on=False))
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25,
                fontweight="bold" if bold else "normal", zorder=4)
    ax.boxes.append((t, (x, y, w, h), pad))


def arrow(ax, x1, y1, x2, y2, col=GREY):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(color=col, **ARROW), zorder=2)


def check(fig, ax, name):
    """Fail if any box text is closer to its box edge than the box's padding (less a small tolerance)."""
    r = fig.canvas.get_renderer()
    for t, (x, y, w, h), pad in ax.boxes:
        (x0, y0), (x1, y1) = ax.transData.inverted().transform(t.get_window_extent(r).get_points())
        clear = min(x0 - x, x + w - x1, y0 - y, y + h - y1)
        if clear < pad - 0.02:
            raise ValueError(f"{name}: text {t.get_text()!r} is {clear:.3f} in from its box edge")


def stack(ax, x, top, w, items, gap, col=GREY):
    """Boxes stacked downward from `top`, joined by arrows centered in each gap. Returns the bottom y."""
    y = top
    for i, (text, h, ec, fc, bold) in enumerate(items):
        box(ax, x, y - h, w, h, text, ec=ec, fc=fc, bold=bold)
        if i < len(items) - 1:
            arrow(ax, x + w / 2, y - h - 0.03, x + w / 2, y - h - gap + 0.03, col=col)
        y -= h + gap
    return y + gap


def context(path):
    left = [("ERCOT (ISO)\nenergy and ancillary services\nco-optimized every 5 min (RTC+B)", GREY, "white", False),
            ("Qualified scheduling entity / retail provider", GREY, "white", False),
            ("AI data center\n(flexible large load)", BLUE, TINT, True)]
    right = [("DoE Abu Dhabi (regulator)\nDR Policy: 80 MW by 2027, 200 MW by 2030", GREY, "white", False),
             ("EWEC (single buyer)\ndispatches DR as a supply resource", GREY, "white", False),
             ("Aggregator (2025 pilot phase)", GREY, "white", False),
             ("AI data center\n(flexible large load)", BLUE, TINT, True)]
    bl = ["Real-time price (15-min settlement)", "Ancillary services as Load Resource / CLR",
          "4CP transmission-charge avoidance", "SB 6: curtailment capability, new loads ≥75 MW"]
    br = ["No wholesale or ancillary-service market", "Contracted DR with capacity payment",
          "ADDC tariffs: flat 20 or 23.1 fils/kWh", "Industrial >1 MW: summer time-of-use"]
    layer = "Digitized DR layer (either route): signal → response → committed baseline → verified settlement"
    fig, ax = canvas(4.2)
    size = {t: measure(ax, t, bold=b) for t, _, _, b in left + right}
    bw = max(max(s[0] for s in size.values()) + 2 * PAD, 2.7)
    hgt = lambda t: size[t][1] + 2 * PAD
    xl, xr = W / 4 - bw / 2, 3 * W / 4 - bw / 2
    # bottom-up: layer box, bullets, then the two stacks with their data-center boxes level
    lh, fsb, line = measure(ax, layer)[1] + 2 * PAD, 8.0, 0.165
    box(ax, xl, 0.02, xr + bw - xl, lh, layer, ec=BLUE, fc=TINT)
    y_bul = 0.02 + lh + 0.16 + 3 * line
    for x, items in ((xl, bl), (xr, br)):
        for i, t in enumerate(items):
            ax.text(x + 0.04, y_bul - i * line, "• " + t, fontsize=fsb, color=P["grey_dark"], va="bottom")
    dc_bottom = y_bul + line + 0.12
    gap_r = 0.22
    top = dc_bottom + sum(hgt(t) for t, *_ in right) + gap_r * (len(right) - 1)
    gap_l = (top - dc_bottom - sum(hgt(t) for t, *_ in left)) / (len(left) - 1)
    stack(ax, xl, top, bw, [(t, hgt(t), ec, fc, b) for t, ec, fc, b in left], gap_l)
    stack(ax, xr, top, bw, [(t, hgt(t), ec, fc, b) for t, ec, fc, b in right], gap_r)
    ax.plot([W / 2, W / 2], [y_bul - 3 * line, top], color=RULE, lw=1.0, zorder=1)
    for x, t in ((W / 4, "Texas (ERCOT): flexibility priced by markets"),
                 (3 * W / 4, "Abu Dhabi: flexibility procured by a single buyer")):
        ax.text(x, top + 0.14, t, ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK)
    ax.set_ylim(0, top + 0.4); fig.set_size_inches(W, top + 0.4)
    check(fig, ax, "context")
    finalize_figure(fig, path, tight=False)


def architecture(path):
    """Study framework in three swim lanes: inputs, operation (RQ1, RQ2) and settlement (RQ3)."""
    fs = 7.8
    lanes = [("Inputs", [("Weather: ERA5 2025;\nECMWF day-ahead\nforecasts", GREY, "white"),
                         ("Workload traces\nAzure LLM,\nAlibaba jobs", GREY, "white"),
                         ("Tariffs, prices\nADDC, DEWA,\nERCOT", GREY, "white"),
                         ("System signal\nsynthetic 2030\nnet load", GREY, "white"),
                         ("Metered buildings\nBDG2, three\nhot U.S. sites", GREY, "white")]),
             ("Operation\n(RQ1, RQ2)", [("Facility LP\ndeferral, COP($\\theta$),\nTES, battery, PV", BLUE, TINT),
                                        ("Scenarios S0–S4, R\nperfect foresight\n(H1–H3)", BLUE, TINT),
                                        ("MPC from forecasts\nexpected price,\nmargin (H6)", VIO, VTINT),
                                        ("Dispatch $P^{\\mathrm{g}}_t$\nfirm reduction $F(d)$,\nvalue $\\Gamma$", BLUE, TINT)]),
             ("Settlement\n(RQ3)", [("Baselines B1–B5\nB5 physics-guided,\npretrained (H5)", VIO, VTINT),
                                     ("Commit before event\n$C=H(\\widehat{B}\\,\\|\\,h_M\\,\\|$\n$h_X\\,\\|\\,\\tau_c\\,\\|\\,n)$", BLUE, TINT),
                                     ("Reveal, verify\nand settle\n$\\Sigma(\\widehat{B}_t-P^{\\mathrm{g}}_t)$", BLUE, TINT),
                                     ("Tests: placebo, inflation,\nshopping (H4); real\nbuildings (H5)", GREY, "white")])]
    fig, ax = canvas(4.0)
    lab_w = 0.78
    x0, x1 = lab_w + 0.06, W - 0.24          # right corridor for the validation connector
    bh = max(measure(ax, t, fs=fs)[1] for _, items in lanes for t, *_ in items) + 2 * 0.07
    lane_h, lane_gap = bh + 0.3, 0.12
    top = 3.95
    centers = {}
    for li, (title, items) in enumerate(lanes):
        y_top = top - li * (lane_h + lane_gap)
        y_bot = y_top - lane_h
        ax.add_patch(FancyBboxPatch((0.02, y_bot), W - 0.04, lane_h, boxstyle="round,pad=0,rounding_size=0.05",
                                    ec="none", fc="#F7F7F8", zorder=0))
        ax.text(0.1, (y_top + y_bot) / 2, title, fontsize=8.3, fontweight="bold", color=INK, va="center",
                linespacing=1.2)
        n = len(items)
        bws = [measure(ax, t, fs=fs)[0] + 2 * 0.07 for t, *_ in items]
        gap = (x1 - x0 - sum(bws)) / (n - 1) if n > 1 else 0
        x = x0
        yb = (y_top + y_bot) / 2 - bh / 2
        for i, ((t, ec, fc), w_) in enumerate(zip(items, bws)):
            box(ax, x, yb, w_, bh, t, ec=ec, fc=fc, fs=fs, pad=0.06)
            centers[(li, i)] = (x, yb, w_, bh)
            if i < n - 1 and li > 0:            # operation and settlement are sequences
                c = min(0.04, gap / 5)
                arrow(ax, x + w_ + c, yb + bh / 2, x + w_ + gap - c, yb + bh / 2, col=INK)
            x += w_ + gap
    def link(a, b, col=GREY, fa=0.5, fb=0.5):
        """Arrow from the bottom of box a to the top of box b, at fractions fa and fb of their widths."""
        xa, ya, wa, _ = centers[a]; xb, yb_, wb, hb = centers[b]
        arrow(ax, xa + fa * wa, ya - 0.04, xb + fb * wb, yb_ + hb + 0.04, col=col)
    link((0, 0), (1, 0)); link((0, 1), (1, 1)); link((0, 2), (1, 1), fa=0.3, fb=0.8); link((0, 3), (1, 2))
    link((1, 3), (2, 2), col=INK, fa=0.3, fb=0.7)            # dispatch is metered and settled
    # metered buildings validate the baselines: elbow along the right corridor into the tests box
    xs, ys_, ws, hs = centers[(0, 4)]
    xt, yt, wt, ht = centers[(2, 3)]
    xc = W - 0.1
    ax.plot([xs + ws + 0.03, xc, xc], [ys_ + hs / 2, ys_ + hs / 2, yt + ht / 2], color=VIO, lw=LW, zorder=2,
            solid_joinstyle="miter")
    arrow(ax, xc, yt + ht / 2, xt + wt + 0.04, yt + ht / 2, col=VIO)
    # label centered on the connector, on a patch of the lane color, clear of the dispatch box
    ax.text(xc, (ys_ + yt + ht) / 2 - 0.1, "validation", fontsize=7, color=VIO, ha="center", va="center",
            rotation=90, zorder=3, bbox=dict(facecolor="#F7F7F8", edgecolor="none", pad=1.2))
    lo = top - 3 * lane_h - 2 * lane_gap - 0.03
    ax.set_ylim(lo, top + 0.03); fig.set_size_inches(W, top + 0.03 - lo)
    check(fig, ax, "architecture")
    finalize_figure(fig, path, tight=False)


def commit_reveal(path):
    lanes = {"Data center": 0.95, "Grid operator (EWEC)": 3.55, "Public log": 5.65}
    rows = [("Data center", "Public log", "$\\tau_c$: publish $C = H(\\widehat{B}\\,\\|\\,h_M\\,\\|\\,h_X\\,\\|\\,\\tau_c\\,\\|\\,n)$", BLUE),
            ("Data center", "Grid operator (EWEC)", "send $C$ with the DR offer", BLUE),
            ("Grid operator (EWEC)", "Data center", "event: dispatch instruction", GREY),
            ("Data center", "Grid operator (EWEC)", "metered response $P^{\\mathrm{g}}_t$", GREY),
            ("Data center", "Grid operator (EWEC)", "after event: reveal $(\\widehat{B}, M, X, \\tau_c, n)$", BLUE)]
    verify = ("verify $H(\\cdot)=C$; recompute $\\widehat{B}=M(X)$;\n"
              "settle $\\Sigma_{t\\in\\mathcal{W}}\\,(\\widehat{B}_t-P^{\\mathrm{g}}_t)$")
    fig, ax = canvas(3.2)
    vw, vh = measure(ax, verify)
    vw, vh = vw + 2 * PAD, vh + 2 * PAD
    step, top = 0.44, 3.2
    y_rows = [top - 0.62 - i * step for i in range(len(rows))]
    vy = y_rows[-1] - 0.3 - vh
    for name, x in lanes.items():
        ax.text(x, top - 0.08, name, ha="center", va="top", fontsize=9, fontweight="bold", color=INK)
        ax.plot([x, x], [vy - 0.05, top - 0.34], color=RULE, lw=1.2, zorder=1)
    for y, (a, b, t, col) in zip(y_rows, rows):
        arrow(ax, lanes[a], y, lanes[b], y, col=col)       # tail and tip exactly on the lifelines
        ax.text((lanes[a] + lanes[b]) / 2, y + 0.07, t, ha="center", va="bottom", fontsize=FS, color=INK)
    box(ax, lanes["Grid operator (EWEC)"] - vw / 2, vy, vw, vh, verify, ec=BLUE, fc=TINT)
    ax.set_ylim(vy - 0.08, top); fig.set_size_inches(W, top - vy + 0.08)
    check(fig, ax, "commit_reveal")
    finalize_figure(fig, path, tight=False)


def mini_curve(ax, x, y, w, h, kind, col):
    """A small sketch of a response curve inside a box region (x, y, w, h), in inches."""
    import numpy as np
    t = np.linspace(0, 1, 50)
    f = {"convex": 0.08 + 0.9 * t ** 2.2, "flat": 0.5 + 0 * t}[kind]
    ax.plot([x, x + w], [y, y], color=GREY, lw=0.6, zorder=4)
    ax.plot([x, x], [y, y + h], color=GREY, lw=0.6, zorder=4)
    ax.plot(x + t * w, y + f * h, color=col, lw=1.3, zorder=4)


def b5(path):
    """B5: the physics-guided baseline network, its cross-building pretraining and its commitment."""
    fs = 8.0
    fig, ax = canvas(3.6)
    top = 3.45
    ins = ["Calendar $\\mathbf{c}_t$\n(hour, weekday)", "Dry bulb $\\theta_t$", "Dew point $\\theta^{\\mathrm{d}}_t$"]
    iw = max(measure(ax, t, fs=fs)[0] for t in ins) + 2 * PAD
    ih = [measure(ax, t, fs=fs)[1] + 2 * PAD for t in ins]
    blocks = [("Base network $b(\\mathbf{c})$\nMLP, positive output", False),
              ("Cooling response $g(\\theta)$\n$\\Sigma_k w_k\\,\\mathrm{softplus}$, $w_k\\geq 0$", True),
              ("Weather gains\n$h(\\theta)$, $h^{\\mathrm{d}}(\\theta^{\\mathrm{d}})$", True)]
    curve_w = 0.42
    tw = max(measure(ax, t, fs=fs)[0] for t, _ in blocks) + 2 * PAD
    bw = tw + curve_w + 0.1
    bh = max(measure(ax, t, fs=fs)[1] for t, _ in blocks) + 2 * PAD
    gap = 0.14
    x_in, x_bl = 0.03, 0.03 + iw + 0.42
    ys = [top - (i + 1) * bh - i * gap for i in range(len(blocks))]
    for i, (t, curve) in enumerate(blocks):
        box(ax, x_bl, ys[i], bw, bh, t, ec=VIO, fc=VTINT, fs=fs)
        if curve:          # text left, sketch of the monotone convex response right
            ax.boxes[-1][0].set_x(x_bl + tw / 2)
            mini_curve(ax, x_bl + tw, ys[i] + 0.1, curve_w, bh - 0.2, "convex", VIO)
        yc_i = ys[i] + bh / 2
        box(ax, x_in, yc_i - ih[i] / 2, iw, ih[i], ins[i], fs=fs)
        arrow(ax, x_in + iw + 0.04, yc_i, x_bl - 0.04, yc_i, col=INK)
    comb = "$\\widehat{B}_t=b\\,(1+g)+h+h^{\\mathrm{d}}$\n$+\\,c_0-\\beta\\,\\mathrm{PV}_t$"
    cw, ch = measure(ax, comb, fs=fs); cw += 2 * PAD; ch += 2 * PAD
    x_c = x_bl + bw + 0.45
    yc = ys[1] + bh / 2
    box(ax, x_c, yc - ch / 2, cw, ch, comb, ec=BLUE, fc=TINT, fs=fs)
    for i in range(len(blocks)):
        y0 = ys[i] + bh / 2
        arrow(ax, x_bl + bw + 0.04, y0, x_c - 0.04, yc + (y0 - yc) * 0.3, col=INK)
    pv = "PV potential\n$\\beta\\in[0,1]$"
    pw_, ph_pv = measure(ax, pv, fs=fs); pw_ += 2 * PAD; ph_pv += 2 * PAD
    y_pv = ys[2] - 0.05
    box(ax, x_c + cw / 2 - pw_ / 2, y_pv - ph_pv + bh / 2 - 0.12, pw_, ph_pv, pv, fs=fs)
    arrow(ax, x_c + cw / 2, y_pv - ph_pv + bh / 2 - 0.12 + ph_pv + 0.04, x_c + cw / 2, yc - ch / 2 - 0.04, col=INK)
    commit = "Ensemble of 5 nets\n$\\rightarrow$ 64-bit weights\n$\\rightarrow$ SHA-256 $=h_M$\nin commitment $C$"
    kw_, kh_ = measure(ax, commit, fs=fs); kw_ += 2 * PAD; kh_ += 2 * PAD
    x_k = W - 0.03 - kw_
    box(ax, x_k, yc - kh_ / 2, kw_, kh_, commit, ec=BLUE, fc=TINT, fs=fs)
    arrow(ax, x_c + cw + 0.04, yc, x_k - 0.04, yc, col=INK)
    # cross-building pretraining (bottom band): embedding -> hypernetwork -> hinge weights of the violet blocks
    band_top = ys[2] - 0.5
    pre = [("Building embedding\n$\\mathbf{e}_i\\in\\mathbb{R}^{16}$", VIO, VTINT),
           ("Hypernetwork (linear)\n$\\mathbf{e}_i\\mapsto a_k$ of $g$, $h$, $h^{\\mathrm{d}}$", VIO, VTINT)]
    note = "Pretrained on the other BDG2 sites;\na new facility fits only $\\mathbf{e}$ (16 numbers)\non its 60 pre-season days"
    pw = [measure(ax, t, fs=fs)[0] + 2 * PAD for t, *_ in pre]
    ph_ = max(measure(ax, t, fs=fs)[1] for t, *_ in pre) + 2 * PAD
    x_h = x_bl + bw / 2 - pw[1] / 2                     # hypernetwork centered under the violet column
    x_e = x_in
    for (t, ec, fc), x, w_ in zip(pre, (x_e, x_h), pw):
        box(ax, x, band_top - ph_, w_, ph_, t, ec=ec, fc=fc, fs=fs)
    yb = band_top - ph_ / 2
    arrow(ax, x_e + pw[0] + 0.04, yb, x_h - 0.04, yb, col=VIO)
    xh = x_bl + bw / 2
    arrow(ax, xh, band_top + 0.04, xh, ys[2] - 0.04, col=VIO)
    ax.text(xh + 0.06, (band_top + ys[2]) / 2, "hinge weights", fontsize=7.2, color=VIO, va="center")
    nw, nh = measure(ax, note, fs=fs)
    ax.text(W - 0.03, yb, note, fontsize=fs, color=P["grey_dark"], ha="right", va="center", linespacing=1.25)
    ax.text(x_in, top + 0.08, "Structured network (physics as architecture)", fontsize=8.5, fontweight="bold",
            color=INK, va="bottom")
    ax.text(x_in, band_top + 0.06, "Cross-building pretraining", fontsize=8.5, fontweight="bold", color=INK,
            va="bottom")
    lo = band_top - ph_ - 0.04
    ax.set_ylim(lo, top + 0.3); fig.set_size_inches(W, top + 0.3 - lo)
    check(fig, ax, "b5")
    finalize_figure(fig, path, tight=False)


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "../figures"); out.mkdir(parents=True, exist_ok=True)
    context(out / "fig-context.pdf"); architecture(out / "fig-architecture.pdf"); commit_reveal(out / "fig-commit-reveal.pdf")
    b5(out / "fig-b5.pdf")
    # fig-cop is written by run_all.py (it adds the 2025 temperature histogram); not redrawn here
    print("concept figures written to", out)
