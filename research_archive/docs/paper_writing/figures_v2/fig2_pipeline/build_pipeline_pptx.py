#!/usr/bin/env python3
r"""Fig. 2 -- EmbodiedDINO representation pipeline, as an editable PowerPoint deck.

Port of pipeline.tex. The TikZ centimetre coordinates are KEPT as the authoring
space (X/Y/S below convert to inches at draw time), so this script stays readable
side-by-side with the .tex and either can be checked against the other.

Run:  ~/.claude/tools/figvenv/bin/python build_pipeline_pptx.py
Out:  pipeline_v2.pptx  (edit this by hand)
      pipeline_v2.pdf   (vector, exactly \textwidth -- what LaTeX includes)

The v2 suffix is deliberate: pipeline.tex / pipeline.pdf are v1's TikZ source and
output. Never write to those names from here.

Why PPT: the layout is hand-composed, so the author has to be able to nudge it.
Every label is a native text run -- no images anywhere -- so all of it is editable.

WHAT CHANGED FROM pipeline.tex (v1), and why
--------------------------------------------
1. Canvas is EXACTLY \textwidth = 505.89 TeX pt = 6.9986 in, so \includegraphics
   scales by 1.0 and the 7 pt labels are 7 pt on paper. v1 emitted 7.273 in and was
   rescaled 0.966x by LaTeX, so its 7 pt labels actually printed at 6.76 pt.
2. The two panel captions moved from BELOW the frames to INSIDE them, top-left.
   v1's bottom-right caption was centred at x=16.42 cm with ~4.3 cm of text, so its
   right edge landed at ~18.57 cm -- past the 17.98 cm frame. That, not the diagram,
   is what made v1 overflow its own stated 17.6 cm target. Moving the captions
   inside fixes the width overflow and reclaims the whole bottom row of height.
3. The query decoder is aligned to the trunk axis (y=4.86 instead of 5.24), so
   Q_t^(0) -> query decoder is a straight line rather than a kink. Same meaning,
   one less bend, and it frees a header strip in the right panel.
4. The dashed mask line now breaks around the point-level rail where it crosses it
   at (5.00, 1.05). v1's source comment claims that tap "crosses nothing"; it does
   cross the lower rail. The auxiliary line jumps, per convention.

5. Collision / stub-arrow pass, found by rendering at 300 dpi and looking:
   "P_t"+"N×3" printed as one word; "Gated Fusion" set vertically overran both the
   object box and the U-Net container; "(unchanged)" was hidden under the query
   decoder; the mask tap crossed the script-G of "Lifted masks"; the DINOv2->Lifting
   and fork->object-box connectors were shafts of 0.05-0.07 in, i.e. arrowheads
   only; and one spurious arrowhead sat mid-run in the FastSAM dashed line.
   Each is fixed at the site, with the reason in a comment.

Everything else -- palette, semantics (colour = ownership, line style = stream),
block-height-encodes-resolution, the fork, the two insertion depths -- is v1's.
"""

import sys

sys.path.insert(0, "/Users/xin/.claude/skills/pptx-paper-figure/scripts")
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402

from figkit import Fig, check_pdf, export_pdf, tex_pt  # noqa: E402

# ----------------------------------------------------------------- geometry ---
# v1 draws in cm; the frames span x 0.06..17.98, y 0.06..6.18.
# One uniform scale maps that onto \textwidth, so the aspect ratio is v1's.
W_IN = tex_pt(505.89)          # 6.9986 in -- AAAI 2027 \textwidth
LEG_H = 0.14                   # legend row, above the frames
TOP = 0.20                     # y of the frame top edge
K = (W_IN - 2 * 0.015) / 17.92  # in per cm
H_IN = TOP + 6.12 * K + 0.02   # frames + bottom margin


def X(cm):
    return 0.015 + (cm - 0.06) * K


def Y(cm):
    return TOP + (6.18 - cm) * K


def S(cm):
    return cm * K


def box_at(cx, cy, w, h):
    """v1 gives shape CENTRES; python-pptx wants the top-left corner."""
    return X(cx) - S(w) / 2.0, Y(cy) - S(h) / 2.0, S(w), S(h)


# ------------------------------------------------------------------ palette ---
# \definecolor values from pipeline.tex, verbatim. Colour encodes OWNERSHIP only.
OURSA, OURSAF = "#D66D1A", "#FDEEDE"   # ours / point level
OURSB, OURSBF = "#008A8A", "#E1F4F4"   # ours / object level
HOSTC, HOSTF = "#3E608A", "#E7EEF8"    # host
UNCHC, UNCHF = "#848484", "#F2F2F2"    # unchanged host
FROZC = "#6884C4"                      # frozen badge


def mix(c1, p, c2="#FFFFFF"):
    """LaTeX's c1!p!c2 -- p percent of c1 over c2."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(p / 100 * x + (1 - p / 100) * y):02X}" for x, y in zip(a, b))


FROZF = mix(FROZC, 12)
FAINT = mix("#000000", 45)      # black!45 -- shape/dimension annotations
UNET_L, UNET_F = mix(HOSTC, 42), mix(HOSTC, 5)
FRAME_L = mix("#808080", 55)    # gray!55

PT = 7.0                        # v1: \scriptsize at 10pt base = 7pt
# script_pt: sub/superscripts print at 4.9 pt = 0.70 x base, which is what LaTeX's
# scriptstyle does and what v1's PDF actually contains (measured: 4.98 pt). Left to
# soffice they come out at 0.58 x base = 4.09 pt, i.e. 18% smaller than the TikZ
# version -- and these carry the identifiers the reader matches to the body text
# (F_{P,t}, Q_t^{(0)}, Ẽ_t^2, o_{t,k}). PowerPoint shrinks by 0.65 rather than 0.58,
# so on screen they look ~0.5 pt larger than they will print; the PDF is the truth.
f = Fig(W_IN, H_IN, font="Times New Roman", base_pt=PT, ink="#000000", script_pt=4.9)

# ============================================================ panel frames ====
# Drawn first so everything else lands on top of them.
left = f.box(X(0.06), Y(6.18), S(14.52), S(6.12), None,
             fill=None, line=FRAME_L, line_pt=0.6, radius=0.04)
right = f.box(X(14.86), Y(6.18), S(3.12), S(6.12), None,
              fill=None, line=FRAME_L, line_pt=0.6, radius=0.04)

# Panel titles, INSIDE the frames (see header note 2).
# Both are wrap=False with hand-placed breaks, because every line here was measured
# against the space it has to fit (Times New Roman 7 pt):
#   "Representation Enrichment"  2.96 cm > 2.94 cm free before the U-Net container
#   "Scene Update (unchanged)"   2.83 cm < 2.90 cm free inside the right frame
# Left as one line would touch the container; right as three lines would leave 1 pt
# above the query decoder. Two lines each, and the renderer cannot re-break them.
f.text(X(0.22), Y(6.10), S(2.90), S(0.66), "Representation\nEnrichment",
       size=PT, bold=True, align="left", anchor="top", wrap=False)
f.text(X(15.02), Y(6.10), S(2.90), S(0.66),
       "**Instance Decoding &**\n**Scene Update** *(unchanged)*",
       size=PT, align="left", anchor="top", color=mix(UNCHC, 25, "#000000"),
       wrap=False)

# ================================================================== lane 1 ====
YAX = 4.86

# input, with a back plate to hint that it is a stream.
# Sits 0.12 cm lower than v1: the two-line panel title reaches down to y=5.53, and
# the plate's top edge was at 5.52.
f.box(X(0.30), Y(5.40), S(1.10), S(1.10), None,
      fill="#FFFFFF", line=mix(HOSTC, 45), line_pt=0.35, radius=0.03)
f.box(*box_at(0.72, 4.78, 1.10, 1.10), "posed\nRGB-D\nframe *t*",
      fill=HOSTF, line=HOSTC, line_pt=0.5, radius=0.04, size=PT)
f.text(*box_at(0.72, 3.96, 1.32, 0.30), "stream O_{1:t}",
       size=PT, italic=True, color=HOSTC)

# the U-Net container. Top at 6.10, not v1's 6.06: the header has to stack the
# container title over the group labels, and at 6.06 their Times line boxes (7.76 pt
# for 7 pt type) overlapped by 1.12 pt. The ink still cleared, but 0.04 cm buys the
# separation outright and still leaves a 0.08 cm inset from the panel frame.
unet = f.box(X(3.16), Y(6.10), S(8.06), S(2.22), None,
             fill=UNET_F, line=UNET_L, line_pt=0.5, radius=0.07)
f.send_to_back(unet)
f.send_to_back(left)
f.send_to_back(right)
f.text(X(3.26), Y(6.08), S(3.20), S(0.28), "Sparse 3D U-Net  *(host)*",
       size=PT, bold=True, align="left", anchor="middle", color=HOSTC)

# encoder / decoder: BLOCK HEIGHT ENCODES RESOLUTION (linear in log2 stride)
# Italic + faint, like the other structural annotations (N×3, N×C, J×C). Set in
# black they out-shouted the blue container title directly above them, and centred
# under it the two lines read as one stacked heading rather than as a heading plus
# a label for the four blocks below.
f.text(X(4.69) - S(1.10), Y(5.50) - S(0.32), S(2.20), S(0.30), "sparse encoder",
       size=PT, anchor="bottom", italic=True, color=FAINT)
for cx, s, h in ((3.64, 1, 1.24), (4.34, 2, 1.00), (5.04, 4, 0.76), (5.74, 8, 0.52)):
    f.box(*box_at(cx, YAX, 0.60, h), f"*{s}*", fill=HOSTF, line=HOSTC,
          line_pt=0.5, radius=0.035, size=PT)
# the stride-2 field is where object-level enrichment lands
f.box(*box_at(4.34, YAX, 0.74, 1.14), None, fill=None, line=OURSB,
      line_pt=1.0, radius=0.04)

f.box(*box_at(6.44, YAX, 0.50, 0.36), None, fill=HOSTF, line=HOSTC,
      line_pt=0.5, radius=0.03)
f.text(X(6.44) - S(0.55), Y(YAX - 0.24) - S(0.02), S(1.10), S(0.62),
       "bottle-\nneck", size=PT, anchor="top")

f.text(X(8.85) - S(1.10), Y(5.50) - S(0.32), S(2.20), S(0.30), "sparse decoder",
       size=PT, anchor="bottom", italic=True, color=FAINT)
for cx, s, h in ((7.24, 8, 0.52), (8.30, 4, 0.76), (9.38, 2, 1.00), (10.46, 1, 1.24)):
    f.box(*box_at(cx, YAX, 0.86, h), f"*{s}*", fill=HOSTF, line=HOSTC,
          line_pt=0.5, radius=0.035, size=PT)

# input -> backbone. Starts at 1.46, clear of the back plate's right edge (1.40),
# and runs flat along the trunk axis: leaving the box off-centre reads better than a
# 2 degree tilt, which just looks like a mistake.
f.arrow(X(1.46), Y(YAX), X(3.34), Y(YAX), color=HOSTC, line_pt=0.6)
f.text(*box_at(2.35, 5.09, 1.90, 0.30), "voxelize", size=PT, color=HOSTC)
# P_t and its shape, spaced apart -- at 1.40/2.00 the two boxes overlapped and
# printed as "P_tN×3".
f.text(*box_at(1.90, 4.45, 0.90, 0.30), "P_t", size=PT)
f.text(*box_at(2.72, 4.45, 1.00, 0.30), "N×3", size=PT, color=FAINT)

f.text(X(11.52) - S(0.70), Y(4.98) - S(0.30), S(1.40), S(0.30), "F_{P,t}",
       size=PT, anchor="bottom")
# 5.32, not 5.26: the compensated subscript in F_{P,t} has a taller line box than a
# plain 7 pt run, and at 5.26 the two boxes overlapped by 0.8 pt.
f.text(X(11.52) - S(0.70), Y(5.32) - S(0.30), S(1.40), S(0.30), "N×C",
       size=PT, anchor="bottom", color=FAINT)

pool = box_at(12.56, YAX, 1.46, 0.92)
f.box(*pool, "geo-aware\npooling", fill=HOSTF, line=HOSTC, line_pt=0.5,
      radius=0.04, size=PT)
f.arrow(X(10.89), Y(YAX), X(11.83), Y(YAX), color=HOSTC, line_pt=0.6)

f.box(*box_at(14.06, YAX, 1.00, 0.72), "Q_t^{(0)}", fill=OURSAF, line=OURSA,
      line_pt=0.8, radius=0.04, size=PT)
f.arrow(X(13.29), Y(YAX), X(13.56), Y(YAX), color=HOSTC, line_pt=0.6)
# Two lines: on one line "Initial queries" is 0.62 in wide, so centred over the Q
# box its right edge crossed the left frame's border at 14.58 cm.
f.text(*box_at(14.06, 5.60, 1.40, 0.62), "Initial\nqueries",
       size=PT, color=OURSA)
f.text(*box_at(14.06, 4.27, 1.40, 0.30), "J×C", size=PT, color=FAINT)

# ========================================================= unchanged head =====
# qd sits ON the trunk axis, so Q -> query decoder is straight (header note 3).
for cy, h, s in ((YAX, 0.70, "query decoder"), (3.54, 0.70, "online merger"),
                 (2.28, 0.62, "scene state M_t")):
    f.box(*box_at(16.10, cy, 2.10, h), s, fill=UNCHF, line=UNCHC, line_pt=0.5,
          radius=0.04, size=PT, color=mix(UNCHC, 30, "#000000"))

f.arrow(X(14.56), Y(YAX), X(15.05), Y(YAX), color=UNCHC, line_pt=0.6)
f.arrow(X(16.10), Y(YAX - 0.35), X(16.10), Y(3.89), color=UNCHC, line_pt=0.6)
f.text(X(16.22), Y(4.20), S(1.00), S(0.30), "Y_t", size=PT, align="left",
       color=mix(UNCHC, 25, "#000000"))
f.arrow(X(16.10), Y(3.19), X(16.10), Y(2.59), color=UNCHC, line_pt=0.6)
# per-frame recurrence -- this is what makes the setting online
f.line(X(17.15), Y(2.28), X(17.52), Y(2.28), color=UNCHC, line_pt=0.6)
f.line(X(17.52), Y(2.28), X(17.52), Y(3.54), color=UNCHC, line_pt=0.6)
f.arrow(X(17.52), Y(3.54), X(17.15), Y(3.54), color=UNCHC, line_pt=0.6)
f.vlabel(X(17.75), Y(2.91), "M_{t-1}", w=S(1.00), h=S(0.30), size=PT,
         color=mix(UNCHC, 25, "#000000"))

# ================================================================== lane 3 ====
# Row respaced so every gap is >= 0.12 in: at v1's spacing (0.12 cm between DINOv2
# and Lifting, 0.18 cm from the fork to the object box) the connectors rendered as
# bare arrowheads with no visible shaft.
f.box(*box_at(0.70, 1.73, 1.10, 0.72), "DINOv2\nF_{I,t}", fill="#FFFFFF",
      line=OURSA, line_pt=0.5, radius=0.04, size=PT)
f.box(*box_at(0.70, 1.12, 0.72, 0.30), "frozen", fill=FROZF, line=FROZC,
      line_pt=0.5, radius=0.03, size=PT, color=FROZC)

f.box(*box_at(2.12, 1.73, 1.08, 0.72), "2D-to-3D\nLifting", fill=OURSAF,
      line=OURSA, line_pt=0.5, radius=0.04, size=PT)
f.text(*box_at(2.12, 1.16, 1.40, 0.30), "(Eq. 1)", size=PT,
       color=mix(OURSA, 75, "#000000"))
f.arrow(X(1.25), Y(1.73), X(1.58), Y(1.73), color=OURSA, line_pt=1.05)
f.text(*box_at(2.85, 1.48, 1.00, 0.28), "d_{t,i}", size=PT, color=OURSA)
f.arrow(X(2.66), Y(1.73), X(2.98), Y(1.73), color=OURSA, line_pt=1.05)

# ------------- THE GRANULARITY FORK : one descriptor field, two aggregations
f.line(X(3.00), Y(1.05), X(3.00), Y(2.35), color=OURSA, line_pt=1.05)

# rail B (upper) : object level -> encoder stride 2, ONE gated write, EARLY
f.arrow(X(3.00), Y(2.35), X(3.34), Y(2.35), color=OURSB, line_pt=1.05)
f.box(*box_at(4.34, 2.35, 1.96, 1.04),
      "**Object-level Enrichment**\nMask Pooling\no_{t,k}", fill=OURSBF,
      line=OURSB, line_pt=0.5, radius=0.04, size=PT, accent=OURSB)

# rail A (lower) : point level -> four decoder scales, FOUR writes, LATE
f.arrow(X(3.00), Y(1.05), X(6.56), Y(1.05), color=OURSA, line_pt=1.05)
f.box(*box_at(8.85, 1.05, 4.48, 1.04),
      "**Point-level Enrichment**\nMulti-scale Voxelization\n"
      "{D_t^s},  s ∈ {1,2,4,8}",
      fill=OURSAF, line=OURSA, line_pt=0.5, radius=0.04, size=PT, accent=OURSA)

# ================================================================== lane 2 ====
# object level -> stride-2 encoder field : ONE write, EARLY in the encoder
# Tail on the box edge (2.87), head just below the highlight ring (4.29).
f.arrow(X(4.34), Y(2.87), X(4.34), Y(4.27), color=OURSB, line_pt=1.05)
# Both labels horizontal, one either side of the arrow. Rotated they did not fit:
# the corridor between the box top (2.87) and the U-Net container (3.84) is 0.97 cm,
# and "Gated Fusion" set vertically needs ~1.4 cm, so it ran over both.
f.text(*box_at(3.72, 3.35, 1.10, 0.52), "Gated\nFusion", size=PT, color=OURSB)
f.text(*box_at(4.80, 3.35, 1.00, 0.30), "Ẽ_t^2", size=PT,
       color=mix(OURSB, 70, "#000000"))

# point level -> four decoder scales : the four tips trace the decoder silhouette
for cx, s, yb in ((7.24, 8, 4.58), (8.30, 4, 4.46), (9.38, 2, 4.34), (10.46, 1, 4.22)):
    f.arrow(X(cx), Y(1.53), X(cx), Y(yb), color=OURSA, line_pt=1.05)
    f.box(*box_at(cx, 3.40, 0.44, 0.30), f"ψ_{s}", fill="#FFFFFF", line=None,
          shape=MSO_SHAPE.RECTANGLE, size=PT, color=OURSA)

# ================================================================== lane 0 ====
f.box(*box_at(0.74, 0.40, 1.18, 0.54), "FastSAM", fill=HOSTF, line=HOSTC,
      line_pt=0.5, radius=0.04, size=PT)
f.box(X(1.42), Y(0.55), S(0.72), S(0.30), "frozen", fill=FROZF, line=FROZC,
      line_pt=0.5, radius=0.03, size=PT, color=FROZC)
# Label placed left of the x=5.00 tap: at v1's x=3.52 the dashed tap ran straight
# through the script-G glyph. "= {G_{t,k}}" dropped -- the LaTeX equation directly
# under the figure already defines G_{t,k}.
f.text(*box_at(4.02, 0.77, 1.70, 0.30), "Lifted masks 𝒢_t",
       size=PT, color=HOSTC)

# One unbroken line: v1 split it at 2.30 with arrow(), which left a spurious
# arrowhead pointing right in the middle of the run.
f.line(X(2.14), Y(0.40), X(12.66), Y(0.40), color=HOSTC, line_pt=0.55,
       dash="sysDash")
# masks define the object-level aggregation groups. This tap crosses the
# point-level rail at y=1.05, so it breaks around it (header note 4).
f.arrow(X(5.00), Y(0.40), X(5.00), Y(0.96), color=HOSTC, line_pt=0.55,
        dash="sysDash", tail=None)
f.arrow(X(5.00), Y(1.14), X(5.00), Y(1.81), color=HOSTC, line_pt=0.55,
        dash="sysDash")
# masks also define the geo-aware pooling groups
f.arrow(X(12.56), Y(0.40), X(12.56), Y(4.33), color=HOSTC, line_pt=0.55,
        dash="sysDash")

# ================================================================== legend ====
LY = 0.02 + LEG_H / 2.0
for x0, lab, col, lw, dash in (
    (0.30, "ours / point level", OURSA, 1.05, None),
    (2.60, "ours / object level", OURSB, 1.05, None),
    (4.96, "host", HOSTC, 0.6, None),
    (6.14, "unchanged host", UNCHC, 0.6, None),
    (8.44, "lifted masks", HOSTC, 0.55, "sysDash"),
):
    f.arrow(X(x0), LY, X(x0 + 0.32), LY, color=col, line_pt=lw, dash=dash,
            tail=None)
    f.text(X(x0 + 0.38), LY - S(0.15), S(2.30), S(0.30), lab, size=PT,
           align="left")
f.box(X(10.36), LY - S(0.15), S(0.72), S(0.30), "frozen", fill=FROZF,
      line=FROZC, line_pt=0.5, radius=0.03, size=PT, color=FROZC)
f.text(X(11.32), LY - S(0.15), S(4.60), S(0.30),
       "*colour = ownership,  line style = stream*", size=PT, align="left",
       color=mix("#000000", 62))

# ================================================================== output ====
if __name__ == "__main__":
    # NOT "pipeline.pptx": export_pdf() derives the pdf name from the stem, and
    # pipeline.pdf is v1's latexmk output.
    pptx = f.save("pipeline_v2.pptx")
    print(f"canvas {W_IN:.4f} x {H_IN:.4f} in")
    pdf = export_pdf(pptx)
    print("pptx", pptx)
    print("pdf ", pdf)
    print("problems:", check_pdf(pdf, min_pt=6.5, expect_w_in=W_IN) or "NONE")
