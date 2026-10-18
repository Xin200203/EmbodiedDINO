#!/usr/bin/env python3
r"""Fig. 1 -- the WHY figure: representation stability, not single-frame masks.

Run:  ~/.claude/tools/figvenv/bin/python build_fig1_motivation.py
Out:  fig1_motivation.pptx   (edit this by hand -- every label is a native run)
      fig1_motivation.pdf    (vector, exactly \textwidth)

WHAT THIS FIGURE MAY AND MAY NOT SAY
------------------------------------
Fig. 2 owns "How" (the pipeline). Fig. 1 owns "Why", and only that. So:
  * no module names, no losses, no Eq. numbers, no DINOv2/FastSAM boxes;
  * no AP, no numbers, no results;
  * the contrast is stable-vs-unstable REPRESENTATION, not DINO-vs-mask.
Any element that would let a reader reconstruct the pipeline from this figure is
a bug here, even if it is true.

THE THREE BANDS
---------------
  A  Sequential RGB-D Observations   -- the online setting (the input condition)
  B  Existing Online Methods | EmbodiedDINO  -- the mechanism (the figure's body)
  C  Object Identity Consistency     -- the consequence (the payoff)
Read top to bottom once; inside band B, top to bottom again. No other direction.

VOCABULARY IS FIXED, not stylistic. Each of these was chosen against a specific
wrong alternative, so do not "tidy" them:
  "Sequential RGB-D Observations" not "Streaming input"  -- names the setting
  O_t  not "Frame t" / "RGB_t"    -- O_t is the WHOLE observation: RGB + depth + pose
  "Mask lifting"     not "VFM masks"   -- names the mechanism, not the tool
  "Geometry-dominated"                 -- the claim is NOT that geometry is useless,
                                          it is that geometry ALONE cannot hold identity
  "Dense visual representation" not "DINO feature injection" -- no implementation here
  "Geometry + dense visual"            -- enrichment, explicitly NOT replacement
  "EmbodiedDINO"     not "Ours"        -- a paper figure names the method
  "Object Identity Consistency" not "Association"/"Tracking"/"Matching"
                                       -- those are downstream consumers of the fix
  q_t^a : t = timestep, a = object identity

ENCODINGS
---------
colour    Existing = low-saturation grey (information has been dropped);
          EmbodiedDINO = teal.  Same teal as Fig. 2's object level. Deliberately
          NOT orange: orange in Fig. 2 means "point level", and reusing it here
          would promise a mechanism this figure must not show.
shape     ● = the representation the identity actually had;  ▲ = a different
          representation for the SAME object.  So on the left the two markers are
          one colour but two shapes (same object, unstable representation), which
          is the whole claim.
line      dashed connector = identity cannot be held;  solid = it can.
Three redundant channels (colour, shape, verdict glyph) so the bottom band still
reads in grayscale and in print.

BAND A's TWO VIEWS -- provenance
--------------------------------
frames_scene0030_00/{0720,0840}.jpg -- ScanNet scene0030_00, the same scene the
existing Fig. 1 renders (README records frame 400), so the two figures show one
room. Chosen because three objects survive both frames -- the light wooden table,
the blue swivel chair, the bookshelf -- while the camera moves far enough that the
swivel chair goes from rear-three-quarter to front-left. That appearance change IS
the figure's claim, so it has to be visible in the input band, not asserted in the
caption. Frames are the raw 640x480 files, hence IW:IH = 4:3 below.

Nothing is drawn ON the photos. Ringing "object a" would be an assertion about
which instance the queries below refer to, and this figure never names one.
Set OBS_A / OBS_B to None to fall back to labelled placeholder boxes.
"""

import os
import sys

sys.path.insert(0, "/Users/xin/.claude/skills/pptx-paper-figure/scripts")
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402

from figkit import Fig, check_pdf, export_pdf, pdf_font_sizes, tex_pt  # noqa: E402

# Two views of ONE scene from different camera poses. None -> labelled placeholder.
# Relative, like every other path here: run this script from its own directory.
OBS_A = "frames_scene0030_00/0720.jpg"
OBS_B = "frames_scene0030_00/0840.jpg"

# ----------------------------------------------------------------- geometry ---
# Written out before any drawing call, so the grid is auditable without running it.
# Every band title sits centred at the top of its own band, so the three bands
# read as three registers of one column rather than as three unrelated strips.
W = tex_pt(505.89)                 # 6.9986 in -- AAAI 2027 \textwidth, exactly
H = 2.980                          # 2.740 + the 0.240 real 4:3 frames cost band A.
                                   # Kept under 3.0 on purpose: past that a
                                   # full-width figure eats a third of page 1.
M = 0.05                           # outer margin
GAP = 0.24                         # gutter between the two panels
PW = (W - 2 * M - GAP) / 2.0       # 3.3293 -- the panels are equal by construction
LX, RX = M, M + PW + GAP
LCX, RCX = LX + PW / 2.0, RX + PW / 2.0
T_H = 0.116                        # band-title row height, shared

# band A -- observations
AY0, AY1 = 0.030, 1.000
A_TITLE = 0.044
IW, IH = 0.76, 0.57                # 4:3 -- the raw 640x480 ScanNet frames, exactly.
O_Y, O_H = 0.172, 0.126            # the O_t / O_{t+1} row. 0.126 not 0.095: a
IY = 0.312                         # compensated subscript has a taller line box,
V_LBL = 0.890                      # and at 0.095 the frame below painted over it.
# 1.60, so the pair spans 3.12 in of the 6.90 band instead of 2.56 and the band
# stops reading as mostly empty. Deliberately NOT wider: the two panel centres
# below are 3.57 in apart, and a pair that spanned that far would sit one photo
# over each column -- see the note at the loop below for why that is false.
IGAP = 1.60
IAX = W / 2.0 - (2 * IW + IGAP) / 2.0
IBX = IAX + IW + IGAP

# band B -- the two panels. One row list, so both columns cannot drift apart.
PY0, PY1 = 1.085, 2.300
R_TITLE = (1.108, T_H)             # (y, h)
R_OBS = (1.256, 0.140)
R_MECH = (1.460, 0.178)
R_REPR = (1.702, 0.280)
R_QURY = (2.046, 0.178)
# Rows 2-4 are EQUAL width: unequal boxes read as "these steps differ in
# importance", which here would be a lie. Only O_t is narrow, because it is the
# input rather than a stage. Row 3 is marked by height + stroke, not by width.
W_OBS, W_STAGE = 0.48, 2.40

# band C -- identity consistency
CY0, CY1 = 2.385, 2.940
C_TITLE = 2.400
MK = 0.085                         # marker side
MOFF = 0.70                        # marker offset from the column centre -- EQUAL
Q_Y, Q_H = 2.534, 0.126
MY = 2.734                         # marker centre line; 0.031 clear of Q_Y + Q_H
V_Y, V_H = 2.804, T_H

# ------------------------------------------------------------------ palette ---
INK = "#1A1A1A"
FRAME_N = "#B9B9B9"                # Fig. 2's frame grey, reused on purpose
FRAME_O = "#A6D6D6"                # same value, teal-tinted

EX_L, EX_F, EX_T = "#9A9A9A", "#F4F4F4", "#4A4A4A"     # existing: line/fill/text
EX_A = "#333333"                                        # existing: accent word
OU_L, OU_F = "#008A8A", "#E1F4F4"                       # ours (Fig. 2 object level)
HOST_L, HOST_F = "#3E608A", "#E7EEF8"                   # the shared observation
FAINT = "#8C8C8C"                                       # annotations

PT = 8.0                           # body labels
PT_T = 8.5                         # band + panel titles
PT_S = 7.0                         # annotations

# script_pt = 0.70 x base = LaTeX \scriptstyle, so t / t+1 / a print at the size
# the body text sets them at. soffice would otherwise shrink them to 4.67 pt.
f = Fig(W, H, font="Times New Roman", base_pt=PT, ink=INK, script_pt=0.70 * PT)


def frame(x, y, w, h, line):
    """A band/panel outline. Drawn first, sent to back, never carries text."""
    s = f.box(x, y, w, h, None, fill=None, line=line, line_pt=0.6, radius=0.04)
    f.send_to_back(s)
    return s


def flow(cx, y1, y2, color):
    """One step of the vertical flow inside a panel."""
    f.arrow(cx, y1, cx, y2, color=color, line_pt=0.7, size="sm")


# =========================================================== A  observations ==
frame(M, AY0, W - 2 * M, AY1 - AY0, FRAME_N)
f.text(M, A_TITLE, W - 2 * M, T_H, "Sequential RGB-D Observations",
       size=PT_T, bold=True)

# The pair is CENTRED, not aligned to the two columns below. Aligning view A over
# the left column and view B over the right would say "each method gets one
# view", which is false and would wreck the argument -- both methods see the
# whole stream, and that is precisely why cross-view stability is the issue.
for x, sym, view, img in ((IAX, "O_t", "viewpoint A", OBS_A),
                          (IBX, "O_{t+1}", "viewpoint B", OBS_B)):
    cx = x + IW / 2.0
    f.text(cx - 0.45, O_Y, 0.90, O_H, sym, size=PT)
    if img and os.path.exists(img):
        f.image(img, x, IY, IW, IH)
        # Hairline on top of the photo, square corners. A photo with no edge floats
        # on the white page and stops reading as "one observation"; rounding the
        # corners would instead make it read as another box in the diagram.
        f.box(x, IY, IW, IH, None, fill=None, line=FRAME_N, line_pt=0.5,
              radius=None, shape=MSO_SHAPE.RECTANGLE)
    else:
        f.box(x, IY, IW, IH, "*RGB-D*", fill="#F7F7F7", line=FRAME_N,
              line_pt=0.5, radius=0.02, size=PT_S, color=FAINT)
    f.text(cx - 0.45, V_LBL, 0.90, 0.090, f"*{view}*", size=PT_S, color=FAINT)

# The two frames are ONE scene seen twice, not two frames of a long trajectory --
# hence a viewpoint-change arrow rather than a "..." continuation.
AMY = IY + IH / 2.0
f.text(W / 2.0 - 0.42, AMY - 0.104, 0.84, 0.096, "*viewpoint change*",
       size=PT_S, color=FAINT)
f.arrow(IAX + IW + 0.12, AMY, IBX - 0.12, AMY, color=FAINT, line_pt=0.7,
        size="sm")

# Same observation stream into both columns: identical arrows, identical O_t row.
for cx in (LCX, RCX):
    f.arrow(cx, AY1 + 0.012, cx, PY0 - 0.006, color=FAINT, line_pt=0.7, size="sm")

# =============================================================== B  panels ====
frame(LX, PY0, PW, PY1 - PY0, FRAME_N)
frame(RX, PY0, PW, PY1 - PY0, FRAME_O)

PANELS = (
    # cx,  line, fill, text,  title,                title colour,
    #      mechanism,          representation,                    accent
    (LCX, EX_L, EX_F, EX_T, "Existing Online Methods", "#4A4A4A",
     "Mask lifting",
     "**Geometry-dominated**\nproposal representation", EX_A),
    (RCX, OU_L, OU_F, INK, "EmbodiedDINO", OU_L,
     "Dense visual representation",
     "Geometry + **dense visual**\nproposal representation", OU_L),
)

# Both titles bold, differing only in colour: two column headers of one comparison
# should carry the same weight, or the reader reads the pair as unequal evidence
# rather than as an equal contrast with one preferred outcome.
for cx, ln, fl, tx, title, tcol, mech, repr_, acc in PANELS:
    f.text(cx - 1.55, R_TITLE[0], 3.10, R_TITLE[1], title,
           size=PT_T, bold=True, color=tcol)

    # The input is IDENTICAL on both sides -- same fill, same line, same width.
    # Any asymmetry here would read as "they solve different problems".
    f.box(cx - W_OBS / 2.0, R_OBS[0], W_OBS, R_OBS[1], "O_t",
          fill=HOST_F, line=HOST_L, line_pt=0.5, radius=0.03, size=PT)
    flow(cx, R_OBS[0] + R_OBS[1], R_MECH[0], ln)

    f.box(cx - W_STAGE / 2.0, R_MECH[0], W_STAGE, R_MECH[1], mech,
          fill=fl, line=ln, line_pt=0.5, radius=0.03, size=PT, color=tx)
    flow(cx, R_MECH[0] + R_MECH[1], R_REPR[0], ln)

    # The bottleneck (left) / the fix (right) -- the row the figure is about.
    # Marked by height and stroke weight, not by width, so the column still
    # reads as one stack.
    f.box(cx - W_STAGE / 2.0, R_REPR[0], W_STAGE, R_REPR[1], repr_,
          fill=fl, line=ln, line_pt=0.75, radius=0.03, size=PT, color=tx,
          accent=acc, wrap=False)
    flow(cx, R_REPR[0] + R_REPR[1], R_QURY[0], ln)

    f.box(cx - W_STAGE / 2.0, R_QURY[0], W_STAGE, R_QURY[1],
          "Proposal query  q_t^a",
          fill=fl, line=ln, line_pt=0.5, radius=0.03, size=PT, color=tx)

# =============================================================== C  identity ==
frame(M, CY0, W - 2 * M, CY1 - CY0, FRAME_N)
f.text(M, C_TITLE, W - 2 * M, T_H, "Object Identity Consistency",
       size=PT_T, bold=True)

# Same object a at t and t+1. Equal offsets on both sides, so the only thing the
# eye can compare is whether the two markers are the same shape.
for cx, col, shp2, dash, verdict, vbold in (
    (LCX, EX_L, MSO_SHAPE.ISOSCELES_TRIANGLE, "sysDash", "✗   Identity drift", False),
    (RCX, OU_L, MSO_SHAPE.OVAL, None, "✓   Stable identity", True),
):
    x1, x2 = cx - MOFF, cx + MOFF
    f.line(x1 + 0.075, MY, x2 - 0.075, MY, color=col, line_pt=0.6, dash=dash)
    for mx, sym, shp in ((x1, "q_t^a", MSO_SHAPE.OVAL), (x2, "q_{t+1}^a", shp2)):
        f.text(mx - 0.32, Q_Y, 0.64, Q_H, sym, size=PT, color=INK)
        f.box(mx - MK / 2.0, MY - MK / 2.0, MK, MK, None,
              fill=col, line=col, line_pt=0.4, shape=shp)
    f.text(cx - 0.85, V_Y, 1.70, V_H, verdict, size=PT, bold=vbold,
           color=col if vbold else EX_T)

# ================================================================== output ====
if __name__ == "__main__":
    pptx = f.save("fig1_motivation.pptx")
    print(f"canvas {W:.4f} x {H:.4f} in")
    pdf = export_pdf(pptx)
    print("pptx", pptx)
    print("pdf ", pdf)
    print("sizes", pdf_font_sizes(pdf))
    print("problems:", check_pdf(pdf, min_pt=6.5, expect_w_in=W,
                                min_script_pt=4.5) or "NONE")
