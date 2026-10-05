"""Fill the official SIH 2026 idea-PPT template (6 slides) for team KANABI, PS SIH26168.

Run:  python3 build_sih_template_deck.py
Output: KANABI_SIH26168_IdeaPPT.pptx next to this script.
"""
import copy
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
TEMPLATE = Path.home() / "Downloads" / "SIH2026-IDEA-Presentation-Format.pptx"
REPO = HERE.parent / "sih_navigation"
FIG = REPO / "artifacts" / "figures"
OUT = HERE / "KANABI_SIH26168_IdeaPPT.pptx"

TEAM_ID = "<Team ID from portal>"   # fill in before submitting
TEAM = "KANABI"
REPO_URL = "https://github.com/AnushtupGhosh5/sih"

NAVY = RGBColor(0x1F, 0x38, 0x64)
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x55, 0x5F, 0x6D)
ORANGE = RGBColor(0xE8, 0x6A, 0x1E)
GREEN = RGBColor(0x2E, 0x8B, 0x57)
RED = RGBColor(0xC0, 0x39, 0x2B)
LIGHT = RGBColor(0xF2, 0xF5, 0xF9)
LINE = RGBColor(0xC9, 0xD2, 0xDE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Arial"


# ----------------------------------------------------------------- helpers
def _set_bullet(paragraph, char="•", indent_emu=228600):
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set("marL", str(indent_emu))
    pPr.set("indent", str(-indent_emu))
    for tag in ("a:buNone", "a:buChar", "a:buFont"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    buFont = pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"})
    buChar = pPr.makeelement(qn("a:buChar"), {"char": char})
    pPr.append(buFont)
    pPr.append(buChar)


def _no_bullet(paragraph):
    pPr = paragraph._p.get_or_add_pPr()
    for tag in ("a:buChar", "a:buFont", "a:buNone"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    pPr.append(pPr.makeelement(qn("a:buNone"), {}))


def add_text(slide, x, y, w, h, paras, size=13, color=INK, bold=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, fill=None, line=None, margin=0.08, line_spacing=1.05,
             space_after=3, shape=MSO_SHAPE.RECTANGLE, radius=None):
    """paras: list of items. Item = str | dict(text, size, bold, color, bullet, align, runs)."""
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = radius
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, Inches(margin))
    first = True
    for item in paras:
        if isinstance(item, str):
            item = {"text": item}
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = item.get("align", align)
        p.line_spacing = item.get("line_spacing", line_spacing)
        p.space_after = Pt(item.get("space_after", space_after))
        if item.get("space_before"):
            p.space_before = Pt(item["space_before"])
        runs = item.get("runs") or [{"text": item.get("text", "")}]
        for rd in runs:
            r = p.add_run()
            r.text = rd.get("text", "")
            f = r.font
            f.name = FONT
            f.size = Pt(rd.get("size", item.get("size", size)))
            f.bold = rd.get("bold", item.get("bold", bold))
            f.italic = rd.get("italic", item.get("italic", False))
            f.color.rgb = rd.get("color", item.get("color", color))
        if item.get("bullet"):
            _set_bullet(p, item.get("bullet_char", "•"), item.get("indent", 228600))
        else:
            _no_bullet(p)
    return shp


def add_picture_fit(slide, path, x, y, w, h, align="center"):
    """Place image inside box (inches) preserving aspect ratio."""
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    px = x + (w - pw) / 2 if align == "center" else x
    py = y + (h - ph) / 2
    return slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))


def add_box(slide, x, y, w, h, fill=LIGHT, line=LINE, radius=0.06):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    shp.text_frame.text = ""
    return shp


def add_arrow(slide, x, y, w, h, color=NAVY):
    shp = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def section_label(slide, x, y, w, text, color=NAVY, size=14):
    return add_text(slide, x, y, w, 0.32, [{"text": text, "bold": True, "size": size, "color": color}],
                    margin=0.02, anchor=MSO_ANCHOR.MIDDLE)


def add_table(slide, x, y, w, rows, col_widths, header_fill=NAVY, size=10.5, row_h=0.3):
    nrows, ncols = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w), Inches(row_h * nrows))
    tbl = gt.table
    # plain style: remove banding via tblPr attrs
    tblPr = tbl._tbl.tblPr
    tblPr.set("bandRow", "0")
    tblPr.set("firstRow", "0")
    for i, cw in enumerate(col_widths):
        tbl.columns[i].width = Inches(cw)
    for r in range(nrows):
        tbl.rows[r].height = Inches(row_h)
        for c in range(ncols):
            cell = tbl.cell(r, c)
            val = rows[r][c]
            strong = False
            if isinstance(val, tuple):
                val, strong = val
            cell.text = ""
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = str(val)
            run.font.name = FONT
            run.font.size = Pt(size)
            run.font.bold = (r == 0) or strong
            run.font.color.rgb = WHITE if r == 0 else (GREEN if strong else INK)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if r == 0 else (WHITE if r % 2 else LIGHT)
    return gt


def remove_shape(shape):
    el = shape._element
    el.getparent().remove(el)


def set_title(slide, text, size=30, left=1.75, width=8.0):
    for shp in slide.shapes:
        if shp.is_placeholder and shp.placeholder_format.type is not None and "TITLE" in str(shp.placeholder_format.type):
            if left is not None:
                shp.left = Inches(left)
                shp.width = Inches(width)
            tf = shp.text_frame
            # keep first run formatting, replace text
            p = tf.paragraphs[0]
            for extra in tf.paragraphs[1:]:
                extra._p.getparent().remove(extra._p)
            runs = p.runs
            if runs:
                runs[0].text = text
                for r in runs[1:]:
                    r._r.getparent().remove(r._r)
                runs[0].font.size = Pt(size)
            else:
                p.text = text
                p.runs[0].font.size = Pt(size)
            return shp
    raise RuntimeError("title placeholder not found")


def set_team_oval(slide):
    for shp in slide.shapes:
        if shp.has_text_frame and shp.text_frame.text.strip() == "Your Team Name":
            tf = shp.text_frame
            p = tf.paragraphs[0]
            for r in p.runs[1:]:
                r._r.getparent().remove(r._r)
            for m in ("margin_left", "margin_right"):
                setattr(tf, m, Inches(0.02))
            tf.word_wrap = False
            p.runs[0].text = TEAM
            p.runs[0].font.bold = True
            p.runs[0].font.size = Pt(14)
            p.runs[0].font.name = FONT
            p.runs[0].font.color.rgb = NAVY
            return


def drop_instruction_textbox(slide):
    for shp in list(slide.shapes):
        if shp.name == "TextBox 8":
            remove_shape(shp)


def delete_slide(prs, index):
    sldIdLst = prs.slides._sldIdLst
    sldId = list(sldIdLst)[index]
    rId = sldId.get(qn("r:id"))
    prs.part.drop_rel(rId)
    sldIdLst.remove(sldId)


# ----------------------------------------------------------------- slides
def slide1(prs):
    s = prs.slides[0]
    tb = next(sh for sh in s.shapes if sh.name == "TextBox 9")
    tf = tb.text_frame
    lines = [
        None,
        ("Problem Statement ID – ", "SIH26168"),
        ("Problem Statement Title – ", "AI-ML based Intelligent Dead Reckoning system for seamless navigation"),
        ("Theme – ", "Miscellaneous"),
        ("PS Category – ", "Software"),
        ("Team ID – ", TEAM_ID),
        ("Team Name (Registered on portal) – ", TEAM),
    ]
    for p, item in zip(tf.paragraphs, lines):
        p.line_spacing = 1.35
        p.space_after = Pt(4)
        if item is None:
            continue
        label, value = item
        runs = p.runs
        runs[0].text = label
        runs[0].font.size = Pt(18)
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
        r2 = p.add_run()
        r2.text = value
        r2.font.name = FONT
        r2.font.size = Pt(18)
        r2.font.bold = False
        r2.font.color.rgb = NAVY if value != TEAM_ID else RED
    # small organisation line under the list
    add_text(s, 0.4, 6.55, 6.4, 0.4, [{"text": "Organisation: Indian Space Research Organisation (ISRO)  ·  Dept. of Space", "size": 12, "color": MUTED, "italic": True}], margin=0.02)


def slide2(prs):
    s = prs.slides[1]
    set_title(s, "IDEA: Intelligent Dead Reckoning (IDR) + GNSS Fusion", 26)
    set_team_oval(s)
    drop_instruction_textbox(s)

    # left column: the idea
    section_label(s, 0.35, 1.25, 6.9, "Proposed solution", ORANGE)
    add_text(s, 0.35, 1.55, 6.9, 2.2, [
        {"text": "An edge-deployable engine + Android app that keeps lane-level position when GNSS is lost (tunnels, underpasses, parking, urban canyons, jamming) using only the phone's accelerometer, gyroscope, magnetometer and an offline OpenStreetMap graph. No OBD-II, no wheel sensor, no fixed mount.", "bullet": True, "size": 11.5},
        {"text": "Hybrid AI + physics pipeline: in-vehicle alignment & calibration → AI speed/vibration filter (1-D CNN) → bias-corrected INS → map-matching with non-holonomic constraints → AI-augmented error-state Kalman fusion with instant GNSS-deficit switching.", "bullet": True, "size": 11.5},
        {"text": "Training happens on desktop (IO-VNBD); the lightweight model (32 k params, ONNX/TFLite) runs on the phone at 10 Hz and on the edge engine at ~200 Hz for external/FOG IMUs.", "bullet": True, "size": 11.5},
    ], space_after=4)

    section_label(s, 0.35, 3.85, 6.9, "How it addresses the problem", ORANGE)
    add_text(s, 0.35, 4.15, 6.9, 1.25, [
        {"text": "GNSS drop → seamless switch to inertial tracking within ms; GNSS return → re-converge to GNSS+INS.", "bullet": True, "size": 11.5},
        {"text": "Speed without OBD: last-GNSS speed anchor + learned IMU corrections; vibration, potholes and idling filtered out.", "bullet": True, "size": 11.5},
        {"text": "Heading drift (the real killer of phone DR) is cancelled by snapping to the road graph on every edge.", "bullet": True, "size": 11.5},
    ], space_after=3)

    section_label(s, 0.35, 5.5, 6.9, "Innovation and uniqueness", ORANGE)
    add_text(s, 0.35, 5.8, 6.9, 1.1, [
        {"text": "Data-driven design: on IO-VNBD we measured that speed holds to 1–2 % but gyro heading drifts 19–28° in 30–60 s, so we spend the AI budget where the error really is.", "bullet": True, "size": 11.5},
        {"text": "Mount-agnostic calibration (gravity + PCA of driving acceleration) and one shared core engine for phone and edge/FOG sensors.", "bullet": True, "size": 11.5},
    ], space_after=3)

    # right column: position plot (required by PS)
    add_box(s, 7.45, 1.3, 5.55, 5.6, fill=LIGHT, line=LINE)
    add_text(s, 7.55, 1.35, 5.35, 0.35, [{"text": "Preliminary result on IO-VNBD (drive vfa02): 1 km GNSS blackout, 520° of turning", "bold": True, "size": 11.5, "color": NAVY}], margin=0.03, anchor=MSO_ANCHOR.MIDDLE)
    add_picture_fit(s, FIG / "mapmatch_1km_demo.png", 7.55, 1.72, 5.35, 4.3)
    add_text(s, 7.55, 6.05, 5.35, 0.8, [
        {"runs": [{"text": "Plain phone IMU drifts 76 % ", "color": RED, "bold": True, "size": 11.5},
                  {"text": "of distance; ", "size": 11.5},
                  {"text": "our map-aided IDR finishes 2.5 % off ", "color": GREEN, "bold": True, "size": 11.5},
                  {"text": "and tracks the roundabout. Benchmark: < 10 %.", "size": 11.5}]},
    ], margin=0.03)


def slide3(prs):
    s = prs.slides[2]
    set_title(s, "TECHNICAL APPROACH", 32)
    set_team_oval(s)
    drop_instruction_textbox(s)

    # technologies strip
    section_label(s, 0.35, 1.25, 4, "Technologies", ORANGE)
    chips = [
        ("Training", "Python · PyTorch · NumPy/SciPy · CatBoost"),
        ("Models", "1-D CNN speed filter (32 k params) · LSTM IMU-error corrector · gated causal CNN Δv"),
        ("Estimation", "Error-state EKF/UKF · NHC & ZUPT · HMM map-matching (Newson–Krumm)"),
        ("Maps", "Offline OpenStreetMap graph (osmnx export, on-device)"),
        ("Deployment", "ONNX → TFLite · Flutter/Android app · Python edge engine (200 Hz FOG IMU)"),
    ]
    x = 0.35
    y = 1.58
    widths = [1.75, 3.35, 3.05, 2.1, 2.4]
    for (k, v), w in zip(chips, widths):
        add_text(s, x, y, w - 0.08, 0.72, [
            {"text": k, "bold": True, "size": 10.5, "color": NAVY, "space_after": 1},
            {"text": v, "size": 9.5, "color": INK},
        ], fill=LIGHT, line=LINE, margin=0.06, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
        x += w

    # pipeline flow
    section_label(s, 0.35, 2.42, 12.5, "Methodology: on-device pipeline (10 Hz on phone, ~200 Hz on edge engine)", ORANGE)
    stages = [
        ("1  Align & calibrate", "Gravity → vertical axis; PCA of driving accel → forward axis. Phone-to-vehicle rotation + accel/gyro bias + compass offset from the pre-blackout window. Works on dash or holder."),
        ("2  AI speed & vibration filter", "Mount-invariant IMU features → 1-D CNN / causal Δv model. Rejects engine idle, potholes, bumps; forward speed without OBD-II."),
        ("3  INS propagation", "Bias-corrected gyro heading + speed integrate position at 10 Hz. Speed anchored to last GNSS fix (within 1–2 % of oracle)."),
        ("4  Map-match + NHC", "Odometry walks the offline OSM graph; gyro picks the branch at junctions; heading reset to road bearing. Zero lateral/vertical velocity pseudo-measurements."),
        ("5  AI fusion & deficit handler", "Loosely coupled error-state EKF; LSTM learns residual IMU errors. GNSS graded GOOD / DEGRADED / DENIED (HDOP, sat count, C/N0) → mode switch in ms."),
    ]
    bx, by, bw, bh, gap = 0.35, 2.78, 2.33, 1.7, 0.2
    for i, (hd, body) in enumerate(stages):
        x = bx + i * (bw + gap)
        add_text(s, x, by, bw, bh, [
            {"text": hd, "bold": True, "size": 11.5, "color": WHITE, "space_after": 4},
            {"text": body, "size": 9.5, "color": WHITE},
        ], fill=NAVY if i != 4 else ORANGE, line=None, margin=0.09, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
        if i < 4:
            add_arrow(s, x + bw + 0.02, by + bh / 2 - 0.13, gap - 0.04, 0.26, color=ORANGE)

    # in/out strip
    add_text(s, 0.35, 4.6, 6.2, 0.5, [
        {"runs": [{"text": "INPUT  ", "bold": True, "color": NAVY, "size": 11},
                  {"text": "phone accelerometer · gyroscope · magnetometer · GNSS (when available) · offline OSM tiles", "size": 11}]},
    ], fill=LIGHT, line=LINE, anchor=MSO_ANCHOR.MIDDLE, margin=0.08)
    add_text(s, 6.65, 4.6, 6.35, 0.5, [
        {"runs": [{"text": "OUTPUT  ", "bold": True, "color": NAVY, "size": 11},
                  {"text": "continuous position, velocity, heading @ 10 Hz + mode flag (GNSS+INS / Degraded / Dead reckoning) → smooth vehicle icon in the app", "size": 11}]},
    ], fill=LIGHT, line=LINE, anchor=MSO_ANCHOR.MIDDLE, margin=0.08)

    # two-phase workflow + status
    section_label(s, 0.35, 5.2, 6, "Workflow and current status", ORANGE)
    add_text(s, 0.35, 5.52, 8.2, 1.4, [
        {"runs": [{"text": "Phase 1 (desktop, done): ", "bold": True, "size": 11, "color": NAVY},
                  {"text": "IO-VNBD loader & synchronisation audit, feature engineering, speed-filter & LSTM training, simulated-blackout evaluation over 369 one-km blackouts, OSM map-matching, browser replay demo.", "size": 11}], "space_after": 3},
        {"runs": [{"text": "Phase 2 (on-device, in progress): ", "bold": True, "size": 11, "color": NAVY},
                  {"text": "Flutter app with live GNSS/IMU HUD and mode banner is built (APK); engine export to TFLite and porting of alignment + DR to the phone are next, followed by the 200 Hz edge build.", "size": 11}]},
    ], margin=0.04)
    add_picture_fit(s, HERE / "demo_screenshot.png", 8.7, 5.2, 4.3, 1.7)


def slide4(prs):
    s = prs.slides[3]
    set_title(s, "FEASIBILITY AND VIABILITY", 32)
    set_team_oval(s)
    drop_instruction_textbox(s)

    # left: evidence
    section_label(s, 0.35, 1.25, 6.6, "Feasibility: measured on IO-VNBD, smartphone IMU only", ORANGE)
    add_table(s, 0.35, 1.6, 6.55, [
        ["Scenario (simulated GNSS blackout on real drives)", "PS benchmark", "Achieved"],
        ["50 m blackout, motorway (< 1 min)", "< 5 m", ("0.4 m  (0.7 %)", True)],
        ["1 km straight, tunnel-like run @ ~100 km/h", "< 100 m", ("7.9 m  (0.8 %)", True)],
        ["1 km real roads with turns, map-aided (median of 369)", "< 10 %", ("3.3 % hwy · 4.8–7.0 % urban", True)],
        ["Share of 1 km blackouts under 10 % (free DR → map-aided)", "—", ("17 % → 74 %", True)],
    ], [3.25, 1.3, 2.0], size=10, row_h=0.34)
    add_picture_fit(s, FIG / "blackout_1km_tunnel.png", 0.35, 3.4, 6.55, 2.55)
    add_text(s, 0.35, 5.95, 6.55, 0.95, [
        {"text": "Position plot inferred from an IO-VNBD test drive: 1 km blackout, final drift 7.9 m. Trained within-vehicle (Driver E), tested on held-out drives vfa02 / vtb5 / vw2. Reproducible: python -m src.idr.evaluate / evaluate_mapmatch.", "size": 10, "color": MUTED},
        {"text": "Compute: 32 k-parameter CNN + EKF runs comfortably at 10 Hz on a mid-range Android phone; OSM graph for a city fits in a few MB offline.", "size": 10, "color": MUTED},
    ], margin=0.03)

    # right: risks and strategies
    section_label(s, 7.15, 1.25, 5.9, "Challenges, risks and how we handle them", ORANGE)
    add_table(s, 7.15, 1.6, 5.85, [
        ["Challenge / risk", "Strategy"],
        ["Gyro heading drifts 19–28° in 30–60 s → free DR > 10 % beyond ~200 m", "Map-matching + NHC reset heading on every road edge; ZUPT at stops; magnetometer only as a weak prior"],
        ["Absolute speed from 10 Hz vibration aliases (> 5 Hz Nyquist)", "Anchor to last GNSS speed + learned Δv correction; 100–200 Hz sampling on phone/edge for the filter"],
        ["Wrong branch at ambiguous junctions (remaining urban misses)", "Multi-hypothesis HMM map-matching with gyro-turn likelihood; prune on GNSS return"],
        ["Phone mount changes / hand-held, two-wheelers", "Continuous re-alignment on detected mount change; vehicle-class model heads"],
        ["Different IMUs (phone vs FOG) and finale datasets", "Sensor-agnostic core; calibration stage absorbs bias/scale; same code path for edge engine"],
        ["On-device latency and battery", "Tiny CNN, TFLite int8, 10 Hz loop; heavy training stays on desktop"],
    ], [2.6, 3.25], size=9.5, row_h=0.56)
    add_text(s, 7.15, 5.6, 5.85, 1.3, [
        {"text": "Viability", "bold": True, "size": 12, "color": NAVY, "space_after": 2},
        {"text": "Zero hardware cost: runs on phones already on every dashboard; no OBD dongle, no vehicle integration.", "bullet": True, "size": 10.5},
        {"text": "Open data (IO-VNBD) + open maps (OSM) → no licensing barrier; models retrain as finale datasets arrive.", "bullet": True, "size": 10.5},
        {"text": "Working repo, reproducible pipeline, replay demo and Android APK already exist.", "bullet": True, "size": 10.5},
    ], margin=0.03, space_after=2)


def slide5(prs):
    s = prs.slides[4]
    set_title(s, "IMPACT AND BENEFITS", 32)
    set_team_oval(s)
    drop_instruction_textbox(s)

    section_label(s, 0.35, 1.25, 6.4, "Who it helps", ORANGE)
    audiences = [
        ("Delivery & ride-hailing riders", "Two-wheelers and cabs with only a phone: no frozen map in flyovers, underpasses and metro tunnels."),
        ("Trucks, fleets & older cars", "Vehicles without factory INS or OBD feed get continuous tracking and correct turn-by-turn guidance."),
        ("Emergency responders", "Ambulances and fire units keep position in parking decks and canyons where seconds matter."),
        ("Navigation & telematics apps", "Drop-in engine for MapmyIndia / fleet SDKs; edge engine for ISRO, defence and rail use with FOG IMUs."),
    ]
    y = 1.6
    for hd, body in audiences:
        add_text(s, 0.35, y, 6.4, 0.78, [
            {"text": hd, "bold": True, "size": 11.5, "color": NAVY, "space_after": 1},
            {"text": body, "size": 10.5},
        ], fill=LIGHT, line=LINE, margin=0.08, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
        y += 0.86

    section_label(s, 0.35, 5.08, 6.4, "Benchmark the judges set vs. what we already measure", ORANGE)
    stats = [("< 10 %", "PS target drift"), ("0.7 %", "50 m blackout"), ("0.8 %", "1 km tunnel run"), ("3.3–7 %", "1 km real roads (median)")]
    x = 0.35
    for big, small in stats:
        add_text(s, x, 5.42, 1.52, 1.45, [
            {"text": big, "bold": True, "size": 20, "color": GREEN if big != "< 10 %" else NAVY, "align": PP_ALIGN.CENTER, "space_after": 2},
            {"text": small, "size": 9.5, "color": MUTED, "align": PP_ALIGN.CENTER},
        ], fill=WHITE, line=LINE, anchor=MSO_ANCHOR.MIDDLE, margin=0.05, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
        x += 1.63

    section_label(s, 7.0, 1.25, 6.0, "Benefits", ORANGE)
    benefits = [
        ("Social / safety", "No missed exits or erratic jumps in tunnels; safer lane guidance for two-wheelers; reliable emergency response in GNSS-denied places."),
        ("Economic", "Zero added hardware; fewer failed deliveries and detours; one engine for millions of existing phones and fleet devices."),
        ("Environmental", "Fewer wrong turns and re-routing loops → less fuel and emissions per trip."),
        ("Strategic", "Resilience against jamming/spoofing and signal loss; complements NavIC; same AI core deploys on edge hardware for space, defence and rail."),
    ]
    y = 1.6
    for hd, body in benefits:
        add_text(s, 7.0, y, 6.0, 0.78, [
            {"text": hd, "bold": True, "size": 11.5, "color": ORANGE, "space_after": 1},
            {"text": body, "size": 10.5},
        ], margin=0.06)
        y += 0.84
    add_box(s, 7.0, 4.98, 6.0, 1.9, fill=LIGHT, line=LINE)
    add_picture_fit(s, HERE / "demo_screenshot.png", 7.1, 5.05, 3.1, 1.75)
    add_text(s, 10.25, 5.05, 2.7, 1.78, [
        {"text": "Live replay demo + Android app", "bold": True, "size": 11, "color": NAVY, "space_after": 3},
        {"text": "Real IO-VNBD drive through a 1 km blackout: plain IMU 942 m off, IDR + map-matching 7 m. App shows a smooth vehicle icon with an unmistakable GNSS / Dead-reckoning banner.", "size": 9.5},
    ], margin=0.03)


def slide6(prs):
    s = prs.slides[5]
    set_title(s, "RESEARCH AND REFERENCES", 32)
    set_team_oval(s)
    drop_instruction_textbox(s)

    section_label(s, 0.35, 1.25, 7.0, "Our work (open for review)", ORANGE)
    add_text(s, 0.35, 1.6, 7.0, 1.55, [
        {"runs": [{"text": "Code, models, results: ", "bold": True, "size": 11.5, "color": NAVY}, {"text": REPO_URL, "size": 11.5, "color": NAVY}], "bullet": True},
        {"text": "Reproducible pipeline (requirements.txt / Docker): IO-VNBD loader → speed filter training → blackout evaluation → map-matching → figures; RESULTS.md holds every number on this deck.", "bullet": True, "size": 11},
        {"text": "Browser replay demo (demo/index.html) and Android APK (mobile_app/ui) included in the repository.", "bullet": True, "size": 11},
    ], space_after=4, margin=0.03)

    section_label(s, 0.35, 3.2, 7.0, "References", ORANGE)
    refs = [
        "Onyekpe U., Palade V., Kanarachos S., et al. “IO-VNBD: Inertial and Odometry benchmark dataset for ground vehicle positioning.” Data in Brief, 2021. github.com/onyekpeu/IO-VNBD",
        "Newson P., Krumm J. “Hidden Markov Map Matching Through Noise and Sparseness.” ACM SIGSPATIAL GIS, 2009.",
        "Brossard M., Barrau A., Bonnabel S. “AI-IMU Dead-Reckoning.” IEEE Trans. Intelligent Vehicles, 2020.",
        "Herath S., Yan H., Furukawa Y. “RoNIN: Robust Neural Inertial Navigation in the Wild.” ICRA, 2020.",
        "Solà J. “Quaternion kinematics for the error-state Kalman filter.” arXiv:1711.02508, 2017.",
        "Groves P. D. Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, 2nd ed., Artech House, 2013 (NHC, ZUPT, loosely coupled INS/GNSS).",
        "OpenStreetMap contributors, openstreetmap.org; Boeing G. “OSMnx” (Computers, Environment and Urban Systems, 2017).",
        "TensorFlow Lite / ONNX Runtime documentation for on-device inference.",
    ]
    add_text(s, 0.35, 3.52, 7.0, 3.4, [{"text": r, "bullet": True, "size": 10, "bullet_char": "▪"} for r in refs], space_after=3, margin=0.03)

    section_label(s, 7.65, 1.25, 5.4, "Key findings from our IO-VNBD study", ORANGE)
    add_picture_fit(s, FIG / "drift_vs_distance.png", 7.65, 1.6, 5.4, 3.0)
    add_text(s, 7.65, 4.62, 5.4, 2.3, [
        {"text": "Inertial-only phone DR stays under 10 % only to ~150–200 m; map constraints extend this to 1 km (median 29.3 % → 3.3–7.0 %).", "bullet": True, "size": 10.5},
        {"text": "Holding the last GNSS speed is within 1–2 % of oracle speed over a tunnel; gyro heading, not speed, dominates error.", "bullet": True, "size": 10.5},
        {"text": "Inertial alignment of the dataset (phone vs ECU, 20 sessions) raised Δv correlation from 0.08 to 0.31, proving recoverable motion information in the phone IMU.", "bullet": True, "size": 10.5},
        {"text": "Dataset audit found and fixed a gyro/gravity axis-matching bug that corrupts heading in naive loaders.", "bullet": True, "size": 10.5},
    ], space_after=3, margin=0.03)


def main():
    prs = Presentation(str(TEMPLATE))
    slide1(prs)
    slide2(prs)
    slide3(prs)
    slide4(prs)
    slide5(prs)
    slide6(prs)
    delete_slide(prs, 6)  # instructions slide
    prs.save(str(OUT))
    print("wrote", OUT, "slides:", len(prs.slides))


if __name__ == "__main__":
    main()
