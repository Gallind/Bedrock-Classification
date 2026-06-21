"""Renders typed slide specs onto a python-pptx Presentation.

Design language — a clean, technical, deep-ocean look:
  * subtle diagonal gradient background + a faint sonar-arc motif
  * monospace eyebrows / labels / footer, bold sans headings, gold *highlights*
  * floated maps (background keyed out), light cards for charts, framed media
  * an accent header marker + hairline, a footer with page numbers and brand

Each spec type from content.py maps to one `_add_<kind>` method; `add()`
dispatches and attaches the scene narration as speaker notes.
"""
from PIL import Image as PILImage
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from . import content as C
from . import narration
from .assets import resolve
from .imageprep import keyed_transparent
from .palette import CLASSES, COLORS, rgb

SLIDE_W = 13.333
SLIDE_H = 7.5
MARGIN_L = 0.92
CONTENT_W = SLIDE_W - 2 * MARGIN_L
BODY_TOP = 2.32
FOOT_Y = 7.0

# Embedded via deck/fonts.py (TTFs in pptx/fonts/). Fall back gracefully to
# system fonts if a viewer somehow lacks them.
SANS = "Manrope"
SANS_HEAVY = "Epilogue"
MONO = "JetBrains Mono"

ORG_LOGOS = ["org_iolr.png", "org_reichman.png", "org_code4good.png"]
_A = "main"
_NS = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'


# ---- low-level XML helpers ----------------------------------------------
def _hex(c: str) -> str:
    return COLORS.get(c, c).lstrip("#").upper()


def _no_shadow(shape):
    shape.shadow.inherit = False


def _gradient(shape, stops, angle_deg):
    spPr = shape._element.spPr
    for tag in ("a:noFill", "a:solidFill", "a:gradFill", "a:blipFill", "a:pattFill"):
        e = spPr.find(qn(tag))
        if e is not None:
            spPr.remove(e)
    gs = "".join(
        f'<a:gs pos="{int(p * 100000)}"><a:srgbClr val="{_hex(c)}"/></a:gs>'
        for p, c in stops
    )
    el = parse_xml(
        f'<a:gradFill {_NS}><a:gsLst>{gs}</a:gsLst>'
        f'<a:lin ang="{int(angle_deg * 60000)}" scaled="1"/></a:gradFill>'
    )
    ln = spPr.find(qn("a:ln"))
    (ln.addprevious if ln is not None else spPr.append)(el)


def _fill_alpha(shape, pct):
    srgb = shape.fill.fore_color._xFill.find(qn("a:srgbClr"))
    srgb.append(parse_xml(f'<a:alpha {_NS} val="{int(pct * 1000)}"/>'))


def _line(shape, color, width_pt, alpha=None):
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(width_pt)
    if alpha is not None:
        srgb = shape.line._get_or_add_ln().find(qn("a:solidFill")).find(qn("a:srgbClr"))
        srgb.append(parse_xml(f'<a:alpha {_NS} val="{int(alpha * 1000)}"/>'))


def _soft_shadow(shape, blur=14, dist=7, alpha=45):
    spPr = shape._element.spPr
    e = spPr.find(qn("a:effectLst"))
    if e is not None:
        spPr.remove(e)
    spPr.append(parse_xml(
        f'<a:effectLst {_NS}><a:outerShdw blurRad="{int(Pt(blur))}" dist="{int(Pt(dist))}" '
        f'dir="5400000" rotWithShape="0"><a:srgbClr val="000000">'
        f'<a:alpha val="{int(alpha * 1000)}"/></a:srgbClr></a:outerShdw></a:effectLst>'
    ))


_P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def _timing_xml(steps, *, dur=380, float_frac=0.05):
    """Build a <p:timing> tree that auto-plays on slide entry and is recognised by
    PowerPoint's Animation pane.

    Animations must live in the mainSeq or PowerPoint ignores them (they won't show
    in the Animation tab). The mainSeq's click group is normally gated on a click
    (<p:cond delay="indefinite"/>); setting that group's delay to 0 makes the whole
    build start automatically when the slide appears — i.e. "Start With Previous" on
    the first effect. Subsequent effects use cumulative `begin_delay_ms` (measured
    from slide load) so band-groups cascade.

    Each shape fades in while a *relative* motion path lifts it from `float_frac` of
    slide-height below to its authored spot (an additive ppt_y offset is mis-read by
    PowerPoint as an absolute coordinate and pins everything to the top).
    """
    cid = [5]  # 1=tmRoot 2=mainSeq 3=clickGroup 4=withGroup

    def nid():
        cid[0] += 1
        return cid[0] - 1

    # start offset (relative to authored position) the shape drifts in from
    offsets = {"up": (0, float_frac), "down": (0, -float_frac),
               "left": (-float_frac, 0), "right": (float_frac, 0)}

    pars = []
    for sid, delay, direction in steps:
        par_id, set_id, mot_id, fade_id = nid(), nid(), nid(), nid()
        dx, dy = offsets[direction]
        pars.append(
            f'<p:par><p:cTn id="{par_id}" presetID="42" presetClass="entr" '
            f'presetSubtype="0" fill="hold" grpId="0" nodeType="withEffect">'
            f'<p:stCondLst><p:cond delay="{delay}"/></p:stCondLst><p:childTnLst>'
            f'<p:set><p:cBhvr><p:cTn id="{set_id}" dur="1" fill="hold">'
            f'<p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>'
            f'<p:tgtEl><p:spTgt spid="{sid}"/></p:tgtEl>'
            f'<p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst>'
            f'</p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set>'
            f'<p:animMotion origin="layout" path="M {dx} {dy} L 0 0 E" '
            f'pathEditMode="relative"><p:cBhvr><p:cTn id="{mot_id}" dur="{dur}" fill="hold"/>'
            f'<p:tgtEl><p:spTgt spid="{sid}"/></p:tgtEl>'
            f'<p:attrNameLst><p:attrName>ppt_x</p:attrName><p:attrName>ppt_y</p:attrName>'
            f'</p:attrNameLst></p:cBhvr></p:animMotion>'
            f'<p:animEffect transition="in" filter="fade"><p:cBhvr>'
            f'<p:cTn id="{fade_id}" dur="{dur}"/>'
            f'<p:tgtEl><p:spTgt spid="{sid}"/></p:tgtEl></p:cBhvr></p:animEffect>'
            f'</p:childTnLst></p:cTn></p:par>'
        )
    nav = ('<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl>'
           '</p:cond></p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0">'
           '<p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst>')
    return parse_xml(
        f'<p:timing xmlns:p="{_P_NS}"><p:tnLst><p:par>'
        f'<p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
        f'<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq">'
        f'<p:childTnLst>'
        f'<p:par><p:cTn id="3" fill="hold" nodeType="withEffect">'
        f'<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
        f'<p:par><p:cTn id="4" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
        f'<p:childTnLst>{"".join(pars)}</p:childTnLst></p:cTn></p:par>'
        f'</p:childTnLst></p:cTn></p:par>'
        f'</p:childTnLst>{nav}</p:cTn>{nav}</p:seq>'
        f'</p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>'
    )


# ---- builder -------------------------------------------------------------
class DeckBuilder:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width = Inches(SLIDE_W)
        self.prs.slide_height = Inches(SLIDE_H)
        self._blank = self.prs.slide_layouts[6]
        self._n = 0
        self._total = 0
        self._bg_count = 0
        self._static_ids = set()  # shapes that must NOT animate (footer)

    # ---- primitives ------------------------------------------------------
    def _slide(self, motif=True):
        slide = self.prs.slides.add_slide(self._blank)
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(SLIDE_W), Inches(SLIDE_H))
        _no_shadow(bg)
        bg.line.fill.background()
        _gradient(bg, [(0.0, "#07151d"), (0.55, "#0a212c"), (1.0, "#0e2b3a")], 110)
        if motif:
            for rad in (2.3, 3.6, 4.9, 6.2):
                cx, cy = SLIDE_W + 0.35, SLIDE_H + 0.35
                o = slide.shapes.add_shape(
                    MSO_SHAPE.OVAL, Inches(cx - rad), Inches(cy - rad),
                    Inches(2 * rad), Inches(2 * rad),
                )
                _no_shadow(o)
                o.fill.background()
                _line(o, "accent", 1.1, alpha=7)
        # everything added from here on is content — remember where it starts so
        # _animate() can cascade just the content, leaving the gradient/motif still.
        self._bg_count = len(slide.shapes)
        self._static_ids = set()
        return slide

    def _box(self, slide, l, t, w, h, *, fill=None, line=None, line_w=1.0,
             radius=0.05, alpha_fill=None, shadow=False):
        shp = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h)
        )
        shp.adjustments[0] = radius
        _no_shadow(shp)
        if fill is None:
            shp.fill.background()
        else:
            shp.fill.solid()
            shp.fill.fore_color.rgb = rgb(fill)
            if alpha_fill is not None:
                _fill_alpha(shp, alpha_fill)
        if line is None:
            shp.line.fill.background()
        else:
            _line(shp, line, line_w)
        if shadow:
            _soft_shadow(shp)
        return shp

    def _text(self, slide, l, t, w, h, text, *, size, color="text", bold=False,
              align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, caps=False, spacing=None,
              line_spacing=None, font=SANS, rich=False):
        box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = align
        if line_spacing:
            p.line_spacing = line_spacing
        body = text.upper() if caps else text
        segments = body.split("*") if rich else [body]
        for i, seg in enumerate(segments):
            if seg == "":
                continue
            run = p.add_run()
            run.text = seg
            f = run.font
            f.size = Pt(size)
            f.name = font
            if rich and i % 2 == 1:
                f.bold = True
                f.color.rgb = rgb("accent2")
            else:
                f.bold = bold
                f.color.rgb = rgb(color)
            if spacing:
                run._r.get_or_add_rPr().set("spc", str(int(spacing)))
        return box

    def _hairline(self, slide, l, t, w):
        base = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(w), Pt(1.0))
        _no_shadow(base)
        base.line.fill.background()
        base.fill.solid()
        base.fill.fore_color.rgb = rgb("panelLine")
        _fill_alpha(base, 60)
        seg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(1.15), Pt(2.4))
        _no_shadow(seg)
        seg.line.fill.background()
        seg.fill.solid()
        seg.fill.fore_color.rgb = rgb("accent")

    def _header(self, slide, eyebrow, heading):
        marker = self._box(slide, MARGIN_L, 0.66, 0.13, 0.46, fill="accent", radius=0.35)
        if eyebrow:
            self._text(slide, MARGIN_L + 0.34, 0.66, CONTENT_W - 0.34, 0.4, eyebrow,
                       size=12.5, color="accent", font=MONO, caps=True, spacing=260,
                       anchor=MSO_ANCHOR.MIDDLE)
        self._text(slide, MARGIN_L, 1.12, CONTENT_W, 0.95, heading,
                   size=33, color="text", bold=True, font=SANS_HEAVY, line_spacing=1.0, rich=True)
        self._hairline(slide, MARGIN_L, 2.02, CONTENT_W)

    def _footer(self, slide):
        b1 = self._text(slide, MARGIN_L, FOOT_Y, 7.0, 0.3, "Seabed Classification",
                        size=9, color="textDim", font=MONO, caps=True, spacing=180,
                        anchor=MSO_ANCHOR.MIDDLE)
        b2 = self._text(slide, SLIDE_W - MARGIN_L - 3.0, FOOT_Y, 3.0, 0.3,
                        f"{self._n:02d} / {self._total:02d}", size=9, color="textDim", font=MONO,
                        align=PP_ALIGN.RIGHT, spacing=180, anchor=MSO_ANCHOR.MIDDLE)
        # the footer brand + page number stay put — exclude them from the cascade
        self._static_ids.update({b1.shape_id, b2.shape_id})

    # ---- images ----------------------------------------------------------
    def _place_image(self, slide, path, l, t, w, h, pad=0.0):
        with PILImage.open(path) as im:
            iw, ih = im.size
        ar = iw / ih if ih else 1.0
        bw, bh = w - 2 * pad, h - 2 * pad
        if ar > bw / bh:
            dw, dh = bw, bw / ar
        else:
            dh, dw = bh, bh * ar
        dl = l + (w - dw) / 2
        dt = t + (h - dh) / 2
        return slide.shapes.add_picture(str(path), Inches(dl), Inches(dt), Inches(dw), Inches(dh))

    def _image(self, slide, name, l, t, w, h, *, style="frame"):
        path = resolve(name)
        if path is None:
            self._box(slide, l, t, w, h, fill="panel", line="panelLine", radius=0.04)
            self._text(slide, l, t, w, h, f"asset missing\n{name}", size=13, color="textDim",
                       font=MONO, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
            return
        if style == "card":
            card = self._box(slide, l, t, w, h, fill="#f4f7f9", radius=0.05, shadow=True)
            _line(card, "panelLine", 1.0)
            self._place_image(slide, path, l, t, w, h, pad=0.28)
        elif style == "float":
            keyed = keyed_transparent(path)
            self._place_image(slide, path if keyed is None else keyed, l, t, w, h, pad=0.12)
        else:  # frame
            self._box(slide, l, t, w, h, fill="bgAlt", line="panelLine", radius=0.04, shadow=True)
            self._place_image(slide, path, l, t, w, h, pad=0.14)

    def _org_strip(self, slide, y, height=0.62):
        present = [(n, resolve(n)) for n in ORG_LOGOS]
        present = [(n, p) for n, p in present if p is not None]
        if not present:
            return
        gap = 0.7
        sizes = []
        for _, p in present:
            with PILImage.open(p) as im:
                iw, ih = im.size
            sizes.append(height * iw / ih)
        total = sum(sizes) + gap * (len(sizes) - 1)
        x = (SLIDE_W - total) / 2
        for (_, p), w in zip(present, sizes):
            slide.shapes.add_picture(str(p), Inches(x), Inches(y), height=Inches(height))
            x += w + gap

    # ---- credit block (names w/ affiliation superscripts + legend) -------
    def _names_line(self, slide, people, l, t, w, *, size, color, sup_color, sep="   ·   "):
        """One centred line of bold names, each followed by a raised superscript."""
        box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(0.45))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        for i, (name, sup) in enumerate(people):
            if i:
                g = p.add_run(); g.text = sep
                g.font.size = Pt(size); g.font.name = SANS; g.font.bold = True
                g.font.color.rgb = rgb(color)
            r = p.add_run(); r.text = name
            r.font.size = Pt(size); r.font.name = SANS; r.font.bold = True
            r.font.color.rgb = rgb(color)
            if sup:
                sp = p.add_run(); sp.text = sup
                sp.font.size = Pt(size * 0.58); sp.font.name = SANS; sp.font.bold = True
                sp.font.color.rgb = rgb(sup_color)
                sp._r.get_or_add_rPr().set("baseline", "30000")  # raise to superscript
        return box

    def _affil_line(self, slide, items, l, t, w, *, size=11):
        box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(0.32))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        for i, (num, text) in enumerate(items):
            if i:
                g = p.add_run(); g.text = "        "
                g.font.size = Pt(size); g.font.name = SANS
            rn = p.add_run(); rn.text = num + "  "
            rn.font.size = Pt(size); rn.font.name = SANS; rn.font.bold = True
            rn.font.color.rgb = rgb("text")
            rt = p.add_run(); rt.text = text
            rt.font.size = Pt(size); rt.font.name = SANS
            rt.font.color.rgb = rgb("textDim")
        return box

    # ---- slide kinds -----------------------------------------------------
    def _add_title(self, s: "C.Title"):
        slide = self._slide()
        # The opening slide carries the full credit block (names + affiliation
        # legend + source); a bare title centres just title + subtitle.
        has_credits = bool(s.credits)
        logo_top, logo_w = (0.5, 1.28) if has_credits else (0.82, 2.0)
        path = resolve(s.logo)
        if path is not None:
            with PILImage.open(path) as im:
                iw, ih = im.size
            self._place_image(slide, path, (SLIDE_W - logo_w) / 2, logo_top, logo_w, logo_w * ih / iw)
        eyebrow_y, title_y, bar_y, sub_y = (1.94, 2.32, 3.36, 3.56) if has_credits else (3.05, 3.5, 4.62, 4.85)
        self._text(slide, 1.0, eyebrow_y, SLIDE_W - 2.0, 0.4, "IOLR · Multibeam seabed mapping",
                   size=13, color="accent", font=MONO, caps=True, spacing=300, align=PP_ALIGN.CENTER)
        self._text(slide, 1.0, title_y, SLIDE_W - 2.0, 1.0, s.title,
                   size=48 if has_credits else 50, color="text", bold=True, font=SANS_HEAVY, align=PP_ALIGN.CENTER)
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches((SLIDE_W - 1.4) / 2), Inches(bar_y), Inches(1.4), Pt(2.6))
        _no_shadow(bar)
        bar.line.fill.background()
        bar.fill.solid()
        bar.fill.fore_color.rgb = rgb("accent")
        if s.subtitle:
            sub_h = 0.7 if has_credits else 1.1
            self._text(slide, 2.6, sub_y, SLIDE_W - 5.2, sub_h, s.subtitle, size=16 if has_credits else 17.5,
                       color="textDim", align=PP_ALIGN.CENTER, line_spacing=1.2, rich=True)
        if has_credits:
            # two name lines — white students, gold advisors — in the previous style
            students = [(n, sup) for n, sup, hi in s.credits if not hi]
            advisors = [(n, sup) for n, sup, hi in s.credits if hi]
            self._names_line(slide, students, 0.5, 4.16, SLIDE_W - 1.0,
                             size=15, color="text", sup_color="textDim")
            self._names_line(slide, advisors, 0.5, 4.62, SLIDE_W - 1.0,
                             size=15, color="accent2", sup_color="accent2")
            self._org_strip(slide, 5.18, height=0.52)
            # affiliation legend (short pair on one line, long one below) + source
            y = 5.86
            if s.affiliations:
                self._affil_line(slide, s.affiliations[:2], 1.0, y, SLIDE_W - 2.0, size=9.5)
                y += 0.25
                for item in s.affiliations[2:]:
                    self._affil_line(slide, [item], 1.0, y, SLIDE_W - 2.0, size=9.5)
                    y += 0.25
            if s.source:
                self._text(slide, 1.0, y + 0.12, SLIDE_W - 2.0, 0.3, s.source,
                           size=10, color="text", align=PP_ALIGN.CENTER)
        else:
            self._org_strip(slide, 5.95)
        if s.footer:
            self._text(slide, 1.0, FOOT_Y, SLIDE_W - 2.0, 0.4, s.footer, size=11,
                       color="textDim", font=MONO, caps=True, spacing=160, align=PP_ALIGN.CENTER)
        return slide

    def _add_bullets(self, s: "C.Bullets"):
        slide = self._slide()
        self._header(slide, s.eyebrow, s.heading)
        self._bullets(slide, s.bullets, MARGIN_L, BODY_TOP, CONTENT_W)
        self._footer(slide)
        return slide

    def _bullets(self, slide, items, l, top, w):
        n = max(len(items), 1)
        step = min(1.02, (FOOT_Y - 0.25 - top) / n)
        y = top
        for item in items:
            self._box(slide, l, y + 0.09, 0.17, 0.17, fill="accent", radius=0.3)
            self._text(slide, l + 0.46, y - 0.04, w - 0.46, step, item,
                       size=19, color="text", line_spacing=1.08, rich=True)
            y += step

    def _add_stats(self, s: "C.Stats"):
        slide = self._slide()
        self._header(slide, s.eyebrow, s.heading)
        n = len(s.stats)
        gap = 0.42
        cw = (CONTENT_W - (n - 1) * gap) / n
        ch, top = 2.65, 2.7
        for i, st in enumerate(s.stats):
            l = MARGIN_L + i * (cw + gap)
            self._box(slide, l, top, cw, ch, fill="panel", line="panelLine", radius=0.06, shadow=True)
            accent = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(top), Inches(cw), Pt(3.4))
            accent.adjustments[0] = 0.5
            _no_shadow(accent)
            accent.line.fill.background()
            accent.fill.solid()
            accent.fill.fore_color.rgb = rgb(st.color)
            self._text(slide, l + 0.34, top + 0.34, cw - 0.6, 0.4, st.caption,
                       size=12, color="textDim", font=MONO, caps=True, spacing=180)
            self._text(slide, l + 0.32, top + 0.8, cw - 0.6, 1.1, st.value,
                       size=52, color=st.color, bold=True, font=SANS_HEAVY)
            if st.note:
                self._text(slide, l + 0.34, top + 1.95, cw - 0.6, 0.5, st.note,
                           size=14, color="textDim", font=MONO)
        if s.footnote:
            self._text(slide, MARGIN_L, top + ch + 0.4, CONTENT_W, 0.6, s.footnote,
                       size=17, color="textDim", line_spacing=1.2, rich=True)
        self._footer(slide)
        return slide

    def _add_image(self, s: "C.Image"):
        slide = self._slide()
        self._header(slide, s.eyebrow, s.heading)
        ih = 3.95 if s.caption else 4.45
        self._image(slide, s.image, MARGIN_L, BODY_TOP, CONTENT_W, ih, style=s.style)
        if s.caption:
            self._text(slide, MARGIN_L, BODY_TOP + ih + 0.16, CONTENT_W, 0.5, s.caption,
                       size=14, color="textDim", font=MONO, align=PP_ALIGN.CENTER)
        self._footer(slide)
        return slide

    def _add_compare(self, s: "C.Compare"):
        slide = self._slide()
        self._header(slide, s.eyebrow, s.heading)
        gap = 0.6
        bw = (CONTENT_W - gap) / 2
        bh = 3.7
        for l, image, label in (
            (MARGIN_L, s.left_image, s.left_label),
            (MARGIN_L + bw + gap, s.right_image, s.right_label),
        ):
            self._image(slide, image, l, BODY_TOP, bw, bh, style=s.style)
            self._text(slide, l, BODY_TOP + bh + 0.14, bw, 0.4, label,
                       size=14, color="accent", font=MONO, caps=True, spacing=150,
                       align=PP_ALIGN.CENTER)
        self._footer(slide)
        return slide

    def _add_imagebullets(self, s: "C.ImageBullets"):
        slide = self._slide()
        self._header(slide, s.eyebrow, s.heading)
        iw, gap, ih = 5.5, 0.7, 4.15
        if s.image_side == "left":
            img_l, text_l = MARGIN_L, MARGIN_L + iw + gap
        else:
            img_l, text_l = MARGIN_L + CONTENT_W - iw, MARGIN_L
        self._image(slide, s.image, img_l, BODY_TOP, iw, ih, style=s.style)
        self._bullets(slide, s.bullets, text_l, BODY_TOP + 0.12, CONTENT_W - iw - gap)
        self._footer(slide)
        return slide

    def _add_legend(self, s: "C.Legend"):
        slide = self._slide()
        self._header(slide, s.eyebrow, s.heading)
        y = BODY_TOP + 0.15
        for c in CLASSES:
            self._box(slide, MARGIN_L, y, 0.36, 0.36, fill=c["color"], radius=0.22)
            self._text(slide, MARGIN_L + 0.62, y - 0.02, 7.0, 0.42, c["label"],
                       size=23, color="text", anchor=MSO_ANCHOR.MIDDLE)
            y += 0.8
        if s.note:
            self._text(slide, MARGIN_L, y + 0.3, CONTENT_W, 0.9, s.note,
                       size=17, color="textDim", line_spacing=1.25, rich=True)
        self._footer(slide)
        return slide

    # ---- dispatch --------------------------------------------------------
    _KIND = {
        C.Title: "_add_title",
        C.Bullets: "_add_bullets",
        C.Stats: "_add_stats",
        C.Image: "_add_image",
        C.Compare: "_add_compare",
        C.ImageBullets: "_add_imagebullets",
        C.Legend: "_add_legend",
    }

    def _transition(self, slide, dur_ms: int = 600) -> None:
        """Inject a subtle smooth fade between slides (python-pptx has no API for it).

        Uses mc:AlternateContent so modern PowerPoint honours the precise duration
        (p14:dur) while older versions fall back to a plain medium-speed fade. The
        <p:transition> belongs right after <p:clrMapOvr> in the CT_Slide sequence.
        """
        sld = slide._element
        xml = (
            '<mc:AlternateContent '
            'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">'
            '<mc:Choice xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            'xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" Requires="p14">'
            f'<p:transition spd="slow" p14:dur="{dur_ms}"><p:fade/></p:transition>'
            '</mc:Choice>'
            '<mc:Fallback xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            '<p:transition spd="med"><p:fade/></p:transition>'
            '</mc:Fallback>'
            '</mc:AlternateContent>'
        )
        clr = sld.find(qn("p:clrMapOvr"))
        (clr.addnext if clr is not None else sld.append)(parse_xml(xml))

    def _animate(self, slide, *, stagger=110, directional=True) -> None:
        """Cascade content in with a Float-In entrance, grouped by vertical band.

        Shapes are ordered top→bottom and bucketed into ~half-inch bands; everything
        in a band floats in together (same begin), and successive bands cascade
        `stagger` ms apart. The footer stays put. When `directional`, the header
        block near the top (title + separator) slides in from the left while the
        body rises from below; otherwise everything rises from below.
        """
        content = [sh for sh in list(slide.shapes)[self._bg_count:]
                   if sh.shape_id not in self._static_ids]
        if not content:
            return
        band = Inches(0.5)
        ordered = sorted(content, key=lambda sh: (sh.top or 0, sh.left or 0))
        groups, prev_key = [], None
        for sh in ordered:
            key = int((sh.top or 0) / band)
            if key != prev_key:
                groups.append([])
                prev_key = key
            groups[-1].append(sh)
        steps = []
        for gi, grp in enumerate(groups):
            top0 = min((s.top or 0) for s in grp)
            direction = "left" if directional and top0 < Inches(2.2) else "up"
            for s in grp:
                steps.append((s.shape_id, gi * stagger, direction))
        slide._element.append(_timing_xml(steps))

    def add(self, spec) -> None:
        method = self._KIND.get(type(spec))
        if method is None:
            raise TypeError(f"no renderer for slide spec {type(spec).__name__}")
        self._n += 1
        slide = getattr(self, method)(spec)
        self._transition(slide)
        self._animate(slide, directional=not isinstance(spec, C.Title))
        notes = spec.notes or narration.note(getattr(spec, "scene", None))
        if notes:
            slide.notes_slide.notes_text_frame.text = notes
