"""Renders typed slide specs onto a python-pptx Presentation.

Mirrors the video's deep-ocean look: dark slides, cyan eyebrows, bold white
headings, panel cards, and the four class colours. Each spec type from
content.py maps to one `_add_<kind>` method; `add()` dispatches and attaches the
scene narration as speaker notes.
"""
from PIL import Image as PILImage
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from . import content as C
from . import narration
from .assets import resolve
from .palette import CLASSES, rgb

SLIDE_W = 13.333
SLIDE_H = 7.5
MARGIN_L = 0.9
CONTENT_W = SLIDE_W - 2 * MARGIN_L
BODY_TOP = 2.3


def _spacing(run, centipoints: int) -> None:
    """Letter-spacing via the rPr@spc attribute (units of 1/100 pt)."""
    run._r.get_or_add_rPr().set("spc", str(int(centipoints)))


class DeckBuilder:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width = Inches(SLIDE_W)
        self.prs.slide_height = Inches(SLIDE_H)
        self._blank = self.prs.slide_layouts[6]

    # ---- primitives -------------------------------------------------------
    def _slide(self):
        slide = self.prs.slides.add_slide(self._blank)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = rgb("bg")
        return slide

    def _text(self, slide, l, t, w, h, text, *, size, color="text", bold=False,
              align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, caps=False,
              spacing=None, line_spacing=None, font="Inter"):
        box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = align
        if line_spacing:
            p.line_spacing = line_spacing
        run = p.add_run()
        run.text = text.upper() if caps else text
        f = run.font
        f.size = Pt(size)
        f.bold = bold
        f.name = font
        f.color.rgb = rgb(color)
        if spacing:
            _spacing(run, spacing)
        return box

    def _round_rect(self, slide, l, t, w, h, *, fill="panel", line="panelLine",
                    line_w=1.0, radius=0.06):
        shp = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h)
        )
        shp.adjustments[0] = radius
        shp.shadow.inherit = False
        if fill is None:
            shp.fill.background()
        else:
            shp.fill.solid()
            shp.fill.fore_color.rgb = rgb(fill)
        if line is None:
            shp.line.fill.background()
        else:
            shp.line.color.rgb = rgb(line)
            shp.line.width = Pt(line_w)
        return shp

    def _chip(self, slide, l, t, size, color):
        shp = self._round_rect(slide, l, t, size, size, fill=color, line=None, radius=0.25)
        return shp

    def _rule(self, slide, l, t, w):
        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(0.045)
        )
        bar.shadow.inherit = False
        bar.fill.solid()
        bar.fill.fore_color.rgb = rgb("accent")
        bar.line.fill.background()
        return bar

    def _header(self, slide, eyebrow, heading):
        if eyebrow:
            self._text(slide, MARGIN_L, 0.55, CONTENT_W, 0.4, eyebrow,
                       size=14, color="accent", bold=True, caps=True, spacing=300)
        self._text(slide, MARGIN_L, 0.95, CONTENT_W, 1.0, heading,
                   size=30, color="text", bold=True, line_spacing=1.0)
        self._rule(slide, MARGIN_L, 1.95, 0.9)

    def _image(self, slide, name, l, t, w, h, *, frame=True):
        path = resolve(name)
        if path is None:
            self._placeholder(slide, name, l, t, w, h)
            return
        if frame:
            self._round_rect(slide, l, t, w, h, fill="bgAlt", line="panelLine", radius=0.04)
        with PILImage.open(path) as im:
            iw, ih = im.size
        ar = iw / ih if ih else 1.0
        pad = 0.12
        bw, bh = w - 2 * pad, h - 2 * pad
        if ar > bw / bh:
            dw, dh = bw, bw / ar
        else:
            dh, dw = bh, bh * ar
        dl = l + (w - dw) / 2
        dt = t + (h - dh) / 2
        slide.shapes.add_picture(str(path), Inches(dl), Inches(dt), Inches(dw), Inches(dh))

    def _placeholder(self, slide, name, l, t, w, h):
        shp = self._round_rect(slide, l, t, w, h, fill="panel", line="panelLine", radius=0.04)
        shp.line.dash_style = None
        self._text(slide, l, t, w, h, f"asset missing:\n{name}",
                   size=14, color="textDim", align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # ---- slide kinds ------------------------------------------------------
    def _add_title(self, s: "C.Title"):
        slide = self._slide()
        path = resolve(s.logo)
        if path is not None:
            with PILImage.open(path) as im:
                iw, ih = im.size
            lw = 2.6
            lh = lw * ih / iw
            self._image(slide, s.logo, (SLIDE_W - lw) / 2, 0.95, lw, lh, frame=False)
        self._text(slide, 1.0, 3.55, SLIDE_W - 2.0, 1.0, s.title,
                   size=46, color="text", bold=True, align=PP_ALIGN.CENTER)
        if s.subtitle:
            self._text(slide, 2.0, 4.75, SLIDE_W - 4.0, 1.2, s.subtitle,
                       size=18, color="textDim", align=PP_ALIGN.CENTER, line_spacing=1.2)
        if s.footer:
            self._text(slide, 1.0, 6.7, SLIDE_W - 2.0, 0.5, s.footer,
                       size=14, color="accent", bold=True, align=PP_ALIGN.CENTER,
                       caps=True, spacing=200)
        return slide

    def _add_section(self, s: "C.Section"):
        slide = self._slide()
        self._text(slide, MARGIN_L, 2.6, CONTENT_W, 0.5, s.eyebrow,
                   size=16, color="accent", bold=True, caps=True, spacing=350)
        self._text(slide, MARGIN_L, 3.1, CONTENT_W, 1.7, s.heading,
                   size=40, color="text", bold=True, line_spacing=1.0)
        self._rule(slide, MARGIN_L, 3.05, 0.9)
        return slide

    def _add_bullets(self, s: "C.Bullets"):
        slide = self._slide()
        self._header(slide, s.eyebrow, s.heading)
        self._bullet_column(slide, s.bullets, MARGIN_L, BODY_TOP, CONTENT_W)
        return slide

    def _bullet_column(self, slide, items, l, top, w):
        n = max(len(items), 1)
        step = min(1.0, (6.9 - top) / n)
        y = top
        for item in items:
            self._chip(slide, l, y + 0.07, 0.16, "accent")
            self._text(slide, l + 0.42, y - 0.06, w - 0.42, step, item,
                       size=20, color="text", line_spacing=1.05)
            y += step

    def _add_stats(self, s: "C.Stats"):
        slide = self._slide()
        self._header(slide, s.eyebrow, s.heading)
        n = len(s.stats)
        gap = 0.4
        card_w = (CONTENT_W - (n - 1) * gap) / n
        card_h = 2.7
        top = 2.7
        for i, st in enumerate(s.stats):
            l = MARGIN_L + i * (card_w + gap)
            self._round_rect(slide, l, top, card_w, card_h, fill="panel", line="panelLine")
            self._text(slide, l + 0.32, top + 0.32, card_w - 0.6, 0.4, st.caption,
                       size=13, color="textDim", bold=True, caps=True, spacing=200)
            self._text(slide, l + 0.3, top + 0.78, card_w - 0.6, 1.2, st.value,
                       size=46, color=st.color, bold=True)
            if st.note:
                self._text(slide, l + 0.32, top + 1.95, card_w - 0.6, 0.5, st.note,
                           size=15, color="textDim")
        if s.footnote:
            self._text(slide, MARGIN_L, top + card_h + 0.35, CONTENT_W, 0.6, s.footnote,
                       size=17, color="textDim")
        return slide

    def _add_image(self, s: "C.Image"):
        slide = self._slide()
        self._header(slide, s.eyebrow, s.heading)
        img_h = 4.0 if s.caption else 4.5
        self._image(slide, s.image, MARGIN_L, BODY_TOP, CONTENT_W, img_h)
        if s.caption:
            self._text(slide, MARGIN_L, BODY_TOP + img_h + 0.18, CONTENT_W, 0.5, s.caption,
                       size=15, color="textDim", align=PP_ALIGN.CENTER)
        return slide

    def _add_compare(self, s: "C.Compare"):
        slide = self._slide()
        self._header(slide, s.eyebrow, s.heading)
        gap = 0.6
        box_w = (CONTENT_W - gap) / 2
        box_h = 3.8
        for l, image, label in (
            (MARGIN_L, s.left_image, s.left_label),
            (MARGIN_L + box_w + gap, s.right_image, s.right_label),
        ):
            self._image(slide, image, l, BODY_TOP, box_w, box_h)
            self._text(slide, l, BODY_TOP + box_h + 0.15, box_w, 0.5, label,
                       size=16, color="accent", bold=True, align=PP_ALIGN.CENTER)
        return slide

    def _add_imagebullets(self, s: "C.ImageBullets"):
        slide = self._slide()
        self._header(slide, s.eyebrow, s.heading)
        img_w = 5.6
        gap = 0.6
        img_h = 4.2
        if s.image_side == "left":
            img_l = MARGIN_L
            text_l = MARGIN_L + img_w + gap
        else:
            img_l = MARGIN_L + CONTENT_W - img_w
            text_l = MARGIN_L
        text_w = CONTENT_W - img_w - gap
        self._image(slide, s.image, img_l, BODY_TOP, img_w, img_h)
        self._bullet_column(slide, s.bullets, text_l, BODY_TOP + 0.1, text_w)
        return slide

    def _add_legend(self, s: "C.Legend"):
        slide = self._slide()
        self._header(slide, s.eyebrow, s.heading)
        y = BODY_TOP + 0.1
        for c in CLASSES:
            self._chip(slide, MARGIN_L, y, 0.34, c["color"])
            self._text(slide, MARGIN_L + 0.6, y - 0.03, 7.0, 0.5, c["label"],
                       size=24, color="text", anchor=MSO_ANCHOR.MIDDLE)
            y += 0.78
        if s.note:
            self._text(slide, MARGIN_L, y + 0.25, CONTENT_W, 0.8, s.note,
                       size=17, color="textDim", line_spacing=1.2)
        return slide

    # ---- dispatch ---------------------------------------------------------
    _KIND = {
        C.Title: "_add_title",
        C.Section: "_add_section",
        C.Bullets: "_add_bullets",
        C.Stats: "_add_stats",
        C.Image: "_add_image",
        C.Compare: "_add_compare",
        C.ImageBullets: "_add_imagebullets",
        C.Legend: "_add_legend",
    }

    def add(self, spec) -> None:
        method = self._KIND.get(type(spec))
        if method is None:
            raise TypeError(f"no renderer for slide spec {type(spec).__name__}")
        slide = getattr(self, method)(spec)
        notes = spec.notes or narration.note(getattr(spec, "scene", None))
        if notes:
            slide.notes_slide.notes_text_frame.text = notes
