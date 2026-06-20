"""The deck: an ordered list of typed slide specs.

This is the deck's editable content layer — reorder, reword, or add slides here
without touching the rendering code.

Tuned for a ~3-minute exhibition talk (IOLR / Code4Good showcase), following the
organisers' flow: title + team → problem (+ audience) → solution figure →
the most impressive results → a short demo → conclusions. Earlier "system
design / alternatives" detail is intentionally dropped to fit the time.

Text may contain *asterisk markup* — wrapped spans render in the gold accent so
key terms and numbers stand out.

Image `style`:
  "float" — key the flat background out and float the subject (maps, renders)
  "card"  — sit on a clean light card (charts / matplotlib graphical content)
  "frame" — dark panel with a thin border (mixed / animated content)

Metrics here are the ORIGINAL published numbers (within-survey macro-Dice 0.784,
cross-survey LOPO 0.608) — matching the already-distributed video. They are not
auto-derived from training/runs; update them here when the deck is refreshed.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Title:
    title: str
    subtitle: str = ""
    footer: str = ""
    logo: str = "brand_mark.png"
    credits: list = field(default_factory=list)       # (name, superscript, highlight)
    affiliations: list = field(default_factory=list)  # (number, full text)
    source: str = ""
    scene: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Bullets:
    heading: str
    bullets: list[str]
    eyebrow: str = ""
    scene: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Stat:
    value: str
    caption: str
    note: str = ""
    color: str = "accent"


@dataclass
class Stats:
    heading: str
    stats: list[Stat]
    eyebrow: str = ""
    footnote: str = ""
    scene: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Image:
    heading: str
    image: str
    caption: str = ""
    eyebrow: str = ""
    style: str = "frame"
    scene: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Compare:
    heading: str
    left_image: str
    left_label: str
    right_image: str
    right_label: str
    eyebrow: str = ""
    style: str = "frame"
    scene: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ImageBullets:
    heading: str
    image: str
    bullets: list[str]
    eyebrow: str = ""
    image_side: str = "right"
    style: str = "frame"
    scene: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Legend:
    heading: str
    eyebrow: str = ""
    note: str = ""
    scene: Optional[str] = None
    notes: Optional[str] = None


DECK: list = [
    # 1 — Title: project, team, mentor, collaboration (~10s)
    Title(
        title="Seabed Classification",
        subtitle="Turning multibeam sonar into per-pixel maps of *rock, shallow buried rock, and sand*.",
        # highlight=False -> white "students" line; True -> gold "advisors" line
        credits=[
            ("Gal Lind", "1,2", False),
            ("Adi Lind", "1,2", False),
            ("Eden Tsarfaty", "1,2", False),
            ("Adi Gotlib", "1,2", False),
            ("Asaf Giladi", "3", True),
            ("Tomer Sheffer", "1", True),
            ("Sarel Cohen", "1", True),
        ],
        affiliations=[
            ("1", "Reichman University"),
            ("2", "Code4Good"),
            ("3", "Israel Oceanographic and Limnological Research (IOLR), National Institute of Oceanography, Haifa, Israel"),
        ],
        source="Method baseline: Garone et al., Frontiers in Earth Science, 2023",
        logo="brand_mark.png",
        scene="intro",
    ),

    # 2 — The problem (+ target audience) (~40s)
    Bullets(
        eyebrow="The problem",
        heading="Why automate *seabed mapping*?",
        bullets=[
            "Marine geologists map the seafloor *by hand* — slow, subjective, hard to reproduce.",
            "IOLR needs reproducible per-pixel maps of *rock, shallow buried rock, and sand*.",
            "Used downstream for *habitat and geohazard* mapping on Israel's Mediterranean shelf.",
            "Input: R/V Bat-Galim *multibeam echosounder* surveys — yet only ~1–2 km² of expert labels.",
        ],
        scene="intro",
    ),

    # 3 — The solution: one informative figure + the approach (~50s)
    ImageBullets(
        eyebrow="Our approach",
        heading="From sonar to a *per-pixel map*",
        image="band_bathymetry.jpg",
        image_side="right",
        style="float",
        bullets=[
            "Three physical layers — *bathymetry, backscatter, slope* — on one shared 1 m grid.",
            "Expert polygons rasterized into per-pixel *training labels*.",
            "A compact *U-Net* learns spatial context; RF / HGB trees as baselines.",
            "Scored *leave-one-survey-out* — tested on a survey never seen in training.",
        ],
        scene="pipeline",
    ),

    # 4 — Results: the headline numbers (~30s)
    Stats(
        eyebrow="Results",
        heading="The 3-band *U-Net* wins",
        stats=[
            Stat("0.784", "Within-survey", "macro-Dice", "accent"),
            Stat("0.608", "Cross-survey (LOPO)", "macro-Dice", "accent2"),
            Stat("0.976", "Rock, best survey", "cross-survey Dice", "rock"),
        ],
        footnote="Beats the 2-band U-Net and the tree baselines on *every* metric.",
        scene="results",
    ),

    # 5 — Results: the most impressive figure (~30s)
    Compare(
        eyebrow="Results",
        heading="*U-Net* vs expert ground truth",
        left_image="map_p1_ground_truth.png",
        left_label="Expert ground truth",
        right_image="map_p1_unet.png",
        right_label="U-Net prediction",
        style="float",
        scene="maps",
    ),

    # 6 — Short demo clip (~30s)
    Image(
        eyebrow="Live demo",
        heading="Watching the model *predict tile by tile*",
        image="watch_polygon3.gif",
        style="frame",
        caption="Our live QA viewer — each model fills in the survey tile by tile.",
        scene="watch",
    ),

    # 7 — Conclusions (~20s)
    Bullets(
        eyebrow="Conclusions",
        heading="*Takeaways*",
        bullets=[
            "Rock is mapped *reliably across surveys* — usable for hazard and habitat work.",
            "The ceiling is *data, not architecture*.",
            "Biggest next gain: more annotated *shallow buried rock* area — not a bigger model.",
        ],
        scene="conclusions",
    ),
]
