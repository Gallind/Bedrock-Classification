"""The deck: an ordered list of typed slide specs.

This is the deck's editable content layer — reorder, reword, or add slides here
without touching the rendering code. It deliberately diverges from the video's
12-scene flow (a bit more detail, a different order), while reusing the same
shared assets, palette, and narration (scene ids below pull the matching
narration in as speaker notes).

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
    scene: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Section:
    eyebrow: str
    heading: str
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
    scene: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ImageBullets:
    heading: str
    image: str
    bullets: list[str]
    eyebrow: str = ""
    image_side: str = "right"
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
    Title(
        title="Seabed Classification",
        subtitle="Turning multibeam sonar into per-pixel maps of rock, shallow rock, and sand",
        footer="IOLR  ·  Code4Good @ Reichman University",
        logo="brand_mark.png",
        scene="intro",
    ),

    Bullets(
        eyebrow="The problem",
        heading="Why automate seabed mapping?",
        bullets=[
            "Experts map the seafloor by hand — slow, subjective, and hard to reproduce.",
            "Goal: a reproducible per-pixel classifier for rock, shallow rock, and sand.",
            "Built with IOLR from R/V Bat-Galim multibeam echosounder surveys.",
        ],
        scene="intro",
    ),

    ImageBullets(
        eyebrow="Data",
        heading="Three physical input layers",
        image="band_bathymetry.jpg",
        image_side="right",
        bullets=[
            "Bathymetry — seabed depth from the multibeam echosounder.",
            "Backscatter — acoustic hardness proxy (scale varies per survey).",
            "Slope — derived from the bathymetry surface.",
            "Only ~1–2 km² of expert labels to learn from.",
        ],
        scene="data",
    ),

    Legend(
        eyebrow="Ground truth",
        heading="The three seabed classes",
        note="Experts hand-draw class polygons. Unlabeled pixels are ignored in both training and scoring.",
        scene="data",
    ),

    Bullets(
        eyebrow="Pipeline",
        heading="A deterministic tiling pipeline",
        bullets=[
            "Every layer reprojected onto one shared 1 m grid (UTM zone 36N).",
            "Expert polygons rasterized into per-pixel class labels.",
            "Each survey cut into overlapping 128 m feature + label tile pairs.",
        ],
        scene="pipeline",
    ),

    Bullets(
        eyebrow="Tiling",
        heading="Three tiling modes",
        bullets=[
            "Standard — axis-aligned tiles over the survey.",
            "Rotated — tiles aligned to the annotation orientation.",
            "Augmented — rigid re-extraction passes (~250 base tiles → 900+).",
        ],
        scene="modes",
    ),

    Bullets(
        eyebrow="Method",
        heading="The augmentation contract",
        bullets=[
            "Pixels are physical measurements → only rigid transforms (rotations, flips).",
            "No brightness, noise, or photometric edits — ever.",
            "Splits are spatial (whole regions), never random tiles.",
            "Backscatter is normalized per survey to bridge the cross-survey domain shift.",
        ],
        scene="augment",
    ),

    ImageBullets(
        eyebrow="Models",
        heading="Two model families, same three bands",
        image="feature_importance.png",
        image_side="right",
        bullets=[
            "U-Net — a compact CNN that learns spatial context.",
            "Random Forest & HistGradientBoosting — per-pixel tree baselines.",
            "Spatial variants smooth tree outputs with a depth-guided filter.",
            "Bathymetry is consistently the most informative band.",
        ],
        scene="models",
    ),

    Bullets(
        eyebrow="Evaluation",
        heading="How we measure generalisation",
        bullets=[
            "Leave-one-polygon-out: train on the other surveys, predict the held-out one.",
            "Repeat for every survey and report the mean — the honest number.",
            "It measures performance on a survey the model has never seen.",
            "Within-survey spatial-block splits are reported alongside for context.",
        ],
        scene="eval",
    ),

    Stats(
        eyebrow="Results",
        heading="The 3-band U-Net wins",
        stats=[
            Stat("0.784", "Within-survey", "macro-Dice", "accent"),
            Stat("0.608", "Cross-survey (LOPO)", "macro-Dice", "accent"),
            Stat("0.976", "Rock, best survey", "cross-survey Dice", "rock"),
        ],
        footnote="Beats the 2-band U-Net and the tree baselines on every metric.",
        scene="results",
    ),

    Image(
        eyebrow="Results",
        heading="Model comparison by tile type",
        image="metrics_by_type.png",
        caption="Within-survey test split across U-Net, Random Forest, and HGB (raw + spatial).",
        scene="results",
    ),

    Compare(
        eyebrow="Maps",
        heading="Classified maps — U-Net vs ground truth",
        left_image="map_p1_ground_truth.png",
        left_label="Expert ground truth",
        right_image="map_p1_unet.png",
        right_label="U-Net prediction",
        scene="maps",
    ),

    Compare(
        eyebrow="Maps",
        heading="Spatial filtering cleans up the trees",
        left_image="map_p1_rf_raw.png",
        left_label="Random Forest (raw)",
        right_image="map_p1_rf_spatial.png",
        right_label="Random Forest (spatial)",
        scene="maps",
    ),

    Image(
        eyebrow="Pipeline in motion",
        heading="Live watch viewer",
        image="watch_polygon3.gif",
        caption="Each model predicts tile by tile — this is how we caught a label-dropout bug.",
        scene="watch",
    ),

    Bullets(
        eyebrow="Takeaways",
        heading="Conclusions",
        bullets=[
            "Rock classification is reliable enough for hazard and habitat mapping.",
            "The ceiling is data, not architecture.",
            "Biggest next gain: more annotated shallow-rock area — not a bigger model.",
        ],
        scene="conclusions",
    ),

    Title(
        title="Thank you",
        subtitle="From raw multibeam survey to a trained, honestly evaluated seabed classifier — fully reproducible.",
        footer="IOLR  ·  Code4Good @ Reichman University",
        logo="brand_mark.png",
        scene="outro",
    ),
]
