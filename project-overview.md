# Project: Israeli Seabed Classification Using Deep Learning (U-Net)

## Context and Background

I am a PM/researcher at IOLR (Israel Oceanographic & Limnological Research),
working on a seabed classification project aboard the R/V Bat-Galim. 
The vessel is equipped with two MBES systems:
- Kongsberg EM302 (30kHz) for deep water
- Kongsberg EM2040 (200–400kHz) for shallow water

The acoustic data processing pipeline is:
  Raw Kongsberg .all files → Qimera (bathymetry processing, exports .sd + txt/tiff XYZ)
                           → FMGT (backscatter processing, exports ascii X Y Z-Phi)
                           → ArcGIS Pro (spatial interpolation → classified seabed map)

The final data products from this pipeline are:
- Bathymetry: txt or tiff (X Y Z) at 1m x 1m resolution
- Backscatter: ascii (X Y Z-Phi) where Phi is the classified acoustic value
- Derived layers from ArcGIS Pro: slope, hillshade (computed from bathymetry)

## Reference Paper

The project is based on and aims to replicate the methodology from:
  "Seabed classification of multibeam echosounder data into bedrock/non-bedrock
   using deep learning" — Garone et al., Frontiers in Earth Science, 2023
   DOI: 10.3389/feart.2023.1285368 
   path: ./DL_article.pdf

### Key specs from that paper:
- Study area: 576 km² off Norway (Søre Sunnmøre region)
- MBES resolution: 1m x 1m (bathymetry and backscatter)
- Input features: bathymetry, backscatter, slope, hillshade — all normalized to [0, 1]
- Patch size: 256 x 256 pixels, with 50% overlap between consecutive patches
- Training patches: ~22,000 per input channel
- Ground truth: expert-delineated map simplified to binary: bedrock / non-bedrock
- Spatial split: 24 manually drawn rectangles → train/val/test, with a mandatory
  1,000m buffer between training and test regions (to prevent spatial autocorrelation)
- Architecture: lightweight U-Net (Leclerc et al., 2019)
- Models trained: 4 single-input + 6 multi-input combinations
- Metrics: Dice Score, Accuracy, Kappa, UAcc, PAcc
- Best performers: depth-only (MD) and slope-only (MS) models (DS: 0.69–0.80)
- Normalization: all input layers normalized to [0, 1] range

## My Data (Israeli Mediterranean Seabed)

At this folder we got 4 differents polygons Data - fill the details what they contain below

## What I Need You To Do

The Goal of this project is to let the researcher easy pipeline that they extract the data and provide those data into our system and we will provide a classification image as a result.
Our pipeline should ingest, edit the data before we run it on our system.


Please create a complete, production-ready research plan and codebase for this
project. Specifically:

### 1. Research Plan & Data Requirements
Answer the following, with justification based on the reference paper:
- How many labeled/annotated km² are needed minimum for training?
- How many 128*128 patches does that translate to? How can I make Agumentations without hurting the data?
- What is the minimum geographic coverage recommended before attempting evaluation
  on new unseen data?
- What percentage of the labeled area should be bedrock vs non-bedrock (sallow sand and sand) for
  a balanced training set?


### 2. Data Ingestion & Format Standardization
Write Python code to:
- Load bathymetry from txt (X Y Z) and GeoTIFF formats
- Load backscatter from ascii (X Y Z-Phi)
- Reproject and align all layers to the same CRS and pixel grid (1m x 1m)
- Compute slope and hillshade from the bathymetry grid (matching the ArcGIS Pro
  "Slope" and "Hillshade" tools: azimuth=315°, altitude=45°)
- Handle no-data values consistently across all layers

### 3. Normalization
Implement the exact normalization strategy from the paper:
- Normalize each input layer independently to [0, 1] range
- Explain in comments WHY we normalize to [0, 1] and not zero-mean/unit-variance
- Store normalization statistics (min/max per channel) for use at inference time
  (critical: normalization params must be computed ONLY on training data,
   then applied to val/test — never fit on the full dataset)

### 4. Patch Generation
- Extract 256x256 patches with 50% overlap (matching the paper exactly)
- Apply the 1,000m spatial buffer between train and test regions
- Handle boundary/edge patches with valid padding strategy
- Save patches as numpy .npy files or HDF5 — recommend the best format with reasoning
- Report how many patches the Israeli dataset will produce given its geographic extent

### 5. Annotation / Labeling Strategy
- Recommend a labeling tool for creating polygon annotations over the MBES rasters
  (e.g., QGIS, LabelMe, Roboflow) — with pros/cons for geospatial data
- Explain the minimum number of polygons / labeled regions to annotate before
  training is feasible
- Describe the annotation schema: what constitutes "hard substrate" vs "soft
  sediment" in the Israeli Mediterranean context

### 6. U-Net Implementation
- Implement the lightweight U-Net from Leclerc et al. (2019) in PyTorch
- Support variable number of input channels (1, 2, or 4 channels)
- Support 3-class output: [soft_sediment, hard_substrate, no_data]
- Use Dice Loss as the primary loss function
- Include class weighting to handle class imbalance

### 7. Training Pipeline
- PyTorch DataLoader with spatial-aware train/val/test split
- Data augmentation: horizontal flip, vertical flip, random rotation (90°, 180°, 270°) —
  ONLY spatial augmentations that are geophysically valid for MBES data
- Training loop with early stopping, LR scheduler
- Save best model checkpoint based on val Dice Score

### 8. Evaluation
Implement all metrics from the paper:
- Dice Score (per class + global)
- Overall Accuracy
- Cohen's Kappa
- Producer's Accuracy (PAcc) per class
- User's Accuracy (UAcc) per class
- Confusion matrix visualization
- Qualitative map reconstruction from overlapping patches
  (using the Vooban smoothly-blending algorithm for patch merging)

### 9. Decision Threshold Sweep
- Implement threshold sweep from 0.1 to 0.9 (as done in the paper)
- Plot threshold vs. Dice Score to find optimal operating point

### 10. Inference & Map Generation
- Run inference on new, unseen survey data (evaluation dataset)
- Reconstruct the full classified seabed map from patches
- Export result as GeoTIFF for import into ArcGIS Pro

## Technical Environment
- Python 3.10+
- PyTorch (CUDA if available, CPU fallback)
- Libraries: numpy, rasterio, geopandas, scikit-learn, matplotlib, tqdm
- No Tensorflow

## Output Structure Expected
Please organize the codebase as:
  israeli_seabed_dl/
  ├── data/
  │   ├── raw/          # bathymetry, backscatter ascii files from IOLR
  │   ├── processed/    # aligned GeoTIFFs, normalized layers
  │   └── patches/      # extracted 256x256 train/val/test patches
  ├── annotations/      # GeoJSON or shapefile with labeled polygons
  ├── models/           # saved .pt checkpoints
  ├── src/
  │   ├── ingest.py     # data loading and alignment
  │   ├── normalize.py  # normalization + stats saving
  │   ├── patches.py    # patch extraction with spatial buffer
  │   ├── dataset.py    # PyTorch Dataset class
  │   ├── unet.py       # U-Net architecture
  │   ├── train.py      # training loop
  │   ├── evaluate.py   # metrics + confusion matrix
  │   └── predict.py    # inference + map reconstruction
  ├── notebooks/
  │   └── exploration.ipynb  # data visualization and EDA
  └── README.md

## Deliverable Order
Please execute in this order:
1. Answer the research plan questions (section 1) before writing any code
2. Implement sections 2–4 (data pipeline) and test with dummy/synthetic data
3. Implement sections 6–8 (model + training + evaluation)
4. Implement sections 9–10 (threshold + inference)
5. Write README.md with full setup instructions and expected data format

Start with step 1 — the research plan questions — before writing a single line of code.