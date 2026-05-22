"""
Render .xyz seabed survey files to publication-quality images using PyGMT.
PyGMT is the official Python wrapper for Generic Mapping Tools (GMT 6).
"""

import os
import pygmt

DB = "DataBase"
DATASETS = [
    {
        "xyz":    f"{DB}/bathymetry Grid 0.5m.xyz",
        "output": "bathymetry_gmt.png",
        "title":  "Bathymetry – 0.5 m grid (UTM 36N)",
        "cmap":   "geo",          # GMT built-in: blue-deep ocean palette
        "label":  "Depth (m)",
        "reverse_cmap": True,     # deeper = darker blue
    },
    {
        "xyz":    f"{DB}/Slope  Grid 0.5m.xyz",
        "output": "slope_gmt.png",
        "title":  "Slope – 0.5 m grid (UTM 36N)",
        "cmap":   "hot",          # GMT built-in: yellow → red
        "label":  "Slope (°)",
        "reverse_cmap": False,
    },
]


def get_region(xyz_path):
    """Read the bounding box directly from GMT so we don't load the full file."""
    info = pygmt.grdinfo if False else None  # unused; use blockmean trick below
    # pygmt.info() reads XYZ and returns region string
    region = pygmt.info(data=xyz_path, spacing=0.5)
    return region   # e.g. [xmin, xmax, ymin, ymax]


def xyz_to_image(xyz_path, output_path, title, cmap, label, reverse_cmap, spacing=0.5):
    print(f"\n[+] Processing {os.path.basename(xyz_path)} …")

    # --- 1. Discover extent ------------------------------------------------
    region = pygmt.info(data=xyz_path, spacing=spacing)
    print(f"    Region: {region}")

    # --- 2. Load data into a pandas DataFrame (PyGMT accepts it directly) --
    import pandas as pd, numpy as np
    print("    Reading points …")
    df = pd.read_csv(xyz_path, header=None, names=["x", "y", "z"], dtype=np.float64)
    print(f"    {len(df):,} points  |  Z range: {df.z.min():.3f} → {df.z.max():.3f}")

    # --- 3. Build a regular grid with xyz2grd ------------------------------
    print("    Gridding …")
    grid = pygmt.xyz2grd(
        data=df,
        region=region,
        spacing=spacing,
        registration="gridline",
    )

    # --- 4. Colour palette -------------------------------------------------
    # Clip to 2–98 percentile so extreme outliers don't wash out the palette
    p2, p98 = float(df.z.quantile(0.02)), float(df.z.quantile(0.98))
    # Write to a named file so the CPT is explicitly tied to this figure
    # (avoids session-level CPT leaking between successive calls)
    cpt_file = output_path.replace(".png", ".cpt")
    pygmt.makecpt(cmap=cmap, series=[p2, p98], reverse=reverse_cmap, output=cpt_file)

    # --- 5. Figure ---------------------------------------------------------
    fig = pygmt.Figure()

    # Mercator projection scaled to 20 cm wide; /0 lets GMT choose height
    proj = "X20c/0"

    fig.grdimage(
        grid=grid,
        projection=proj,
        region=region,
        cmap=cpt_file,
        frame=["WSne", "xa", "ya"],
        interpolation="n",   # nearest-neighbour (data is already gridded)
    )

    fig.colorbar(
        cmap=cpt_file,
        frame=[f"x+l{label}"],
        position="JBC+w12c/0.4c+h+o0/1c",  # horizontal bar below map
    )

    fig.text(
        x=(region[0] + region[1]) / 2,
        y=region[3],
        text=title,
        font="14p,Helvetica-Bold,black",
        justify="BC",
        offset="0/0.5c",
        no_clip=True,
    )

    print(f"    Saving → {output_path}")
    fig.savefig(output_path, dpi=150, anti_alias=True)
    os.remove(cpt_file)
    print(f"    Saved  → {output_path}")


if __name__ == "__main__":
    for ds in DATASETS:
        xyz_to_image(
            ds["xyz"],
            ds["output"],
            ds["title"],
            ds["cmap"],
            ds["label"],
            ds["reverse_cmap"],
        )
    print("\nDone.")
