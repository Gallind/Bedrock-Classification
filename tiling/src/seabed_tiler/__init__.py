"""Overlapping-tile splitter for georeferenced seabed survey data.

Cuts co-located raster (.jpg+.jgw) and point-grid (.xyz) feature layers plus a
shapefile class map into equally sized, overlapping, fully georeferenced GeoTIFF
tiles for ML bedrock classification. See config/ for the tunable knobs.
"""

__version__ = "0.1.0"
