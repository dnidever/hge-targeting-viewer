# HGE Targeting Photometry Viewer

Interactive Streamlit viewer for an HGE photometric catalog. It displays a binned density map in Galactic longitude and latitude; hovering or clicking the map moves a circular selection and updates the adjacent `J-Ks` versus `H` CMD.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload a FITS, CSV, Parquet, or whitespace-delimited table containing at least:

- `l`, `b` in degrees
- `jmag`, `hmag`, `kmag` in magnitudes

The sidebar controls the circular selection radius and density-map resolution. A manual center entry is available as a fallback. Extra catalog columns, including `ak`, `jk0`, and `h0`, are retained but are not required.

## Notes

- The density map is binned, so the hover center is the center of the hovered bin.
- Circular selections use a 3D unit-vector KD-tree and exact spherical chord distances, including across the `l=0/360` boundary.
- For very large files, Parquet generally loads faster and uses less memory than text or CSV.
