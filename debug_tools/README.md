# Debug & Analysis Tools

Scripts for analyzing, visualizing, and debugging the glyph extraction and embedding pipeline.

## Scripts

### 1. `compare_thresholding.py`
Compares different image preprocessing/thresholding methods on a single text line.

**Purpose:** Understand how different thresholding approaches affect glyph extraction quality.

**Usage:**
```bash
cd examples/gospels  # or your example directory
python ../../debug_tools/compare_thresholding.py
```

**Outputs:**
- `compare_*.png` - Thresholded line images for each method
- `compare_*_components/` - Extracted components for each method

**Configuration:**
Edit the script to change which line to analyze:
```python
line_info = lines[1]  # Change to different line index
```

### 2. `debug_glyph.py`
Analyzes image preprocessing on the first few text lines with detailed statistics.

**Purpose:** Troubleshoot why glyphs aren't being extracted (or too few are found).

**Usage:**
```bash
cd examples/gospels
python ../../debug_tools/debug_glyph.py
```

**Outputs:**
- `debug_line_*_cropped.png` - Original cropped line
- `debug_line_*_binary.png` - Binary thresholded line
- Console output with pixel statistics

### 3. `test_embeddings.py`
Tests pre-computed embeddings and performs sample similarity searches.

**Purpose:** Validate that embeddings were generated correctly and find visually similar glyphs.

**Usage:**
```bash
cd examples/gospels
python ../../debug_tools/test_embeddings.py
```

**Outputs:**
- Console output showing similar glyphs and statistics

## Quick Troubleshooting

**Problem:** Very few glyphs extracted
- Use `compare_thresholding.py` to see which thresholding method works best
- Adjust `min_area` and `max_area` parameters in `glyph_extraction.py`

**Problem:** Too many false positives (noise extracted as glyphs)
- Check preprocessing with `debug_glyph.py` 
- Increase `min_area` to filter smaller noise
- Adjust threshold value (127 default) to make binary image cleaner

**Problem:** Embeddings don't seem to group similar glyphs
- Run `test_embeddings.py` to verify embeddings are computed
- Check that glyphs have reasonable variation (not all identical)
- May need different model or fine-tuning approach
