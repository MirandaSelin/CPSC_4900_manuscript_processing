# CPSC_4900_Manuscript_Processing

A pipeline for extracting, embedding, and analyzing individual glyphs from historical manuscripts using computer vision and deep learning.

## Repository Structure

```
.
├── pipeline/              # Core reusable scripts for the full workflow
│   ├── glyph_extraction.py          # Extract glyphs using connected component analysis
│   ├── glyph_embedder.py            # Generate embeddings using ResNet50
│   ├── glyph_similarity_search.py   # Find similar glyphs via cosine similarity
│   └── README.md                    # Usage and API documentation
│
├── debug_tools/           # Analysis and debugging utilities
│   ├── compare_thresholding.py      # Compare different preprocessing methods
│   ├── debug_glyph.py               # Analyze preprocessing on sample lines
│   ├── test_embeddings.py           # Test embeddings and similarity search
│   └── README.md                    # Troubleshooting guide
│
├── examples/              # Example manuscripts and results
│   └── gospels/                     # Gospel manuscript example
│       ├── gospels.tiff             # Original manuscript image
│       ├── segmentation.xml         # Kraken OCR output (text line boundaries)
│       ├── components/              # Extracted glyphs and metadata
│       └── embeddings_output/       # Generated embeddings and metadata
│
├── archive/               # Old/experimental code
│   └── alto_overlay.py              # Experimental visualization script
```

## Quick Start

### 1. Extract Glyphs
```python
from pipeline.glyph_extraction import GlyphExtractor

extractor = GlyphExtractor(min_area=20, max_area=5000)
metadata = extractor.process_manuscript(
    tiff_path="examples/gospels/gospels.tiff",
    xml_path="examples/gospels/segmentation.xml",
    output_dir="examples/gospels/components"
)
```

### 2. Generate Embeddings
```python
from pipeline.glyph_embedder import GlyphEmbedder

embedder = GlyphEmbedder()
embeddings = embedder.extract_embeddings(
    components_dir="examples/gospels/components",
    metadata_csv_path="examples/gospels/components/glyph_metadata.csv"
)
embedder.save_embeddings(embeddings, ...)
```

### 3. Search for Similar Glyphs
```python
from pipeline.glyph_similarity_search import GlyphSimilaritySearch

search = GlyphSimilaritySearch(
    embeddings_file="examples/gospels/embeddings_output/glyph_embeddings.npz",
    metadata_csv="examples/gospels/embeddings_output/glyph_metadata_with_embeddings.csv",
    glyphs_dir="examples/gospels/components/glyphs"
)

results = search.find_similar("glyph_000001", top_k=10)
search.print_results("glyph_000001", results)
```

## Full Documentation

- **Pipeline Scripts**: See `pipeline/README.md`
- **Debug Tools**: See `debug_tools/README.md`
- **Example Manuscript**: See `examples/gospels/` for the Gospel example

## Dependencies

- OpenCV (`opencv-python`)
- PyTorch and TorchVision (`torch`, `torchvision`)
- Scientific computing (`numpy`, `pandas`, `scikit-learn`)
- Image processing (`Pillow`)
- XML parsing (built-in `xml.etree`)

Install all dependencies:
```bash
pip install opencv-python numpy pandas torch torchvision scikit-learn Pillow
```
