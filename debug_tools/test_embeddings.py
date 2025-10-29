import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def test_embeddings():
    """Quick test of the generated embeddings"""
    
    # Load embeddings
    print("Loading embeddings...")
    data = np.load("embeddings_output/glyph_embeddings.npz")
    embeddings = data['embeddings']
    glyph_ids = data['glyph_ids']
    
    # Load metadata
    metadata = pd.read_csv("embeddings_output/glyph_metadata_with_embeddings.csv")
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Glyph IDs: {len(glyph_ids)}")
    print(f"Metadata rows: {len(metadata)}")
    
    # Test: Find similar glyphs to the first one
    print(f"\n=== Finding glyphs similar to {glyph_ids[0]} ===")
    
    # Calculate cosine similarities
    query_embedding = embeddings[0:1]  # First glyph
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top 5 most similar (including itself)
    similar_indices = np.argsort(similarities)[::-1][:5]
    
    for i, idx in enumerate(similar_indices):
        glyph_id = glyph_ids[idx]
        similarity = similarities[idx]
        
        # Get metadata for this glyph
        glyph_meta = metadata[metadata['glyph_id'] == glyph_id].iloc[0]
        
        print(f"{i+1}. {glyph_id} (similarity: {similarity:.3f})")
        print(f"   Line: {glyph_meta['line_id'][:20]}...")
        print(f"   Position: {glyph_meta['position_in_line']}, Area: {glyph_meta['area']}")
    
    # Show some statistics
    print(f"\n=== Embedding Statistics ===")
    print(f"Mean embedding magnitude: {np.mean(np.linalg.norm(embeddings, axis=1)):.2f}")
    print(f"Min similarity: {similarities.min():.3f}")
    print(f"Max similarity: {similarities.max():.3f}")
    print(f"Mean pairwise similarity: {similarities.mean():.3f}")

if __name__ == "__main__":
    test_embeddings()