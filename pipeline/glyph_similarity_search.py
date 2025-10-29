import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import argparse
from config import PipelineConfig

class GlyphSimilaritySearch:
    def __init__(self, embeddings_file, metadata_csv, glyphs_dir):
        """
        Initialize similarity search with precomputed embeddings.
        
        Args:
            embeddings_file: Path to glyph_embeddings.npz
            metadata_csv: Path to glyph_metadata_with_embeddings.csv
            glyphs_dir: Path to directory containing glyph PNG files
        """
        # Load embeddings
        data = np.load(embeddings_file)
        self.embeddings = data['embeddings']
        self.glyph_ids = list(data['glyph_ids'])
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_csv)
        self.glyphs_dir = Path(glyphs_dir)
        
        # Create a mapping from glyph_id to index
        self.glyph_id_to_idx = {gid: idx for idx, gid in enumerate(self.glyph_ids)}
    
    def find_similar(self, glyph_id, top_k, visualize_path=None):
        """
        Find the top-k most similar glyphs to a query glyph.
        
        Args:
            glyph_id: ID of the query glyph (e.g., 'glyph_000001')
            top_k: Number of similar glyphs to return (mandatory)
            visualize_path: Optional path to save a visual grid of results
            
        Returns:
            List of glyph IDs for the top-k most similar glyphs (includes query itself)
        """
        if glyph_id not in self.glyph_id_to_idx:
            raise ValueError(f"Glyph {glyph_id} not found in embeddings")
        
        # Get query embedding
        query_idx = self.glyph_id_to_idx[glyph_id]
        query_embedding = self.embeddings[query_idx:query_idx+1]
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        similar_glyph_ids = [self.glyph_ids[idx] for idx in top_indices]
        
        # Generate visualization if requested
        if visualize_path is not None:
            self._visualize_results(similar_glyph_ids, top_indices, similarities, visualize_path)
        
        return similar_glyph_ids
    
    def _visualize_results(self, glyph_ids, indices, similarities, output_path):
        """
        Create and save a visual grid of the query glyph and its similar matches.
        """
        # Load glyph images
        glyph_images = []
        for idx, glyph_id in zip(indices, glyph_ids):
            glyph_meta = self.metadata[self.metadata['glyph_id'] == glyph_id].iloc[0]
            glyph_path = self.glyphs_dir / glyph_meta['filename']
            
            if glyph_path.exists():
                img = cv2.imread(str(glyph_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    glyph_images.append(img)
                else:
                    glyph_images.append(None)
            else:
                glyph_images.append(None)
        
        # Create grid layout
        num_results = len(glyph_images)
        cols = min(5, num_results)  # Max 5 columns
        rows = (num_results + cols - 1) // cols
        
        # Create canvas
        cell_size = 100
        grid = np.ones((rows * cell_size, cols * cell_size), dtype=np.uint8) * 255
        
        # Place images in grid
        for i, img in enumerate(glyph_images):
            row = i // cols
            col = i % cols
            
            y_start = row * cell_size
            x_start = col * cell_size
            y_end = y_start + cell_size
            x_end = x_start + cell_size
            
            if img is not None:
                # Resize and center image within cell
                h, w = img.shape
                scale = min((cell_size - 10) / h, (cell_size - 10) / w)
                new_h, new_w = int(h * scale), int(w * scale)
                resized = cv2.resize(img, (new_w, new_h))
                
                # Center in cell
                y_offset = (cell_size - new_h) // 2
                x_offset = (cell_size - new_w) // 2
                
                grid[y_start + y_offset:y_start + y_offset + new_h,
                     x_start + x_offset:x_start + x_offset + new_w] = resized
        
        # Save visualization
        cv2.imwrite(str(output_path), grid)


def main():
    """Command-line interface for glyph similarity search"""
    
    parser = argparse.ArgumentParser(
        description='Find similar glyphs using embeddings (configured via config file)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python glyph_similarity_search.py gospel_config.ini
  python glyph_similarity_search.py /path/to/my_manuscript_config.ini
        '''
    )
    
    parser.add_argument(
        'config',
        type=str,
        nargs='?',
        default='gospel_config.ini',
        help='Path to configuration file (default: gospel_config.ini)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = PipelineConfig(args.config)
        config.print_config()
        config.ensure_directories_exist()
        
        # Get paths and search parameters
        paths = config.get_paths()
        search_params = config.get_search_params()
        
        # Check if a query glyph is specified
        query_glyph_id = search_params['query_glyph_id']
        if not query_glyph_id:
            print("\nNo query_glyph_id specified in config file.")
            print("Update the [search] section with a glyph ID to search for.")
            return
        
        # Prepare file paths
        embeddings_file = paths['embeddings_dir'] / 'glyph_embeddings.npz'
        metadata_csv = paths['embeddings_dir'] / 'glyph_metadata_with_embeddings.csv'
        glyphs_dir = paths['components_dir'] / 'glyphs'
        
        # Initialize search
        search = GlyphSimilaritySearch(str(embeddings_file), str(metadata_csv), str(glyphs_dir))
        
        # Perform search
        print(f"\nSearching for {search_params['top_k']} glyphs similar to {query_glyph_id}...")
        similar_ids = search.find_similar(
            query_glyph_id, 
            search_params['top_k'],
            visualize_path=None  # Will set if visualization is enabled
        )
        
        # Print results
        print(f"\nResults (ranked by similarity):")
        print("-" * 40)
        for i, glyph_id in enumerate(similar_ids, 1):
            print(f"{i:2d}. {glyph_id}")
        
        # Generate visualization if enabled
        if search_params['generate_visualization']:
            viz_filename = f"similar_to_{query_glyph_id}.png"
            viz_path = str(paths['visualizations_dir'] / viz_filename)
            
            print(f"\nGenerating visualization...")
            # Re-run search with visualization
            similar_ids = search.find_similar(
                query_glyph_id, 
                search_params['top_k'],
                visualize_path=viz_path
            )
            print(f"✓ Visualization saved to: {viz_path}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Make sure the config file, embeddings, and metadata files exist.")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()