import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import argparse
import base64
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
            Tuple of (glyph_ids, indices, similarities) for the top-k results
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
        
        return similar_glyph_ids, top_indices, similarities
    
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
    
    def generate_html_report(self, glyph_ids, indices, similarities, output_path, source_images_dir):
        """
        Generate an interactive HTML report showing similar glyphs with context.
        
        Args:
            glyph_ids: List of similar glyph IDs
            indices: Indices in the embeddings array
            similarities: Similarity scores
            output_path: Path to save HTML file
            source_images_dir: Directory containing source TIFF/image files
        """
        source_images_dir = Path(source_images_dir)
        
        # Collect data for each glyph
        results_data = []
        for idx, glyph_id in zip(indices, glyph_ids):
            glyph_meta = self.metadata[self.metadata['glyph_id'] == glyph_id].iloc[0]
            
            # Get similarity score
            similarity_score = similarities[idx]
            
            # Encode glyph thumbnail as base64
            glyph_path = self.glyphs_dir / glyph_meta['filename']
            glyph_b64 = self._image_to_base64(glyph_path)
            
            # Generate context image (glyph highlighted in source)
            context_b64 = self._generate_context_image(glyph_meta, source_images_dir)
            
            results_data.append({
                'glyph_id': glyph_id,
                'similarity': f"{similarity_score:.4f}",
                'source_page': glyph_meta['source_page'],
                'line_index': glyph_meta['line_index'],
                'position_in_line': glyph_meta['position_in_line'],
                'coordinates': f"({int(glyph_meta['x_global'])}, {int(glyph_meta['y_global'])})",
                'size': f"{int(glyph_meta['width'])}×{int(glyph_meta['height'])}",
                'glyph_img': glyph_b64,
                'context_img': context_b64
            })
        
        # Generate HTML
        html = self._create_html_template(results_data, glyph_ids[0])
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def _image_to_base64(self, image_path):
        """Convert image file to base64 string for embedding in HTML."""
        if not image_path.exists():
            return None
        
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _generate_context_image(self, glyph_meta, source_images_dir):
        """
        Generate a context image showing the glyph highlighted in its source manuscript page.
        
        Args:
            glyph_meta: Row from metadata DataFrame
            source_images_dir: Directory containing source images
            
        Returns:
            Base64 encoded image string
        """
        # Find source image
        source_filename = glyph_meta['source_image']
        source_path = source_images_dir / source_filename
        
        # Try different extensions if not found
        if not source_path.exists():
            for ext in ['.tiff', '.tif', '.png', '.jpg']:
                alt_path = source_images_dir / (Path(source_filename).stem + ext)
                if alt_path.exists():
                    source_path = alt_path
                    break
        
        if not source_path.exists():
            return None
        
        # Load source image
        source_img = cv2.imread(str(source_path))
        if source_img is None:
            return None
        
        # Extract glyph coordinates
        x = int(glyph_meta['x_global'])
        y = int(glyph_meta['y_global'])
        w = int(glyph_meta['width'])
        h = int(glyph_meta['height'])
        
        # Define context region (larger area around glyph)
        context_padding = 100
        x1 = max(0, x - context_padding)
        y1 = max(0, y - context_padding)
        x2 = min(source_img.shape[1], x + w + context_padding)
        y2 = min(source_img.shape[0], y + h + context_padding)
        
        # Crop context region
        context_region = source_img[y1:y2, x1:x2].copy()
        
        # Draw rectangle around glyph (adjusted for cropped region)
        glyph_x1 = x - x1
        glyph_y1 = y - y1
        glyph_x2 = glyph_x1 + w
        glyph_y2 = glyph_y1 + h
        
        # Draw red rectangle to highlight the glyph
        cv2.rectangle(context_region, (glyph_x1, glyph_y1), (glyph_x2, glyph_y2), (0, 0, 255), 3)
        
        # Resize if too large (max 800px wide)
        max_width = 800
        if context_region.shape[1] > max_width:
            scale = max_width / context_region.shape[1]
            new_width = max_width
            new_height = int(context_region.shape[0] * scale)
            context_region = cv2.resize(context_region, (new_width, new_height))
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', context_region)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _create_html_template(self, results_data, query_glyph_id):
        """Create the HTML template with embedded images and interactivity."""
        
        # Build result cards HTML
        cards_html = ""
        for i, result in enumerate(results_data, 1):
            is_query = (i == 1)  # First result is the query itself
            card_class = "result-card query-card" if is_query else "result-card"
            
            context_img_html = ""
            if result['context_img']:
                context_img_html = f'<img src="data:image/png;base64,{result["context_img"]}" alt="Context">'
            else:
                context_img_html = '<p class="no-context">Context image not available</p>'
            
            glyph_img_html = f'<img src="data:image/png;base64,{result["glyph_img"]}" alt="{result["glyph_id"]}">' if result['glyph_img'] else '<p>No image</p>'
            
            cards_html += f'''
            <div class="{card_class}">
                <div class="rank">#{i}</div>
                <div class="card-content">
                    <div class="glyph-thumbnail">
                        {glyph_img_html}
                    </div>
                    <div class="metadata">
                        <h3>{result['glyph_id']} {'(Query)' if is_query else ''}</h3>
                        <p><strong>Similarity:</strong> {result['similarity']}</p>
                        <p><strong>Source:</strong> {result['source_page']}</p>
                        <p><strong>Position:</strong> Line {result['line_index']}, Glyph {result['position_in_line']}</p>
                        <p><strong>Coordinates:</strong> {result['coordinates']}</p>
                        <p><strong>Size:</strong> {result['size']}</p>
                        <button class="toggle-context-btn" onclick="toggleContext(this)">Show Context</button>
                    </div>
                </div>
                <div class="context-view" style="display: none;">
                    {context_img_html}
                    <p class="context-caption">Red box shows glyph location in manuscript</p>
                </div>
            </div>
            '''
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Glyph Similarity Search Results - {query_glyph_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            color: #666;
            font-size: 16px;
        }}
        
        .results {{
            display: grid;
            gap: 20px;
        }}
        
        .result-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .result-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }}
        
        .query-card {{
            border: 3px solid #667eea;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        
        .rank {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-bottom: 15px;
        }}
        
        .query-card .rank {{
            background: #764ba2;
        }}
        
        .card-content {{
            display: grid;
            grid-template-columns: 150px 1fr;
            gap: 20px;
            align-items: start;
        }}
        
        .glyph-thumbnail {{
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100px;
        }}
        
        .glyph-thumbnail img {{
            max-width: 100%;
            height: auto;
            display: block;
        }}
        
        .metadata {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .metadata h3 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 18px;
        }}
        
        .metadata p {{
            color: #666;
            font-size: 14px;
            line-height: 1.6;
        }}
        
        .metadata strong {{
            color: #333;
        }}
        
        .toggle-context-btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: background 0.2s;
            margin-top: 10px;
            width: fit-content;
        }}
        
        .toggle-context-btn:hover {{
            background: #764ba2;
        }}
        
        .toggle-context-btn.active {{
            background: #764ba2;
        }}
        
        .context-view {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            text-align: center;
        }}
        
        .context-view img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        .context-caption {{
            margin-top: 10px;
            color: #666;
            font-style: italic;
            font-size: 14px;
        }}
        
        .no-context {{
            color: #999;
            padding: 40px;
            background: #f5f5f5;
            border-radius: 8px;
        }}
        
        @media (max-width: 768px) {{
            .card-content {{
                grid-template-columns: 1fr;
            }}
            
            .glyph-thumbnail {{
                justify-content: center;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 Glyph Similarity Search Results</h1>
            <p class="subtitle">Query: <strong>{query_glyph_id}</strong> | Found {len(results_data)} similar glyphs (including query)</p>
        </header>
        
        <div class="results">
            {cards_html}
        </div>
    </div>
    
    <script>
        function toggleContext(button) {{
            const card = button.closest('.result-card');
            const contextView = card.querySelector('.context-view');
            
            if (contextView.style.display === 'none') {{
                contextView.style.display = 'block';
                button.textContent = 'Hide Context';
                button.classList.add('active');
            }} else {{
                contextView.style.display = 'none';
                button.textContent = 'Show Context';
                button.classList.remove('active');
            }}
        }}
    </script>
</body>
</html>'''
        
        return html


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
        similar_ids, top_indices, similarities = search.find_similar(
            query_glyph_id, 
            search_params['top_k'],
            visualize_path=None  # Will set if visualization is enabled
        )
        
        # Print results
        print(f"\nResults (ranked by similarity):")
        print("-" * 40)
        for i, (glyph_id, idx) in enumerate(zip(similar_ids, top_indices), 1):
            print(f"{i:2d}. {glyph_id} (similarity: {similarities[idx]:.4f})")
        
        # Generate visualization if enabled
        if search_params['generate_visualization']:
            viz_filename = f"similar_to_{query_glyph_id}.png"
            viz_path = str(paths['visualizations_dir'] / viz_filename)
            
            print(f"\nGenerating visualization...")
            # Re-run search with visualization
            similar_ids, top_indices, similarities = search.find_similar(
                query_glyph_id, 
                search_params['top_k'],
                visualize_path=viz_path
            )
            print(f"✓ Visualization saved to: {viz_path}")
        
        # Generate HTML report if enabled
        if search_params['generate_html_report']:
            html_filename = f"report_{query_glyph_id}.html"
            html_path = str(paths['visualizations_dir'] / html_filename)
            
            # Determine source images directory
            # Check if using IIIF (cached images) or local TIFF
            if 'cache_dir' in paths:
                source_images_dir = paths['cache_dir']
            else:
                source_images_dir = paths['tiff_file'].parent
            
            print(f"\nGenerating interactive HTML report...")
            search.generate_html_report(
                similar_ids,
                top_indices,
                similarities,
                html_path,
                source_images_dir
            )
            print(f"✓ HTML report saved to: {html_path}")
            print(f"  Open it in your browser to view interactive results with context images.")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Make sure the config file, embeddings, and metadata files exist.")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()