"""
Interactive Manuscript Viewer

Generates an HTML interface for browsing manuscript pages with interactive
glyph highlighting. Users can flip through pages and hover over glyphs to
see their IDs and metadata.

Usage:
    python manuscript_viewer.py config.ini
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import base64
import json
from config import PipelineConfig


class ManuscriptViewer:
    def __init__(self, metadata_csv, glyphs_dir, source_images_dir):
        """
        Initialize the manuscript viewer.
        
        Args:
            metadata_csv: Path to glyph metadata CSV file
            glyphs_dir: Path to directory containing glyph PNG files
            source_images_dir: Directory containing source TIFF/image files
        """
        self.metadata = pd.read_csv(metadata_csv)
        self.glyphs_dir = Path(glyphs_dir)
        self.source_images_dir = Path(source_images_dir)
        
        # Group glyphs by source page
        self.pages = self.metadata.groupby('source_page')
    
    def generate_viewer(self, output_path):
        """
        Generate the interactive HTML viewer.
        
        Args:
            output_path: Path to save the HTML file
        """
        output_path = Path(output_path)
        
        # Prepare data for each page
        pages_data = []
        
        for page_name, page_glyphs in self.pages:
            # Load source image
            page_img_b64, scale_factor = self._load_page_image(page_name)
            
            if not page_img_b64:
                print(f"Warning: Could not load image for page {page_name}")
                continue
            
            # Collect glyph data for this page
            # Scale coordinates if image was resized
            glyphs_data = []
            for _, glyph in page_glyphs.iterrows():
                glyphs_data.append({
                    'id': glyph['glyph_id'],
                    'x': int(glyph['x_global'] * scale_factor),
                    'y': int(glyph['y_global'] * scale_factor),
                    'width': int(glyph['width'] * scale_factor),
                    'height': int(glyph['height'] * scale_factor),
                    'line': int(glyph['line_index']),
                    'position': int(glyph['position_in_line'])
                })
            
            pages_data.append({
                'name': page_name,
                'image': page_img_b64,
                'glyphs': glyphs_data,
                'glyph_count': len(glyphs_data)
            })
        
        # Generate HTML
        html = self._create_html_template(pages_data)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ Interactive viewer generated with {len(pages_data)} pages")
        print(f"  Total glyphs: {sum(p['glyph_count'] for p in pages_data)}")
    
    def _load_page_image(self, page_name):
        """
        Load and encode a page image as base64.
        Returns tuple of (base64_image, scale_factor) where scale_factor is how much
        the image was resized (1.0 if not resized).
        """
        # Try to find the source image
        source_filename = f"{page_name}.tiff"
        source_path = self.source_images_dir / source_filename
        
        # Try different extensions
        if not source_path.exists():
            for ext in ['.tif', '.png', '.jpg', '.jpeg']:
                alt_path = self.source_images_dir / f"{page_name}{ext}"
                if alt_path.exists():
                    source_path = alt_path
                    break
        
        if not source_path.exists():
            return None, 1.0
        
        # Load image
        img = cv2.imread(str(source_path))
        if img is None:
            return None, 1.0
        
        original_width = img.shape[1]
        scale_factor = 1.0
        
        # Resize if too large (max 1200px wide for web display)
        max_width = 1200
        if img.shape[1] > max_width:
            scale_factor = max_width / img.shape[1]
            new_width = max_width
            new_height = int(img.shape[0] * scale_factor)
            img = cv2.resize(img, (new_width, new_height))
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8'), scale_factor
    
    def _create_html_template(self, pages_data):
        """Create the HTML template for the interactive viewer."""
        
        # Convert pages data to JSON for JavaScript
        pages_json = json.dumps(pages_data)
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Manuscript Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            padding: 20px;
            color: white;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            background: rgba(255, 255, 255, 0.1);
            padding: 20px 30px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        h1 {{
            font-size: 28px;
            margin: 0;
        }}
        
        .page-info {{
            font-size: 16px;
            color: #ddd;
        }}
        
        .controls {{
            background: rgba(255, 255, 255, 0.1);
            padding: 15px 30px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        
        .nav-buttons {{
            display: flex;
            gap: 10px;
        }}
        
        button {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: background 0.2s;
        }}
        
        button:hover {{
            background: #45a049;
        }}
        
        button:disabled {{
            background: #666;
            cursor: not-allowed;
            opacity: 0.5;
        }}
        
        .page-select {{
            flex: 1;
            max-width: 400px;
        }}
        
        select {{
            width: 100%;
            padding: 10px;
            border-radius: 5px;
            border: none;
            font-size: 14px;
            background: white;
            cursor: pointer;
        }}
        
        .viewer-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        
        .page-display {{
            position: relative;
            overflow: auto;
            max-height: 80vh;
            background: #f5f5f5;
            border-radius: 5px;
            display: inline-block;
        }}
        
        .image-wrapper {{
            position: relative;
            display: inline-block;
        }}
        
        #manuscript-canvas {{
            display: block;
            max-width: 100%;
            height: auto;
        }}
        
        #glyph-overlays {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}
        
        .glyph-overlay {{
            position: absolute;
            border: 2px solid rgba(76, 175, 80, 0.6);
            transition: all 0.1s;
            cursor: pointer;
            pointer-events: all;
        }}
        
        .glyph-overlay:hover {{
            border-color: #FF5722;
            border-width: 3px;
            background: rgba(255, 87, 34, 0.15);
            z-index: 10;
        }}
        
        .tooltip {{
            position: fixed;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 14px;
            pointer-events: none;
            z-index: 1000;
            display: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        
        .tooltip.visible {{
            display: block;
        }}
        
        .tooltip-id {{
            font-weight: bold;
            color: #4CAF50;
            font-size: 16px;
            margin-bottom: 5px;
        }}
        
        .tooltip-info {{
            color: #ddd;
            font-size: 12px;
        }}
        
        .stats {{
            margin-top: 15px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.05);
            border-radius: 5px;
            color: #333;
        }}
        
        .stats h3 {{
            margin-bottom: 10px;
            color: #1e3c72;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }}
        
        .stat-item {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📜 Interactive Manuscript Viewer</h1>
            <div class="page-info">
                <span id="current-page-name">-</span> | 
                <span id="glyph-count">0</span> glyphs
            </div>
        </header>
        
        <div class="controls">
            <div class="nav-buttons">
                <button id="prev-btn" onclick="previousPage()">← Previous</button>
                <button id="next-btn" onclick="nextPage()">Next →</button>
            </div>
            
            <div class="page-select">
                <select id="page-selector" onchange="goToPage(this.value)">
                    <!-- Populated by JavaScript -->
                </select>
            </div>
        </div>
        
        <div class="viewer-container">
            <div class="page-display">
                <div class="image-wrapper">
                    <img id="manuscript-canvas" src="" alt="Manuscript page">
                    <div id="glyph-overlays"></div>
                </div>
            </div>
            
            <div class="stats">
                <h3>Page Statistics</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="stat-glyphs">0</div>
                        <div class="stat-label">Glyphs on Page</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-lines">0</div>
                        <div class="stat-label">Lines</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-page">1</div>
                        <div class="stat-label">Page Number</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-total">{len(pages_data)}</div>
                        <div class="stat-label">Total Pages</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="tooltip" id="tooltip">
        <div class="tooltip-id"></div>
        <div class="tooltip-info"></div>
    </div>
    
    <script>
        // Load pages data
        const pagesData = {pages_json};
        let currentPageIndex = 0;
        
        // Initialize viewer
        function init() {{
            populatePageSelector();
            loadPage(0);
        }}
        
        function populatePageSelector() {{
            const selector = document.getElementById('page-selector');
            pagesData.forEach((page, index) => {{
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `Page ${{index + 1}}: ${{page.name}} (${{page.glyph_count}} glyphs)`;
                selector.appendChild(option);
            }});
        }}
        
        function loadPage(index) {{
            if (index < 0 || index >= pagesData.length) return;
            
            currentPageIndex = index;
            const page = pagesData[index];
            
            // Update image
            const canvas = document.getElementById('manuscript-canvas');
            canvas.src = 'data:image/png;base64,' + page.image;
            
            // Update page info
            document.getElementById('current-page-name').textContent = page.name;
            document.getElementById('glyph-count').textContent = page.glyph_count;
            document.getElementById('page-selector').value = index;
            
            // Update stats
            document.getElementById('stat-glyphs').textContent = page.glyph_count;
            document.getElementById('stat-lines').textContent = new Set(page.glyphs.map(g => g.line)).size;
            document.getElementById('stat-page').textContent = index + 1;
            
            // Update navigation buttons
            document.getElementById('prev-btn').disabled = index === 0;
            document.getElementById('next-btn').disabled = index === pagesData.length - 1;
            
            // Wait for image to load before creating overlays
            canvas.onload = () => createGlyphOverlays(page.glyphs);
        }}
        
        function createGlyphOverlays(glyphs) {{
            const overlaysContainer = document.getElementById('glyph-overlays');
            overlaysContainer.innerHTML = '';
            
            const canvas = document.getElementById('manuscript-canvas');
            
            // Coordinates in glyphs are already scaled to match the image we're displaying
            // So we can use them directly
            glyphs.forEach(glyph => {{
                const overlay = document.createElement('div');
                overlay.className = 'glyph-overlay';
                
                // Use coordinates directly - they're already scaled
                overlay.style.left = glyph.x + 'px';
                overlay.style.top = glyph.y + 'px';
                overlay.style.width = glyph.width + 'px';
                overlay.style.height = glyph.height + 'px';
                
                // Add hover events
                overlay.addEventListener('mouseenter', (e) => showTooltip(e, glyph));
                overlay.addEventListener('mousemove', (e) => updateTooltipPosition(e));
                overlay.addEventListener('mouseleave', hideTooltip);
                
                overlaysContainer.appendChild(overlay);
            }});
        }}
        
        function showTooltip(event, glyph) {{
            const tooltip = document.getElementById('tooltip');
            const tooltipId = tooltip.querySelector('.tooltip-id');
            const tooltipInfo = tooltip.querySelector('.tooltip-info');
            
            tooltipId.textContent = glyph.id;
            tooltipInfo.textContent = `Line ${{glyph.line}}, Position ${{glyph.position}} | ${{glyph.width}}×${{glyph.height}}px`;
            
            tooltip.classList.add('visible');
            updateTooltipPosition(event);
        }}
        
        function updateTooltipPosition(event) {{
            const tooltip = document.getElementById('tooltip');
            tooltip.style.left = (event.pageX + 15) + 'px';
            tooltip.style.top = (event.pageY + 15) + 'px';
        }}
        
        function hideTooltip() {{
            const tooltip = document.getElementById('tooltip');
            tooltip.classList.remove('visible');
        }}
        
        function nextPage() {{
            if (currentPageIndex < pagesData.length - 1) {{
                loadPage(currentPageIndex + 1);
            }}
        }}
        
        function previousPage() {{
            if (currentPageIndex > 0) {{
                loadPage(currentPageIndex - 1);
            }}
        }}
        
        function goToPage(index) {{
            loadPage(parseInt(index));
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') previousPage();
            if (e.key === 'ArrowRight') nextPage();
        }});
        
        // Initialize on page load
        window.onload = init;
    </script>
</body>
</html>'''
        
        return html


def main():
    """Command-line interface for manuscript viewer"""
    
    parser = argparse.ArgumentParser(
        description='Generate interactive manuscript viewer (configured via config file)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python manuscript_viewer.py ms402_config.ini
  python manuscript_viewer.py /path/to/my_manuscript_config.ini
        '''
    )
    
    parser.add_argument(
        'config',
        type=str,
        nargs='?',
        default='ms402_config.ini',
        help='Path to configuration file (default: ms402_config.ini)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = PipelineConfig(args.config)
        config.print_config()
        
        # Get paths
        paths = config.get_paths()
        
        # Prepare file paths
        metadata_csv = paths['embeddings_dir'] / 'glyph_metadata_with_embeddings.csv'
        glyphs_dir = paths['components_dir'] / 'glyphs'
        
        # Determine source images directory
        if 'cache_dir' in paths:
            source_images_dir = paths['cache_dir']
        else:
            source_images_dir = paths['tiff_file'].parent
        
        # Check if metadata exists
        if not metadata_csv.exists():
            print(f"\nError: Metadata file not found: {metadata_csv}")
            print("Please run glyph_embedder.py first to generate the metadata file.")
            return
        
        # Generate viewer
        output_filename = "manuscript_viewer.html"
        output_path = paths['visualizations_dir'] / output_filename
        
        print(f"\nGenerating interactive manuscript viewer...")
        viewer = ManuscriptViewer(str(metadata_csv), str(glyphs_dir), str(source_images_dir))
        viewer.generate_viewer(str(output_path))
        
        print(f"✓ Viewer saved to: {output_path}")
        print(f"\nOpen it in your browser to explore the manuscript interactively!")
        print(f"  - Hover over glyphs to see their IDs")
        print(f"  - Use arrow keys or buttons to navigate between pages")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
