import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import csv
import os
import argparse
from pathlib import Path
from config import PipelineConfig

class GlyphExtractor:
    def __init__(self, 
                 min_area=20, 
                 max_area=5000, 
                 threshold_value=127,
                 padding=5):
        """
        Initialize the glyph extractor.
        
        Args:
            min_area: Minimum pixel area for a component to be considered a glyph
            max_area: Maximum pixel area (to filter out merged glyphs or artifacts)
            threshold_value: Binary threshold value (0-255)
            padding: Extra pixels to add around each glyph bounding box
        """
        self.min_area = min_area
        self.max_area = max_area
        self.threshold_value = threshold_value
        self.padding = padding
    
    def parse_kraken_xml(self, xml_path):
        """
        Parse ALTO XML to extract line boundaries.
        
        Returns:
            List of dicts with line_id and polygon points
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        lines = []
        # Handle ALTO XML format with namespace
        namespaces = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        
        # Try both with and without namespace
        text_lines = root.findall('.//TextLine') or root.findall('.//alto:TextLine', namespaces)
        
        for line in text_lines:
            line_id = line.get('ID')
            
            # Find Shape/Polygon element
            shape = line.find('Shape') or line.find('alto:Shape', namespaces)
            if shape is not None:
                polygon = shape.find('Polygon') or shape.find('alto:Polygon', namespaces)
                if polygon is not None:
                    points_str = polygon.get('POINTS')
                    # Parse points: "x1 y1 x2 y2 x3 y3 ..."
                    points = []
                    coords = points_str.split()
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            x, y = int(coords[i]), int(coords[i + 1])
                            points.append((x, y))
                    
                    lines.append({
                        'line_id': line_id,
                        'polygon': points
                    })
        
        return lines
    
    def polygon_to_bbox(self, polygon):
        """Convert polygon points to bounding box (x, y, w, h)."""
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return x_min, y_min, x_max - x_min, y_max - y_min
    
    def crop_line(self, image, polygon):
        """
        Crop a line from the image using the polygon boundary.
        Returns the cropped line image.
        """
        # Get bounding box
        x, y, w, h = self.polygon_to_bbox(polygon)
        
        # Crop the region
        cropped = image[y:y+h, x:x+w]
        
        return cropped, (x, y)
    
    def preprocess_line(self, line_image):
        """
        Preprocess line image using Otsu's automatic thresholding.
        
        Otsu's method automatically finds the optimal threshold value
        by analyzing the histogram of the image.
        
        Returns:
            Binary image (black text on white background)
        """
        # Convert to grayscale if needed
        if len(line_image.shape) == 3:
            gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_image.copy()
        
        # Use Otsu's automatic threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # If most pixels are black, invert (we want black text on white background)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        return binary
    
    def extract_components(self, binary_image):
        """
        Extract connected components from binary image using mask-based cropping.
        
        This method creates a mask for each component to ensure only pixels
        belonging to that specific component are saved, eliminating stray
        pixels from neighboring letters.
        
        Returns:
            List of dicts with component info (bbox, area, image)
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cv2.bitwise_not(binary_image),  # OpenCV expects white components on black
            connectivity=8
        )
        
        components = []
        
        # Skip label 0 (background)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding box
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Add padding
            x_pad = max(0, x - self.padding)
            y_pad = max(0, y - self.padding)
            w_pad = min(binary_image.shape[1] - x_pad, w + 2 * self.padding)
            h_pad = min(binary_image.shape[0] - y_pad, h + 2 * self.padding)
            
            # Create mask for this specific component
            # Mask is True only for pixels belonging to this component
            component_mask = (labels == i).astype(np.uint8) * 255
            
            # Crop the mask to the padded bounding box
            mask_crop = component_mask[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            
            # Crop the binary image to the padded bounding box
            image_crop = binary_image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            
            # Apply mask: keep only pixels that belong to this component
            # Set everything else to white (255)
            component_img = np.full_like(image_crop, 255)  # Start with all white
            component_img[mask_crop == 255] = image_crop[mask_crop == 255]  # Copy only masked pixels
            
            components.append({
                'bbox': (x_pad, y_pad, w_pad, h_pad),
                'area': area,
                'image': component_img,
                'centroid': centroids[i]
            })
        
        # Sort components left to right (by x coordinate)
        components.sort(key=lambda c: c['bbox'][0])
        
        return components
    
    def process_manuscript(self, tiff_path, xml_path, output_dir, source_page=None, start_glyph_number=1):
        """
        Process entire manuscript: extract all glyphs and save with metadata.
        
        Args:
            tiff_path: Path to manuscript TIFF file
            xml_path: Path to Kraken XML output
            output_dir: Directory to save glyph images and CSV
            source_page: Optional identifier for the source page (e.g., "page_001")
            start_glyph_number: Starting number for glyph IDs (for multi-page processing)
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        glyph_dir = output_path / "glyphs"
        glyph_dir.mkdir(exist_ok=True)
        
        # Determine source identifier
        if source_page is None:
            source_page = Path(tiff_path).stem  # Use filename without extension
        
        # Load manuscript image
        print(f"Loading manuscript image: {tiff_path}")
        manuscript_img = cv2.imread(str(tiff_path))
        if manuscript_img is None:
            raise ValueError(f"Could not load image: {tiff_path}")
        
        # Parse XML
        print(f"Parsing XML: {xml_path}")
        lines = self.parse_kraken_xml(xml_path)
        print(f"Found {len(lines)} lines")
        
        # Process each line
        metadata = []
        glyph_count = start_glyph_number - 1  # Start from the provided number
        
        for line_idx, line_info in enumerate(lines):
            line_id = line_info['line_id']
            polygon = line_info['polygon']
            
            print(f"Processing {line_id}...")
            
            # Crop line
            line_img, line_offset = self.crop_line(manuscript_img, polygon)
            
            # Preprocess
            binary = self.preprocess_line(line_img)
            
            # Extract components
            components = self.extract_components(binary)
            
            print(f"  Found {len(components)} glyphs")
            
            # Save each component
            for comp_idx, component in enumerate(components):
                glyph_count += 1
                glyph_id = f"glyph_{glyph_count:06d}"
                
                # Save glyph image
                glyph_filename = f"{glyph_id}.png"
                glyph_path = glyph_dir / glyph_filename
                cv2.imwrite(str(glyph_path), component['image'])
                
                # Calculate global coordinates
                x_local, y_local, w, h = component['bbox']
                x_global = line_offset[0] + x_local
                y_global = line_offset[1] + y_local
                
                # Store metadata (including source page)
                metadata.append({
                    'glyph_id': glyph_id,
                    'filename': glyph_filename,
                    'source_page': source_page,
                    'source_image': Path(tiff_path).name,
                    'source_xml': Path(xml_path).name,
                    'line_id': line_id,
                    'line_index': line_idx,
                    'position_in_line': comp_idx,
                    'x_global': x_global,
                    'y_global': y_global,
                    'x_in_line': x_local,
                    'y_in_line': y_local,
                    'width': w,
                    'height': h,
                    'area': component['area']
                })
        
        # Don't save metadata CSV here for multi-page mode
        # (it will be saved after all pages are processed)
        
        print(f"\nExtracted {len(metadata)} glyphs from {len(lines)} lines")
        
        return metadata


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract glyphs from manuscript images using line segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single-page manuscript
  python glyph_extraction.py gospel_config.ini
  
  # Multi-page manuscript (IIIF)
  python glyph_extraction.py yale_ms402_config.ini

This script automatically detects whether you have:
  - A single page (uses tiff_file and xml_file from config)
  - Multiple pages from IIIF (processes all pages in cache_dir/segmentation/)

For multi-page manuscripts, metadata will include source_page and source_image
fields to track which page each glyph comes from.
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
        
        # Get paths and parameters
        paths = config.get_paths()
        extraction_params = config.get_extraction_params()
        
        # Create extractor with configured parameters
        extractor = GlyphExtractor(
            min_area=extraction_params['min_area'],
            max_area=extraction_params['max_area'],
            padding=extraction_params['padding']
        )
        
        # Check if we're processing multiple pages (IIIF) or single page
        if 'cache_dir' in paths and paths.get('cache_dir'):
            cache_dir = paths['cache_dir']
            seg_dir = cache_dir / 'segmentation'
            
            # Check if segmentation directory exists with XML files
            if seg_dir.exists():
                xml_files = sorted(seg_dir.glob("*.xml"))
                
                if xml_files:
                    # Multi-page mode
                    print(f"\n{'='*60}")
                    print(f"Multi-page mode: Found {len(xml_files)} segmented pages")
                    print(f"{'='*60}\n")
                    
                    components_dir = paths['components_dir']
                    glyph_dir = components_dir / 'glyphs'
                    glyph_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Process each page and collect metadata
                    all_metadata = []
                    next_glyph_number = 1  # Track glyph numbering across pages
                    
                    for i, xml_path in enumerate(xml_files, 1):
                        print(f"[{i}/{len(xml_files)}] Processing {xml_path.name}")
                        
                        try:
                            # Find matching TIFF image
                            stem = xml_path.stem
                            image_path = cache_dir / f"{stem}.tiff"
                            if not image_path.exists():
                                image_path = cache_dir / f"{stem}.tif"
                            
                            if not image_path.exists():
                                print(f"  ✗ No matching image found for {xml_path.name}")
                                continue
                            
                            # Extract page identifier
                            page_id = xml_path.stem
                            
                            # Process this page with continuous numbering
                            page_metadata = extractor.process_manuscript(
                                tiff_path=str(image_path),
                                xml_path=str(xml_path),
                                output_dir=str(components_dir),
                                source_page=page_id,
                                start_glyph_number=next_glyph_number
                            )
                            
                            all_metadata.extend(page_metadata)
                            next_glyph_number += len(page_metadata)  # Update for next page
                            
                            print(f"  ✓ Extracted {len(page_metadata)} glyphs (total so far: {len(all_metadata)})\n")
                            
                        except Exception as e:
                            print(f"  ✗ Error: {e}\n")
                            continue
                    
                    # Save combined metadata (no renumbering needed!)
                    csv_path = components_dir / "glyph_metadata.csv"
                    print(f"{'='*60}")
                    print(f"Saving combined metadata to {csv_path}")
                    print(f"{'='*60}\n")
                    
                    with open(csv_path, 'w', newline='') as f:
                        if all_metadata:
                            writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
                            writer.writeheader()
                            writer.writerows(all_metadata)
                    
                    print(f"\n{'='*60}")
                    print(f"✓ Multi-page extraction complete!")
                    print(f"{'='*60}")
                    print(f"  Total pages:  {len(xml_files)}")
                    print(f"  Total glyphs: {len(all_metadata)}")
                    print(f"  Glyphs dir:   {glyph_dir}")
                    print(f"  Metadata CSV: {csv_path}")
                    
                    # Show sample metadata
                    print(f"\nFirst few glyphs:")
                    for glyph in all_metadata[:5]:
                        print(f"  {glyph['glyph_id']}: page={glyph['source_page']}, "
                              f"line={glyph['line_id']}, position={glyph['position_in_line']}")
                    
                    print(f"\nNext steps:")
                    print(f"  python glyph_embedder.py {args.config}")
                    print(f"  python glyph_similarity_search.py {args.config}")
                    
                    import sys
                    sys.exit(0)
        
        # Single-page mode (fallback)
        print("\nStarting glyph extraction (single-page mode)...")
        metadata = extractor.process_manuscript(
            str(paths['tiff_file']),
            str(paths['xml_file']),
            str(paths['components_dir'])
        )
        
        # Save metadata CSV for single-page mode
        csv_path = paths['components_dir'] / "glyph_metadata.csv"
        print(f"\nSaving metadata to {csv_path}")
        
        with open(csv_path, 'w', newline='') as f:
            if metadata:
                writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
                writer.writeheader()
                writer.writerows(metadata)
        
        print(f"Glyphs saved to: {paths['components_dir'] / 'glyphs'}")
        print(f"Metadata saved to: {csv_path}")
        
        print("\nFirst few glyphs:")
        for i, glyph in enumerate(metadata[:5]):
            print(f"  {glyph['glyph_id']}: line={glyph['line_id']}, "
                  f"position={glyph['position_in_line']}, area={glyph['area']}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Configuration error: {e}")