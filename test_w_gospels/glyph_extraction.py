import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import csv
import os
from pathlib import Path

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
        Preprocess line image for connected component analysis.
        
        Returns:
            Binary image (black text on white background)
        """
        # Convert to grayscale if needed
        if len(line_image.shape) == 3:
            gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_image.copy()
        
        # Try multiple thresholding approaches to find the best one
        
        # Method 1: Otsu's automatic threshold
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive threshold
        binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
        
        # Method 3: Fixed threshold (original approach)
        _, binary_fixed = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        # Choose the best threshold based on the amount of white vs black pixels
        # Good segmentation should have more white (background) than black (text)
        candidates = [
            ('otsu', binary_otsu),
            ('adaptive', binary_adaptive), 
            ('fixed', binary_fixed),
            ('otsu_inv', cv2.bitwise_not(binary_otsu)),
            ('adaptive_inv', cv2.bitwise_not(binary_adaptive)),
            ('fixed_inv', cv2.bitwise_not(binary_fixed))
        ]
        
        best_binary = None
        best_score = -1
        
        for name, candidate in candidates:
            # Score based on having reasonable amount of foreground pixels (5-50% black)
            black_ratio = 1 - np.mean(candidate) / 255.0
            if 0.05 <= black_ratio <= 0.5:  # 5-50% black pixels seems reasonable for text
                # Prefer less black (cleaner background)
                score = 1 - black_ratio
                if score > best_score:
                    best_score = score
                    best_binary = candidate
        
        # If no candidate meets criteria, fall back to Otsu with appropriate inversion
        if best_binary is None:
            binary = binary_otsu
            # If most pixels are black, invert (we want black text on white background)
            if np.mean(binary) < 127:
                binary = cv2.bitwise_not(binary)
        else:
            binary = best_binary
        
        return binary
    
    def extract_components(self, binary_image):
        """
        Extract connected components from binary image.
        
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
            
            # Extract component image
            component_img = binary_image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            
            components.append({
                'bbox': (x_pad, y_pad, w_pad, h_pad),
                'area': area,
                'image': component_img,
                'centroid': centroids[i]
            })
        
        # Sort components left to right (by x coordinate)
        components.sort(key=lambda c: c['bbox'][0])
        
        return components
    
    def process_manuscript(self, tiff_path, xml_path, output_dir):
        """
        Process entire manuscript: extract all glyphs and save with metadata.
        
        Args:
            tiff_path: Path to manuscript TIFF file
            xml_path: Path to Kraken XML output
            output_dir: Directory to save glyph images and CSV
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        glyph_dir = output_path / "glyphs"
        glyph_dir.mkdir(exist_ok=True)
        
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
        glyph_count = 0
        
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
                
                # Store metadata
                metadata.append({
                    'glyph_id': glyph_id,
                    'filename': glyph_filename,
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
        
        # Save metadata CSV
        csv_path = output_path / "glyph_metadata.csv"
        print(f"\nSaving metadata to {csv_path}")
        
        with open(csv_path, 'w', newline='') as f:
            if metadata:
                writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
                writer.writeheader()
                writer.writerows(metadata)
        
        print(f"\nComplete! Extracted {glyph_count} glyphs from {len(lines)} lines")
        print(f"Glyphs saved to: {glyph_dir}")
        print(f"Metadata saved to: {csv_path}")
        
        return metadata


# Example usage
if __name__ == "__main__":
    # Configuration
    TIFF_PATH = "gospels.tiff"  # Replace with your TIFF file path
    XML_PATH = "segmentation.xml"    # Replace with your XML file path
    OUTPUT_DIR = "components"          # Directory for results
    
    # Create extractor with default parameters
    # Adjust these based on your manuscript:
    extractor = GlyphExtractor(
        min_area=20,           # Minimum glyph size (filters noise)
        max_area=5000,         # Maximum glyph size (filters merged text)
        threshold_value=127,   # Binarization threshold (127 is middle)
        padding=5              # Pixels around each glyph
    )
    
    # Process manuscript
    metadata = extractor.process_manuscript(TIFF_PATH, XML_PATH, OUTPUT_DIR)
    
    print("\nFirst few glyphs:")
    for i, glyph in enumerate(metadata[:5]):
        print(f"  {glyph['glyph_id']}: line={glyph['line_id']}, "
              f"position={glyph['position_in_line']}, area={glyph['area']}")