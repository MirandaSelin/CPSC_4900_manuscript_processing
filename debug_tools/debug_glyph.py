import cv2
import numpy as np
from glyph_extraction import GlyphExtractor

def debug_line_processing():
    """Debug function to examine line processing"""
    
    # Create extractor
    extractor = GlyphExtractor(
        min_area=20,
        max_area=5000,
        threshold_value=127,
        padding=5
    )
    
    # Load image and parse XML
    manuscript_img = cv2.imread("gospels.tiff")
    lines = extractor.parse_kraken_xml("segmentation.xml")
    
    print(f"Found {len(lines)} lines")
    
    # Test first few lines
    for i, line_info in enumerate(lines[:5]):
        line_id = line_info['line_id']
        polygon = line_info['polygon']
        
        print(f"\n--- Line {i}: {line_id} ---")
        print(f"Polygon has {len(polygon)} points")
        
        # Crop line
        line_img, line_offset = extractor.crop_line(manuscript_img, polygon)
        print(f"Cropped line shape: {line_img.shape}")
        
        # Save cropped line for inspection
        cv2.imwrite(f"debug_line_{i}_cropped.png", line_img)
        
        # Preprocess
        binary = extractor.preprocess_line(line_img)
        print(f"Binary image stats - mean: {np.mean(binary):.1f}, unique values: {np.unique(binary)}")
        
        # Save binary image
        cv2.imwrite(f"debug_line_{i}_binary.png", binary)
        
        # Extract components
        components = extractor.extract_components(binary)
        print(f"Found {len(components)} components")
        
        # Print component details
        for j, comp in enumerate(components):
            area = comp['area']
            bbox = comp['bbox']
            print(f"  Component {j}: area={area}, bbox={bbox}")
            
        # Check if threshold is good by showing some pixel values
        if line_img.shape[0] > 0 and line_img.shape[1] > 0:
            gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY) if len(line_img.shape) == 3 else line_img
            print(f"Gray image stats - min: {gray.min()}, max: {gray.max()}, mean: {gray.mean():.1f}")
            # Sample some pixels
            sample_pixels = gray.flatten()[:100] 
            print(f"Sample pixel values: {sample_pixels[:10]}")

if __name__ == "__main__":
    debug_line_processing()