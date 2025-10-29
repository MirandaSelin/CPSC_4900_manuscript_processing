import cv2
import numpy as np
from glyph_extraction import GlyphExtractor
import os

def extract_components_direct(binary_image, min_area, max_area, padding):
    """
    Extract connected components directly from a binary image 
    (bypasses the extractor's preprocessing)
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
        if area < min_area or area > max_area:
            continue
        
        # Get bounding box
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Add padding
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(binary_image.shape[1] - x_pad, w + 2 * padding)
        h_pad = min(binary_image.shape[0] - y_pad, h + 2 * padding)
        
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

def save_components_for_method(binary_image, components, method_name):
    """Save individual component images for a specific thresholding method"""
    method_dir = f"compare_{method_name}_components"
    os.makedirs(method_dir, exist_ok=True)
    
    for i, component in enumerate(components):
        filename = f"{method_dir}/component_{i:03d}_area_{component['area']}.png"
        cv2.imwrite(filename, component['image'])
    
    print(f"  Saved {len(components)} components to {method_dir}/")
    return method_dir

def compare_thresholding_methods():
    """Compare all 6 thresholding methods on the first text line"""
    
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
    
    if len(lines) < 2:
        print("Need at least 2 lines!")
        return
    
    # Process second line
    line_info = lines[1]
    line_id = line_info['line_id']
    polygon = line_info['polygon']
    
    print(f"Analyzing second line: {line_id}")
    print(f"Polygon has {len(polygon)} points")
    
    # Crop line
    line_img, line_offset = extractor.crop_line(manuscript_img, polygon)
    print(f"Cropped line shape: {line_img.shape}")
    
    # Convert to grayscale
    if len(line_img.shape) == 3:
        gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = line_img.copy()
    
    print(f"Gray image stats - min: {gray.min()}, max: {gray.max()}, mean: {gray.mean():.1f}")
    
    # Save original cropped line
    cv2.imwrite("compare_original.png", line_img)
    cv2.imwrite("compare_gray.png", gray)
    
    # Generate all 6 thresholding methods
    print("\n=== THRESHOLDING METHODS ===")
    
    # Method 1: Otsu's automatic threshold
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 2: Adaptive threshold
    binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
    
    # Method 3: Fixed threshold
    _, binary_fixed = cv2.threshold(gray, extractor.threshold_value, 255, cv2.THRESH_BINARY)
    
    # Create all candidates
    candidates = [
        ('otsu', binary_otsu),
        ('adaptive', binary_adaptive), 
        ('fixed', binary_fixed),
        ('otsu_inv', cv2.bitwise_not(binary_otsu)),
        ('adaptive_inv', cv2.bitwise_not(binary_adaptive)),
        ('fixed_inv', cv2.bitwise_not(binary_fixed))
    ]
    
    # Analyze each candidate
    results = []
    for name, binary in candidates:
        # Calculate black pixel ratio
        black_ratio = 1 - np.mean(binary) / 255.0
        
        # Save binary image
        cv2.imwrite(f"compare_{name}.png", binary)
        
        # Extract components directly from this binary image (bypass extractor's preprocessing)
        components = extract_components_direct(binary, extractor.min_area, extractor.max_area, extractor.padding)
        num_components = len(components)
        
        # Save individual component images for this method
        save_components_for_method(binary, components, name)
        
        # Calculate score (same as in original code)
        score = -1
        passes_filter = False
        if 0.05 <= black_ratio <= 0.5:
            passes_filter = True
            score = 1 - black_ratio
        
        results.append({
            'name': name,
            'black_ratio': black_ratio,
            'black_percent': black_ratio * 100,
            'num_components': num_components,
            'score': score,
            'passes_filter': passes_filter
        })
        
        print(f"\n{name.upper()}:")
        print(f"  Black pixels: {black_ratio * 100:.1f}%")
        print(f"  Components found: {num_components}")
        print(f"  Passes filter (5-50%): {passes_filter}")
        if passes_filter:
            print(f"  Score: {score:.3f}")
        else:
            print(f"  Score: N/A (filtered out)")
    
    # Show which one would be selected
    print("\n=== SELECTION RESULTS ===")
    
    # Find best according to original algorithm
    best_result = None
    best_score = -1
    
    for result in results:
        if result['passes_filter'] and result['score'] > best_score:
            best_score = result['score']
            best_result = result
    
    if best_result:
        print(f"SELECTED METHOD: {best_result['name'].upper()}")
        print(f"  Reason: Cleanest result with {best_result['black_percent']:.1f}% black pixels")
        print(f"  Found {best_result['num_components']} components")
    else:
        print("NO METHOD PASSED FILTER - would fall back to Otsu")
    
    # Show component size distribution for selected method
    if best_result:
        selected_name = best_result['name']
        selected_binary = None
        for name, binary in candidates:
            if name == selected_name:
                selected_binary = binary
                break
        
        if selected_binary is not None:
            components = extractor.extract_components(selected_binary)
            if components:
                areas = [c['area'] for c in components]
                print(f"\nComponent areas: min={min(areas)}, max={max(areas)}, avg={np.mean(areas):.1f}")
                print(f"Area distribution: {sorted(areas)}")
    
    print(f"\n=== FILES SAVED ===")
    print("compare_original.png  - Original cropped line")
    print("compare_gray.png      - Grayscale version") 
    for name, _ in candidates:
        print(f"compare_{name}.png    - {name.replace('_', ' ').title()} method")
    
    print(f"\nOpen these images to visually compare the thresholding results!")
    
    return results

if __name__ == "__main__":
    results = compare_thresholding_methods()