#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

def _ns_stripped(tag: str) -> str:
    """Return local tag name without namespace."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def _get_bbox(elem):
    """Extract (x, y, w, h) from ALTO element via attributes or Shape/Polygon if present.
    Returns None if no bbox is found.
    """
    # Direct attributes first
    try:
        x = float(elem.attrib["HPOS"])
        y = float(elem.attrib["VPOS"])
        w = float(elem.attrib["WIDTH"])
        h = float(elem.attrib["HEIGHT"])
        return (x, y, w, h)
    except KeyError:
        pass

    # Try Shape/Polygon with POINTS="x1,y1 x2,y2 ..."
    for child in elem.iter():
        if _ns_stripped(child.tag) == "Polygon":
            pts = child.attrib.get("POINTS", "").strip().split()
            if not pts:
                continue
            xs, ys = [], []
            for p in pts:
                try:
                    x, y = p.split(",")
                    xs.append(float(x)); ys.append(float(y))
                except Exception:
                    pass
            if xs and ys:
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                return (x0, y0, x1 - x0, y1 - y0)
    return None

def _iter_targets(root, level):
    """Yield elements matching the requested level."""
    # Walk all elements and match by local tag name to avoid namespace issues
    target_tag = {"block": "TextBlock", "line": "TextLine", "word": "String"}[level]
    for elem in root.iter():
        if _ns_stripped(elem.tag) == target_tag:
            yield elem

def overlay(alto_path, image_path, level="word", out_path=None, line_width=2):
    tree = ET.parse(alto_path)
    root = tree.getroot()

    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Choose color per level (different alpha). Uses default named colors.
    color_map = {
        "block": (0, 0, 255, 120),   # blue
        "line":  (0, 255, 0, 120),   # green
        "word":  (255, 0, 0, 120),   # red
    }
    color = color_map[level]

    count = 0
    for elem in _iter_targets(root, level):
        bbox = _get_bbox(elem)
        if not bbox:
            continue
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        # Draw rectangle
        draw.rectangle([x, y, x1, y1], outline=color, width=line_width)
        count += 1

    merged = Image.alpha_composite(img, overlay).convert("RGB")

    if out_path:
        merged.save(out_path)
        print(f"Wrote overlay to {out_path} with {count} {level} boxes.")
    else:
        merged.show()
        print(f"Displayed overlay with {count} {level} boxes.")

def main():
    parser = argparse.ArgumentParser(description="Overlay ALTO segments on page image.")
    parser.add_argument("--alto", required=True, help="Path to ALTO XML file")
    parser.add_argument("--image", required=True, help="Path to source page image")
    parser.add_argument("--level", choices=["block", "line", "word"], default="word",
                        help="Granularity to overlay")
    parser.add_argument("--out", help="Output image path. If omitted, opens a viewer window")
    parser.add_argument("--linewidth", type=int, default=2, help="Rectangle border width")
    args = parser.parse_args()
    overlay(args.alto, args.image, level=args.level, out_path=args.out, line_width=args.linewidth)

if __name__ == "__main__":
    main()
