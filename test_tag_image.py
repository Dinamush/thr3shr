#!/usr/bin/env python3
"""
Test script for ML-Danbooru tagging on an image.
Usage: python test_tag_image.py [image_path]
"""

import sys
from pathlib import Path

# Default image path (placeholder: set your own or pass as argument)
DEFAULT_IMAGE = "path/to/your/image.jpg"


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    path = Path(image_path)

    if not path.exists():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    print(f"Loading and tagging: {image_path}")
    print("-" * 60)

    try:
        from imgutils.tagging import get_mldanbooru_tags
    except ImportError:
        print("Error: dghs-imgutils is not installed. Run: pip install dghs-imgutils")
        sys.exit(1)

    tags = get_mldanbooru_tags(
        str(path),
        threshold=0.5,
        size=448,
        keep_ratio=True,
        drop_overlap=True,
        use_real_name=False,
    )

    if isinstance(tags, list):
        for item in tags:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                print(f"  {item[0]}: {item[1]:.3f}")
            else:
                print(f"  {item}")
    elif isinstance(tags, dict):
        for tag, score in tags.items():
            print(f"  {tag}: {score:.3f}")
    else:
        print(tags)

    print("-" * 60)
    print(f"Done. {len(tags) if hasattr(tags, '__len__') else '?'} tag(s).")


if __name__ == "__main__":
    main()
