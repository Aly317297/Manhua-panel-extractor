# Comic Panel Extractor

A standalone Python CLI tool and library for splitting tall vertical manhua/webtoon strip images (PNG, JPG, WEBP) into individual panels and saving them to an output folder.

It uses multi-pass horizontal gutter detection, CLAHE-boosted orphan chunk merging, saturation/brightness-based splitting for dark panels, and tight final gutter cropping.

---

## Features

- **Tall Strip Support**: Disables Pillow's decompression bomb limit and handles truncated images for ultra-tall vertical strips.
- **Smart Gutter & Edge Merging**: Detects white/uniform gutters, merges split panel fragments using edge similarity, and splits tall dark scenes by brightness transitions.
- **Tight Cropping**: Automatically trims excess whitespace around each extracted panel.
- **Noise Filtering**: Drops tiny sliver artifacts using configurable minimum area and ribbon-size thresholds.
- **CLI & Python API**: Run directly from the terminal or import into your own scripts.

---

## Installation

### Option 1: Install as a CLI package (recommended)

```bash
git clone <your-repo-url>
cd manhua-panel-extractor
pip install -e .
```

### Option 2: Install dependencies only

```bash
git clone <your-repo-url>
cd manhua-panel-extractor
pip install -r requirements.txt
```

---

## CLI Usage

Extract panels from a long PNG strip into a folder:

```bash
# Outputs to ./chapter_1_panels/ by default
extract-panels chapter_1.png
```

Specify a custom output folder and extraction options:

```bash
extract-panels chapter_1.png -o ./extracted_panels --format png --aggression 0.5 --min-area 30000
```

If you didn't install the package with `pip install -e .`, run the module directly:

```bash
python -m panel_extractor.cli chapter_1.png -o ./extracted_panels
```

### CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `input` | *(required)* | Path to the input strip image (`.png`, `.jpg`, `.webp`). |
| `-o`, `--output` | `./<input_stem>_panels` | Output directory where extracted panels will be saved. |
| `--aggression` | `0.5` | Gutter detection sensitivity (`0.0` to `1.0`). Higher values split more aggressively. |
| `--ribbon-size` | `0.3` | Minimum panel area ratio relative to the average panel area. |
| `--min-area` | `30000` | Minimum pixel area (`width * height`) required to keep a panel. |
| `--format` | `png` | Output image format (`png`, `jpg`, or `webp`). |
| `--prefix` | `panel` | Filename prefix (`panel_001.png`, `panel_002.png`, ...). |

---

## Python API

### Extract directly to a folder

```python
from panel_extractor import extract_panels_to_folder

saved_paths = extract_panels_to_folder(
    image_path="chapter_1.png",
    output_dir="output_panels",
    aggression=0.5,
    ribbon_size=0.3,
    min_panel_area=30000,
    output_format="png",
    prefix="panel",
)

print(f"Saved {len(saved_paths)} panels.")
```

### Work with PIL Images in memory

```python
from PIL import Image
from panel_extractor import extract_panels_from_image

with Image.open("chapter_1.png") as img:
    panels = extract_panels_from_image(img, aggression=0.5, ribbon_size=0.3)

for i, panel in enumerate(panels, start=1):
    print(f"Panel {i}: {panel.size}")
```

---

## Requirements

- Python 3.9+
- `opencv-python`
- `numpy`
- `Pillow`

---

## License

[MIT License](LICENSE)
