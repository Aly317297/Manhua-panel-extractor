from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageFile

# Allow loading very tall manhua/webtoon strips without Pillow bomb errors
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

EXTRACTOR_CONFIG = {
    "gutter_min_valley_depth": 0.90,
    "gutter_min_width": 20,
    "strict_std_dev_threshold": 15,
    "lenient_std_dev_threshold": 35,
    "min_panel_height_after_split": 60,
    "orphan_content_threshold": 20,
    "merge_similarity_threshold": 520,
    "white_edge_threshold": 230,
    "orphan_clahe_clip_limit": 4.0,
    "edge_blur_kernel_size": (9, 3),
    "outlier_height_ratio": 1.55,
    "height_avg_trim": 0.11,
    "aggressive_split_min_height": 22,
    "aggressive_merge_gap_threshold": 100,
    "final_pass_min_thickness": 16,
    "final_pass_saturation_thresh": 10,
    "dark_panel_white_pixel_ratio": 0.12,
    "dark_panel_split_threshold": 15.0,
    "dark_panel_min_height": 200,
    "min_final_panel_area_ratio_of_avg": 0.15,
}


def _to_rgb(img: Image.Image) -> Image.Image:
    """Convert image to RGB, compositing any alpha channel onto a white background."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        return background
    return img.convert("RGB")


def crop_panel_gutters(img: Image.Image, margin: int = 2, aggressive: bool = False) -> Image.Image:
    arr = np.array(_to_rgb(img))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    _, color_mask = cv2.threshold(hsv[:, :, 1], 15, 255, cv2.THRESH_BINARY)
    mid_tones_mask = cv2.inRange(gray, 20, 235)
    combined = cv2.bitwise_or(color_mask, mid_tones_mask)

    k_size = 21 if aggressive else 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    content_rows = np.where(np.any(cleaned > 0, axis=1))[0]

    if len(content_rows) == 0:
        return img

    top, bottom = int(content_rows[0]), int(content_rows[-1])
    top = max(0, top - margin)
    bottom = min(img.height, bottom + margin)

    if bottom <= top:
        return img

    return img.crop((0, top, img.width, bottom))


def get_white_pixel_ratio(panel_img: np.ndarray) -> float:
    if panel_img is None or panel_img.size == 0:
        return 1.0
    gray = cv2.cvtColor(panel_img, cv2.COLOR_BGR2GRAY) if len(panel_img.shape) == 3 else panel_img
    white_pixels = np.sum(gray > 240)
    total_pixels = gray.shape[0] * gray.shape[1]
    return float(white_pixels / total_pixels)


def is_edge_pure_white(edge_img: np.ndarray, white_threshold: float) -> bool:
    if edge_img is None or edge_img.size == 0:
        return False
    gray_edge = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY) if len(edge_img.shape) == 3 else edge_img
    return bool(np.mean(gray_edge) > white_threshold)


def detect_horizontal_gutters(gray_img: np.ndarray, min_width: int, valley_depth: float, std_dev_thresh: float):
    h, w = gray_img.shape
    projection = np.sum(gray_img, axis=1) / (w * 255)
    gutter_lines, in_gutter, start = [], False, 0
    for y, white_ratio in enumerate(projection):
        if white_ratio > valley_depth:
            if not in_gutter:
                start = y
                in_gutter = True
        elif in_gutter:
            if y - start >= min_width and np.std(gray_img[start:y, :]) < std_dev_thresh:
                gutter_lines.append((start, y))
            in_gutter = False
    if in_gutter and h - start >= min_width and np.std(gray_img[start:h, :]) < std_dev_thresh:
        gutter_lines.append((start, h))
    return gutter_lines


def calculate_edge_similarity(img1: np.ndarray, img2: np.ndarray, cfg: dict) -> float:
    c1 = img1.shape[2] if len(img1.shape) == 3 else 1
    c2 = img2.shape[2] if len(img2.shape) == 3 else 1
    if c1 != c2:
        if c1 == 1:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if c2 == 1:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    edge1, edge2 = img1[-1:, :, :], img2[:1, :, :]
    blur_kernel = cfg["edge_blur_kernel_size"]
    edge1, edge2 = cv2.GaussianBlur(edge1, blur_kernel, 0), cv2.GaussianBlur(edge2, blur_kernel, 0)

    w1, w2 = img1.shape[1], img2.shape[1]
    if w1 != w2:
        target_w = max(w1, w2)
        edge1, edge2 = cv2.resize(edge1, (target_w, 1)), cv2.resize(edge2, (target_w, 1))
    return float(np.mean((edge1.astype(np.float32) - edge2.astype(np.float32)) ** 2))


def compute_avg_metric(metric_list: list, trim_ratio: float) -> float:
    if not metric_list:
        return 0.0
    n = len(metric_list)
    if n < 3:
        return float(np.mean(metric_list)) if n > 0 else 0.0
    trim = max(1, int(n * trim_ratio))
    filtered = sorted(metric_list)[trim:-trim]
    return float(np.mean(filtered)) if filtered else float(np.mean(metric_list))


def aggressive_split_panel_with_merge(panel_img: np.ndarray, gutter_lines: list, cfg: dict) -> List[np.ndarray]:
    h, _ = panel_img.shape[:2]
    y_prev, candidates = 0, []
    min_split_height = cfg["aggressive_split_min_height"]

    for y_start, y_end in gutter_lines:
        if y_start - y_prev >= min_split_height:
            candidates.append((y_prev, y_start))
        y_prev = y_end
    if h - y_prev >= min_split_height:
        candidates.append((y_prev, h))

    if not candidates:
        return [panel_img]

    merged = []
    cur_y1, cur_y2 = candidates[0]
    for next_y1, next_y2 in candidates[1:]:
        if next_y1 - cur_y2 <= cfg["aggressive_merge_gap_threshold"]:
            cur_y2 = next_y2
        else:
            merged.append((cur_y1, cur_y2))
            cur_y1, cur_y2 = next_y1, next_y2
    merged.append((cur_y1, cur_y2))

    panels = [panel_img[y1:y2, :] for y1, y2 in merged if y2 - y1 > min_split_height * 2.5]
    return panels if panels else [panel_img]


def line_contains_color(line: np.ndarray, saturation_thresh: int) -> bool:
    if len(line.shape) != 3:
        return False
    hsv_line = cv2.cvtColor(line, cv2.COLOR_BGR2HSV)
    return bool(np.any(hsv_line[:, :, 1] > saturation_thresh))


def final_pass_split(panel_img: np.ndarray, cfg: dict) -> List[np.ndarray]:
    h, _ = panel_img.shape[:2]
    split_points, in_gutter, gutter_start = [], False, 0
    for y in range(h):
        row = panel_img[y : y + 1, :, :]
        if not line_contains_color(row, cfg["final_pass_saturation_thresh"]):
            if not in_gutter:
                gutter_start = y
                in_gutter = True
        elif in_gutter:
            if y - gutter_start >= cfg["final_pass_min_thickness"]:
                split_points.append(gutter_start + (y - gutter_start) // 2)
            in_gutter = False

    if not split_points:
        return [panel_img]

    subpanels, y_prev = [], 0
    for y_split in split_points:
        subpanel = panel_img[y_prev:y_split, :]
        if subpanel.size > 0:
            subpanels.append(subpanel)
        y_prev = y_split
    last_subpanel = panel_img[y_prev:h, :]
    if last_subpanel.size > 0:
        subpanels.append(last_subpanel)
    return subpanels


def split_dark_panel_by_brightness_change(panel_img: np.ndarray, cfg: dict) -> List[np.ndarray]:
    gray = cv2.cvtColor(panel_img, cv2.COLOR_BGR2GRAY)
    h, _ = gray.shape
    row_brightness = np.mean(gray, axis=1)
    brightness_diffs = np.abs(np.diff(row_brightness))

    split_points = []
    candidates = np.where(brightness_diffs > cfg["dark_panel_split_threshold"])[0]

    if len(candidates) > 0:
        last_split = candidates[0]
        split_points.append(last_split)
        for cand in candidates:
            if cand - last_split > 50:
                split_points.append(cand)
                last_split = cand

    if not split_points:
        return [panel_img]

    subpanels, y_prev = [], 0
    for y_split in sorted(split_points):
        subpanel = panel_img[y_prev:y_split, :]
        if subpanel.size > 0:
            subpanels.append(subpanel)
        y_prev = y_split
    last_subpanel = panel_img[y_prev:h, :]
    if last_subpanel.size > 0:
        subpanels.append(last_subpanel)
    return subpanels


def run_gutter_method_for_initial_coords(gray_original: np.ndarray, cfg: dict):
    coords, page_h = [], gray_original.shape[0]
    strict_gutters = detect_horizontal_gutters(
        gray_original,
        cfg["gutter_min_width"],
        cfg["gutter_min_valley_depth"],
        cfg["strict_std_dev_threshold"],
    )

    processed_regions, leftover_chunks, last_y = [], [], 0
    if strict_gutters:
        y_prev = 0
        for y_start, y_end in strict_gutters:
            processed_regions.append((y_prev, y_start))
            y_prev = y_end
        processed_regions.append((y_prev, page_h))

    for y1, y2 in sorted(processed_regions):
        if y1 > last_y:
            leftover_chunks.append((last_y, y1))
        panel_candidate_img = gray_original[y1:y2]
        if panel_candidate_img.size == 0:
            continue

        internal_gutters = detect_horizontal_gutters(
            panel_candidate_img,
            cfg["gutter_min_width"],
            cfg["gutter_min_valley_depth"],
            cfg["lenient_std_dev_threshold"],
        )

        if not internal_gutters and y2 - y1 > cfg["min_panel_height_after_split"]:
            coords.append((y1, y2))
        else:
            leftover_chunks.append((y1, y2))
        last_y = y2

    for y1, y2 in leftover_chunks:
        chunk_h = y2 - y1
        if chunk_h < cfg["min_panel_height_after_split"]:
            continue
        chunk_gray = gray_original[y1:y2]
        lenient_gutters = detect_horizontal_gutters(
            chunk_gray,
            cfg["gutter_min_width"],
            cfg["gutter_min_valley_depth"],
            cfg["lenient_std_dev_threshold"],
        )

        if lenient_gutters:
            y_prev = 0
            for gy1, gy2 in lenient_gutters:
                if gy1 - y_prev > cfg["min_panel_height_after_split"]:
                    coords.append((y1 + y_prev, y1 + gy1))
                y_prev = gy2
            if chunk_h - y_prev > cfg["min_panel_height_after_split"]:
                coords.append((y1 + y_prev, y1 + chunk_h))
        else:
            coords.append((y1, y2))

    return sorted(list(set(coords)))


def extract_panels_from_image(
    img_pil: Image.Image,
    aggression: float = 0.5,
    ribbon_size: float = 0.3,
    config: Optional[dict] = None,
) -> List[Image.Image]:
    """
    Extract individual panels from a tall vertical manhua/webtoon strip.

    Args:
        img_pil: Input PIL Image.
        aggression: Gutter detection sensitivity in [0.0, 1.0]. Default 0.5.
        ribbon_size: Minimum panel area ratio relative to average panel area (e.g. 0.1 - 0.5). Default 0.3.
        config: Optional override dictionary for EXTRACTOR_CONFIG.
    """
    cfg = (config or EXTRACTOR_CONFIG).copy()
    aggression = max(0.0, min(1.0, float(aggression)))
    cfg["min_final_panel_area_ratio_of_avg"] = float(ribbon_size)

    mult = 0.5 + aggression
    cfg["strict_std_dev_threshold"] = 15 * mult
    cfg["lenient_std_dev_threshold"] = 35 * mult
    cfg["orphan_content_threshold"] = max(5, 20 * (2.0 - mult))

    original = cv2.cvtColor(np.array(_to_rgb(img_pil)), cv2.COLOR_RGB2BGR)
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    initial_coords = run_gutter_method_for_initial_coords(gray_original, cfg)
    panels = [{"coords": (y1, y2), "img": original[y1:y2, :]} for y1, y2 in initial_coords if y2 > y1]

    clahe = cv2.createCLAHE(clipLimit=cfg["orphan_clahe_clip_limit"], tileGridSize=(4, 4))
    pass_num = 0
    while True:
        pass_num += 1
        merges_made_this_pass = 0
        i = 0
        while i < len(panels) - 1:
            p_above, p_below = panels[i], panels[i + 1]
            y2_above, y1_below = p_above["coords"][1], p_below["coords"][0]

            if y1_below > y2_above:
                orphan_gray = gray_original[y2_above:y1_below, :]
                if orphan_gray.size == 0:
                    i += 1
                    continue

                boosted_orphan_gray = clahe.apply(orphan_gray)
                if np.std(boosted_orphan_gray) > cfg["orphan_content_threshold"]:
                    orphan_color = original[y2_above:y1_below, :]

                    sim_above = calculate_edge_similarity(p_above["img"], orphan_color, cfg)
                    sim_below = calculate_edge_similarity(orphan_color, p_below["img"], cfg)
                    edge_pab, edge_ot = p_above["img"][-1:, :, :], orphan_color[:1, :, :]
                    edge_ob, edge_pbt = orphan_color[-1:, :, :], p_below["img"][:1, :, :]

                    prevent_up = is_edge_pure_white(edge_pab, cfg["white_edge_threshold"]) and is_edge_pure_white(
                        edge_ot, cfg["white_edge_threshold"]
                    )
                    prevent_down = is_edge_pure_white(edge_ob, cfg["white_edge_threshold"]) and is_edge_pure_white(
                        edge_pbt, cfg["white_edge_threshold"]
                    )

                    merge_up = (sim_above < cfg["merge_similarity_threshold"]) and not prevent_up
                    merge_down = (sim_below < cfg["merge_similarity_threshold"]) and not prevent_down

                    if merge_up or merge_down:
                        if merge_up and (not merge_down or sim_above <= sim_below):
                            new_y1 = p_above["coords"][0]
                            new_y2 = y1_below
                            panels[i]["coords"] = (new_y1, new_y2)
                            panels[i]["img"] = original[new_y1:new_y2, :]
                        else:
                            new_y1 = y2_above
                            new_y2 = p_below["coords"][1]
                            panels[i + 1]["coords"] = (new_y1, new_y2)
                            panels[i + 1]["img"] = original[new_y1:new_y2, :]
                        merges_made_this_pass += 1
            i += 1

        j = 0
        while j < len(panels) - 1:
            if panels[j]["coords"][1] >= panels[j + 1]["coords"][0]:
                new_y1 = panels[j]["coords"][0]
                new_y2 = max(panels[j]["coords"][1], panels[j + 1]["coords"][1])
                panels[j]["coords"] = (new_y1, new_y2)
                panels[j]["img"] = original[new_y1:new_y2, :]
                panels.pop(j + 1)
            else:
                j += 1

        if merges_made_this_pass == 0 or pass_num > 40:
            break

    heights = [p["img"].shape[0] for p in panels]
    avg_height = compute_avg_metric(heights, cfg["height_avg_trim"])
    refined_panels = []
    if avg_height > 0:
        for p_data in panels:
            p_img = p_data["img"]
            if p_img.shape[0] > avg_height * cfg["outlier_height_ratio"]:
                gray_crop = cv2.cvtColor(p_img, cv2.COLOR_BGR2GRAY)
                gutters = detect_horizontal_gutters(
                    gray_crop,
                    cfg["gutter_min_width"],
                    cfg["gutter_min_valley_depth"],
                    cfg["lenient_std_dev_threshold"],
                )
                subpanels = aggressive_split_panel_with_merge(p_img, gutters, cfg)
                for sub in subpanels:
                    if sub.size > 0:
                        refined_panels.append(sub)
            else:
                refined_panels.append(p_img)
    else:
        refined_panels = [p["img"] for p in panels]

    post_pass_panels = []
    for p_img in refined_panels:
        subpanels = final_pass_split(p_img, cfg)
        post_pass_panels.extend(subpanels)

    p_heights = [p.shape[0] for p in post_pass_panels]
    avg_p_height = compute_avg_metric(p_heights, cfg["height_avg_trim"])
    specialist_panels = []
    for p_img in post_pass_panels:
        if p_img.size == 0:
            continue
        is_h_outlier = avg_p_height > 0 and (p_img.shape[0] > avg_p_height * cfg["outlier_height_ratio"])
        is_d_outlier = get_white_pixel_ratio(p_img) < cfg["dark_panel_white_pixel_ratio"]

        if is_h_outlier and is_d_outlier:
            subpanels = split_dark_panel_by_brightness_change(p_img, cfg)
            for sub in subpanels:
                if sub.shape[0] > cfg["dark_panel_min_height"]:
                    specialist_panels.append(sub)
        else:
            specialist_panels.append(p_img)

    final_panels = []
    if len(specialist_panels) > 1:
        panel_areas = [p.shape[0] * p.shape[1] for p in specialist_panels]
        avg_panel_area = compute_avg_metric(panel_areas, cfg["height_avg_trim"])
        min_area_threshold = avg_panel_area * cfg["min_final_panel_area_ratio_of_avg"]
        for p_img in specialist_panels:
            if p_img.shape[0] * p_img.shape[1] > min_area_threshold:
                final_panels.append(p_img)
    else:
        final_panels = specialist_panels

    if not final_panels:
        final_panels.append(original)

    final_pil_panels = []
    for p_img in final_panels:
        rgb_img = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tight_img = crop_panel_gutters(pil_img, margin=2, aggressive=False)
        final_pil_panels.append(tight_img)

    return final_pil_panels


def extract_panels_to_folder(
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    aggression: float = 0.5,
    ribbon_size: float = 0.3,
    min_panel_area: int = 30000,
    output_format: str = "png",
    prefix: str = "panel",
) -> List[Path]:
    """
    Load a long PNG/JPG strip, extract panels, and save them sequentially to output_dir.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = output_format.lower().lstrip(".")
    if ext not in ("png", "jpg", "jpeg", "webp"):
        raise ValueError(f"Unsupported output format: {output_format}")

    with Image.open(image_path) as img:
        panels = extract_panels_from_image(img, aggression=aggression, ribbon_size=ribbon_size)

    saved_paths: List[Path] = []
    panel_idx = 0
    for panel in panels:
        if panel.width * panel.height < min_panel_area:
            continue
        panel_idx += 1
        out_path = output_dir / f"{prefix}_{panel_idx:03d}.{ext}"
        if ext in ("jpg", "jpeg"):
            panel.save(out_path, "JPEG", quality=95)
        elif ext == "webp":
            panel.save(out_path, "WEBP", quality=95)
        else:
            panel.save(out_path, "PNG")
        saved_paths.append(out_path)

    return saved_paths