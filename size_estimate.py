from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO(
    "/mnt/DATA/quabot/work/github/HSCD/model-weight/"
    "SCCNet/train/weights/best.pt"
)

source = "/mnt/DATA/quabot/work/HRSC/images/val"
output_dir = Path("./runs/segment/blue_green_width")
output_dir.mkdir(parents=True, exist_ok=True)

print("类别映射：", model.names)

# 请根据 model.names 确认类别编号。
CRAB_CLASS_ID = 0

# OpenCV 使用 BGR 颜色。
colors = {
    0: (255, 0, 0),  # 马蹄蟹：蓝色
    1: (0, 255, 0),  # 轨迹：绿色
}
width_color = (0, 255, 255)  # 宽度线：黄色


def measure_width(contour):
    """以最大连通轮廓的最小外接旋转矩形短边作为宽度，单位 px。"""
    if len(contour) < 3:
        return None

    rect = cv2.minAreaRect(contour)
    side_a, side_b = rect[1]

    if min(side_a, side_b) <= 0:
        return None

    width_px = float(min(side_a, side_b))
    length_px = float(max(side_a, side_b))

    corners = cv2.boxPoints(rect).astype(np.float32)
    edges = np.roll(corners, -1, axis=0) - corners
    edge_lengths = np.linalg.norm(edges, axis=1)

    short_index = int(np.argmin(edge_lengths))
    if edge_lengths[short_index] <= 0:
        return None

    direction = edges[short_index] / edge_lengths[short_index]
    center = np.asarray(rect[0], dtype=np.float32)

    point1 = center - direction * width_px / 2.0
    point2 = center + direction * width_px / 2.0

    return {
        "width_px": width_px,
        "length_px": length_px,
        "point1": tuple(np.rint(point1).astype(int)),
        "point2": tuple(np.rint(point2).astype(int)),
    }


def draw_width_label(
    image,
    contour,
    width_px,
    font_scale,
    text_thickness,
    padding,
):
    """优先在马蹄蟹右侧标注宽度，空间不足时改到左侧或上下方。"""
    height, width = image.shape[:2]
    x, y, box_width, box_height = cv2.boundingRect(contour)

    text = f"W: {width_px:.1f} px"
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = font_scale

    # 根据图像大小限制文字和背景框尺寸。
    while True:
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            label_scale,
            text_thickness,
        )
        label_width = text_width + 2 * padding
        label_height = text_height + baseline + 2 * padding

        if (
            label_width <= width
            and label_height <= height
        ) or label_scale <= 0.1:
            break

        label_scale *= 0.9

    gap = padding
    center_y = y + box_height // 2

    # 优先右侧，其次左侧、上方、下方。
    candidates = [
        (x + box_width + gap, center_y - label_height // 2),
        (x - label_width - gap, center_y - label_height // 2),
        (x, y - label_height - gap),
        (x, y + box_height + gap),
    ]

    position = None
    for left, top in candidates:
        if (
            left >= 0
            and top >= 0
            and left + label_width <= width
            and top + label_height <= height
        ):
            position = (left, top)
            break

    # 图像边缘空间不足时，将标注限制在画面内。
    if position is None:
        left, top = candidates[0]
        left = max(0, min(left, width - label_width))
        top = max(0, min(top, height - label_height))
        position = (left, top)

    left, top = position

    # 黑色背景、黄色宽度文字。
    cv2.rectangle(
        image,
        (left, top),
        (left + label_width, top + label_height),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        image,
        text,
        (left + padding, top + padding + text_height),
        font,
        label_scale,
        width_color,
        text_thickness,
        cv2.LINE_AA,
    )


results = model.predict(
    source=source,
    device="0",
    save=False,
    imgsz=640,
    conf=0.5,  # 保留预测筛选阈值，但不显示置信度。
    iou=0.5,
    retina_masks=True,
    stream=True,
    show_conf=False,
    show_labels=False,
    visualize=False,
    verbose=False,
)

csv_path = output_dir / "horseshoe_crab_widths.csv"

with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        "image",
        "crab_id",
        "width_px",
        "length_px",
        "image_width_px",
        "image_height_px",
    ])

    for result in results:
        image = result.orig_img.copy()
        height, width = image.shape[:2]
        annotations = []
        crab_count = 0

        # 按原图分辨率调整绘图尺寸。
        display_scale = max(height, width) / 1280.0
        font_scale = max(0.7, 0.9 * display_scale)
        text_thickness = max(2, round(2 * display_scale))
        contour_thickness = max(2, round(2 * display_scale))
        padding = max(5, round(6 * display_scale))
        alpha = 0.45

        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for mask, class_id in zip(masks, classes):
                class_id = int(class_id)
                color = colors.get(class_id, (0, 255, 255))

                if mask.shape != (height, width):
                    mask = cv2.resize(
                        mask,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )

                binary_mask = (mask > 0.5).astype(np.uint8)
                if not binary_mask.any():
                    continue

                # 半透明分割掩膜。
                foreground = binary_mask.astype(bool)
                image[foreground] = (
                    image[foreground].astype(np.float32) * (1 - alpha)
                    + np.asarray(color, dtype=np.float32) * alpha
                ).astype(np.uint8)

                contours, _ = cv2.findContours(
                    binary_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                if not contours:
                    continue

                largest = max(contours, key=cv2.contourArea)
                measurement = None

                # 只测量马蹄蟹，轨迹不测量、不显示文字。
                if class_id == CRAB_CLASS_ID:
                    crab_count += 1
                    measurement = measure_width(largest)

                    writer.writerow([
                        Path(result.path).name,
                        crab_count,
                        (
                            f"{measurement['width_px']:.2f}"
                            if measurement is not None else ""
                        ),
                        (
                            f"{measurement['length_px']:.2f}"
                            if measurement is not None else ""
                        ),
                        width,
                        height,
                    ])

                    if measurement is not None:
                        print(
                            f"{Path(result.path).name} | "
                            f"马蹄蟹 {crab_count} | "
                            f"宽度：{measurement['width_px']:.1f} px"
                        )

                annotations.append({
                    "contours": contours,
                    "largest": largest,
                    "color": color,
                    "measurement": measurement,
                })

        # 先绘制所有轮廓，避免后续掩膜覆盖。
        for item in annotations:
            cv2.drawContours(
                image,
                item["contours"],
                -1,
                item["color"],
                thickness=contour_thickness,
            )

        # 绘制马蹄蟹宽度线及端点。
        for item in annotations:
            measurement = item["measurement"]
            if measurement is None:
                continue

            p1 = measurement["point1"]
            p2 = measurement["point2"]

            cv2.line(
                image,
                p1,
                p2,
                width_color,
                thickness=contour_thickness,
                lineType=cv2.LINE_AA,
            )

            for point in (p1, p2):
                cv2.circle(
                    image,
                    point,
                    radius=max(3, contour_thickness * 2),
                    color=width_color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

        # 最后在马蹄蟹旁标注宽度。
        for item in annotations:
            measurement = item["measurement"]
            if measurement is None:
                continue

            draw_width_label(
                image=image,
                contour=item["largest"],
                width_px=measurement["width_px"],
                font_scale=font_scale,
                text_thickness=text_thickness,
                padding=padding,
            )

        # 保留原文件名及后缀，避免不同格式的同名图片相互覆盖。
        output_path = output_dir / f"{Path(result.path).name}.png"

        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"保存失败：{output_path}")

        print(f"已保存：{output_path}，马蹄蟹数量：{crab_count}")

print(f"结果目录：{output_dir.resolve()}")
print(f"宽度数据：{csv_path.resolve()}")