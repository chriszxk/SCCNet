from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

model = YOLO("/mnt/DATA/quabot/work/github/HSCD/model-weight/SCCNet/train/weights/best.pt")

source = "/mnt/DATA/quabot/work/HRSC/images/val"
output_dir = Path("./runs/segment/blue_green_conf")
output_dir.mkdir(parents=True, exist_ok=True)

print("类别映射：", model.names)

# 请根据 model.names 确认类别编号。
# OpenCV 颜色格式为 BGR。
colors = {
    0: (255, 0, 0),  # 马蹄蟹：蓝色
    1: (0, 255, 0),  # 轨迹：绿色
}

results = model.predict(
    source=source,
    device="0",
    save=False,
    imgsz=640,
    conf=0.5,
    iou=0.5,
    retina_masks=True,
    stream=True,
    visualize=False,
    verbose=False,
)

for result in results:
    image = result.orig_img.copy()
    height, width = image.shape[:2]
    annotations = []

    # 随原图分辨率调整文字大小，避免高分辨率图像中文字太小。
    display_scale = max(height, width) / 1280.0
    font_scale = max(0.7, 0.9 * display_scale)
    text_thickness = max(2, round(2 * display_scale))
    contour_thickness = max(2, round(2 * display_scale))
    padding = max(5, round(6 * display_scale))
    alpha = 0.45

    if result.masks is not None and result.boxes is not None:
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for mask, class_id, confidence in zip(masks, classes, confidences):
            color = colors.get(int(class_id), (0, 255, 255))

            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            binary_mask = (mask > 0.5).astype(np.uint8)
            if not binary_mask.any():
                continue

            foreground = binary_mask.astype(bool)

            image[foreground] = (
                image[foreground].astype(np.float32) * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if contours:
                annotations.append((contours, color, float(confidence)))

        # 先画所有轮廓。
        for contours, color, _ in annotations:
            cv2.drawContours(
                image,
                contours,
                -1,
                color,
                thickness=contour_thickness,
            )

        # 最后统一绘制置信度，避免被后续掩膜覆盖。
        for contours, color, confidence in annotations:
            largest = max(contours, key=cv2.contourArea)
            x, y, _, _ = cv2.boundingRect(largest)

            text = f"{confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX

            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                font,
                font_scale,
                text_thickness,
            )

            label_width = text_width + 2 * padding
            label_height = text_height + baseline + 2 * padding

            left = max(0, min(x, width - label_width))

            # 优先放在目标上方；空间不足则放在目标顶部。
            top = y - label_height - padding
            if top < 0:
                top = y
            top = max(0, min(top, height - label_height))

            # 黑底白字，提高蓝色和绿色掩膜上的可读性。
            cv2.rectangle(
                image,
                (left, top),
                (left + label_width, top + label_height),
                (0, 0, 0),
                thickness=-1,
            )

            # 标签边框沿用对应类别颜色。
            cv2.rectangle(
                image,
                (left, top),
                (left + label_width, top + label_height),
                color,
                thickness=max(1, contour_thickness // 2),
            )

            cv2.putText(
                image,
                text,
                (left + padding, top + padding + text_height),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

    output_path = output_dir / f"{Path(result.path).name}.png"

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"保存失败：{output_path}")

    print(f"已保存：{output_path}，绘制置信度数量：{len(annotations)}")

print(f"结果目录：{output_dir.resolve()}")
