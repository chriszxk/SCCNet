import inspect
import os
import traceback

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ultralytics import YOLO

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

from pytorch_grad_cam import (
    EigenCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    LayerCAM,
    RandomCAM,
    XGradCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image


def letterbox(image, new_shape=640, stride=32):
    """等比例缩放，并填充至 stride 的整数倍。."""
    height, width = image.shape[:2]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio = min(
        new_shape[0] / height,
        new_shape[1] / width,
    )

    resized_width = round(width * ratio)
    resized_height = round(height * ratio)

    dw = (new_shape[1] - resized_width) % stride
    dh = (new_shape[0] - resized_height) % stride

    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

    left = round(dw / 2 - 0.1)
    right = round(dw / 2 + 0.1)
    top = round(dh / 2 - 0.1)
    bottom = round(dh / 2 + 0.1)

    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def print_output_structure(output, name="output", depth=0):
    """只打印类型、形状和梯度状态，不打印张量内容。."""
    indent = "  " * depth

    if isinstance(output, torch.Tensor):
        print(
            f"{indent}{name}: Tensor, "
            f"shape={tuple(output.shape)}, "
            f"dtype={output.dtype}, "
            f"device={output.device}, "
            f"requires_grad={output.requires_grad}"
        )

    elif isinstance(output, dict):
        print(f"{indent}{name}: dict, keys={list(output.keys())}")
        for key, value in output.items():
            print_output_structure(
                value,
                name=str(key),
                depth=depth + 1,
            )

    elif isinstance(output, (tuple, list)):
        print(f"{indent}{name}: {type(output).__name__}, len={len(output)}")
        for index, value in enumerate(output):
            print_output_structure(
                value,
                name=f"[{index}]",
                depth=depth + 1,
            )

    else:
        print(f"{indent}{name}: {type(output).__name__}")


def extract_predictions(model_output, nc):
    """当前 Segment 输出： ( (predictions, proto), auxiliary_dict ).

    predictions: [B, 4 + nc + nm, N] 返回检测部分：[B, 4 + nc, N]
    """
    if not isinstance(model_output, (tuple, list)):
        raise TypeError(f"预期外层输出为 tuple/list，实际为 {type(model_output).__name__}")

    inference_output = model_output[0]

    if not isinstance(inference_output, (tuple, list)):
        raise TypeError("预期 model_output[0] 为 (predictions, proto)。")

    predictions = inference_output[0]

    if not isinstance(predictions, torch.Tensor):
        raise TypeError("model_output[0][0] 不是 Tensor。")

    if predictions.ndim != 3:
        raise ValueError(f"预测张量应为三维，实际为 {tuple(predictions.shape)}")

    if predictions.shape[1] < 4 + nc:
        raise ValueError(f"预测通道不足：{predictions.shape[1]}，至少需要 {4 + nc}。")

    # 仅取框坐标和类别分数，排除掩码系数。
    # 不使用 detach，保留 CAM 所需的梯度链路。
    return predictions[:, : 4 + nc, :]


class ActivationsAndGradients:
    def __init__(self, model, target_layers, nc):
        self.model = model
        self.nc = nc
        self.activations = []
        self.gradients = []
        self.handles = []

        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("目标层输出不是 Tensor，请选择输出为 [B, C, H, W] 的特征层。")

        if output.ndim != 4:
            raise ValueError(f"目标层输出不是四维特征图：{tuple(output.shape)}")

        self.activations.append(output.detach().cpu())

    def save_gradient(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return
        if not output.requires_grad:
            return

        def store_gradient(gradient):
            self.gradients.insert(0, gradient.detach().cpu())

        output.register_hook(store_gradient)

    def __call__(self, x):
        self.activations.clear()
        self.gradients.clear()

        predictions = extract_predictions(
            self.model(x),
            self.nc,
        )

        boxes = predictions[0, :4].transpose(0, 1)
        scores = predictions[0, 4 : 4 + self.nc].transpose(0, 1)

        indices = scores.max(dim=1).values.argsort(descending=True)

        return [[scores[indices], boxes[indices]]]

    def release(self):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()
        self.activations.clear()
        self.gradients.clear()


class YOLOTarget(torch.nn.Module):
    def __init__(self, output_type="class", conf=0.5, ratio=0.02):
        super().__init__()

        if output_type not in {"class", "box", "all"}:
            raise ValueError("backward_type 必须为 class、box 或 all。")

        if not 0 < ratio <= 1:
            raise ValueError("ratio 必须在 (0, 1] 范围内。")

        self.output_type = output_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        scores, boxes = data

        if scores.shape[0] == 0:
            raise RuntimeError("模型没有输出候选目标。")

        count = max(1, int(scores.shape[0] * self.ratio))

        confidence = scores[:count].max(dim=1).values
        selected_boxes = boxes[:count]
        valid = confidence >= self.conf

        if not valid.any().item():
            raise RuntimeError(f"没有置信度达到 {self.conf} 的目标。")

        target = confidence.sum() * 0.0

        if self.output_type in {"class", "all"}:
            target = target + confidence[valid].sum()

        if self.output_type in {"box", "all"}:
            target = target + selected_boxes[valid].sum()

        return target


class YOLOHeatmap:
    def __init__(
        self,
        weight,
        device="cuda:0",
        method="GradCAMPlusPlus",
        layer=(10, 12, 14, 16, 18),
        backward_type="class",
        conf_threshold=0.5,
        ratio=0.02,
        show_box=False,
        renormalize=True,
        img_size=640,
        debug_only=True,
    ):
        self.device = torch.device(device)
        self.debug_only = debug_only
        self.method = None

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请将 device 改为 cpu。")

        if not os.path.isfile(weight):
            raise FileNotFoundError(f"权重不存在：{weight}")

        self.model = YOLO(weight).model
        self.model.to(self.device).float().eval()

        for parameter in self.model.parameters():
            parameter.requires_grad_(True)

        self.model_names = self.model.names
        self.nc = len(self.model_names)
        self.stride = int(self.model.stride.max().item())

        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.show_box = show_box
        self.renormalize = renormalize
        self.layer_indices = list(layer)

        rng = np.random.default_rng(0)
        self.colors = rng.integers(0, 256, size=(self.nc, 3))

        self.model.info()

        # 诊断模式不创建 CAM hooks，直接检查原始模型输出。
        if self.debug_only:
            return

        methods = {
            "GradCAMPlusPlus": GradCAMPlusPlus,
            "GradCAM": GradCAM,
            "XGradCAM": XGradCAM,
            "EigenCAM": EigenCAM,
            "HiResCAM": HiResCAM,
            "LayerCAM": LayerCAM,
            "RandomCAM": RandomCAM,
            "EigenGradCAM": EigenGradCAM,
        }

        if method not in methods:
            raise ValueError(f"不支持的 CAM 方法：{method}")

        self.target = YOLOTarget(
            output_type=backward_type,
            conf=conf_threshold,
            ratio=ratio,
        )

        target_layers = [self.model.model[index] for index in self.layer_indices]

        cam_class = methods[method]
        kwargs = {
            "model": self.model,
            "target_layers": target_layers,
        }

        if "use_cuda" in inspect.signature(cam_class.__init__).parameters:
            kwargs["use_cuda"] = self.device.type == "cuda"

        self.method = cam_class(**kwargs)
        self.method.activations_and_grads.release()

        self.method.activations_and_grads = ActivationsAndGradients(
            self.model,
            target_layers,
            self.nc,
        )

    def load_image(self, path):
        image = cv2.imdecode(
            np.fromfile(path, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(f"无法读取图片：{path}")

        image = letterbox(
            image,
            new_shape=self.img_size,
            stride=self.stride,
        )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_float = image.astype(np.float32) / 255.0

        tensor = torch.from_numpy(np.ascontiguousarray(image_float.transpose(2, 0, 1))).unsqueeze(0).to(self.device)

        return image_float, tensor

    def debug(self, image_path):
        print(f"\n诊断图片：{image_path}", flush=True)
        print(f"PyTorch：{torch.__version__}")
        print(f"NumPy：{np.__version__}")
        print(f"类别：{self.model_names}")

        head = self.model.model[-1]
        print(f"检测头类型：{type(head).__name__}")

        for attribute in ("nc", "nm", "end2end", "export"):
            if hasattr(head, attribute):
                print(f"head.{attribute} = {getattr(head, attribute)}")

        _, tensor = self.load_image(image_path)
        print(f"输入形状：{tuple(tensor.shape)}")

        # 同时记录选定层的输出结构，辅助判断 CAM 层是否合适。
        handles = []

        def make_hook(index):
            def hook(module, inputs, output):
                print_output_structure(
                    output,
                    name=f"model.model[{index}]",
                )

            return hook

        try:
            print("\n========== 目标层输出结构 ==========")
            for index in self.layer_indices:
                handles.append(self.model.model[index].register_forward_hook(make_hook(index)))

            with torch.enable_grad():
                output = self.model(tensor)

            print("\n========== 模型最终输出结构 ==========")
            print_output_structure(output)
            print("=====================================")
            print("\n诊断完成，请复制上面的输出。")

        finally:
            for handle in handles:
                handle.remove()
            self.model.zero_grad(set_to_none=True)

    def normalize_in_boxes(self, boxes, grayscale_cam):
        height, width = grayscale_cam.shape
        normalized = np.zeros_like(grayscale_cam, dtype=np.float32)

        for x1, y1, x2, y2 in boxes:
            x1 = int(np.clip(np.floor(x1), 0, width))
            y1 = int(np.clip(np.floor(y1), 0, height))
            x2 = int(np.clip(np.ceil(x2), 0, width))
            y2 = int(np.clip(np.ceil(y2), 0, height))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = grayscale_cam[y1:y2, x1:x2].copy()
            crop -= crop.min()
            crop /= crop.max() + 1e-7

            normalized[y1:y2, x1:x2] = np.maximum(
                normalized[y1:y2, x1:x2],
                crop,
            )

        return normalized

    def process(self, input_path, output_path):
        image_float, tensor = self.load_image(input_path)
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            grayscale_cam = self.method(
                input_tensor=tensor,
                targets=[self.target],
            )[0]

        if not np.isfinite(grayscale_cam).all():
            raise RuntimeError("CAM 中出现 NaN 或 Inf。")

        detections = None

        if self.show_box or self.renormalize:
            with torch.no_grad():
                predictions = extract_predictions(
                    self.model(tensor),
                    self.nc,
                )

                # 显式传入类别数，避免混淆检测与分割通道。
                detections = non_max_suppression(
                    predictions.contiguous(),
                    conf_thres=self.conf_threshold,
                    iou_thres=0.65,
                    nc=self.nc,
                    end2end=False,
                )[0]

        if self.renormalize:
            if len(detections) == 0:
                raise RuntimeError("没有检测框，无法执行框内归一化。")

            grayscale_cam = self.normalize_in_boxes(
                detections[:, :4].detach().cpu().numpy(),
                grayscale_cam,
            )

        cam_image = show_cam_on_image(
            image_float,
            grayscale_cam,
            use_rgb=True,
        )

        if self.show_box:
            for detection in detections.detach().cpu().numpy():
                x1, y1, x2, y2 = map(int, detection[:4])
                confidence = float(detection[4])
                class_id = int(detection[5])

                color = tuple(int(value) for value in self.colors[class_id])
                label = f"{self.model_names[class_id]} {confidence:.2f}"

                cv2.rectangle(
                    cam_image,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )
                cv2.putText(
                    cam_image,
                    label,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        Image.fromarray(cam_image).save(output_path)

    def __call__(self, img_path, save_path):
        img_path = os.path.abspath(img_path)
        save_path = os.path.abspath(save_path)

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }

        if os.path.isdir(img_path):
            image_files = [
                os.path.join(img_path, filename)
                for filename in sorted(os.listdir(img_path))
                if (
                    os.path.isfile(os.path.join(img_path, filename))
                    and os.path.splitext(filename)[1].lower() in extensions
                )
            ]
        elif os.path.isfile(img_path):
            image_files = [img_path]
        else:
            raise FileNotFoundError(f"输入路径不存在：{img_path}")

        if not image_files:
            raise RuntimeError(f"未找到图片：{img_path}")

        print(f"找到 {len(image_files)} 张图片")

        # 只检查第一张图片，不进入批处理。
        if self.debug_only:
            self.debug(image_files[0])
            return

        if os.path.realpath(img_path) == os.path.realpath(save_path):
            raise ValueError("输入目录与输出目录不能相同。")

        os.makedirs(save_path, exist_ok=True)

        success_count = 0
        failures = []

        for input_file in tqdm(
            image_files,
            desc="生成热力图",
            unit="张",
        ):
            filename = os.path.basename(input_file)
            output_file = os.path.join(save_path, filename)

            try:
                if os.path.realpath(input_file) == os.path.realpath(output_file):
                    raise ValueError("输出路径与原图相同，拒绝覆盖。")

                self.process(input_file, output_file)
                success_count += 1

            except Exception as error:
                failures.append((filename, str(error)))
                tqdm.write(f"[失败] {filename}：{error}")

                # 第一次失败打印完整堆栈，方便定位。
                if len(failures) == 1:
                    traceback.print_exc()

            finally:
                self.model.zero_grad(set_to_none=True)

                if self.method is not None:
                    hooks = self.method.activations_and_grads
                    hooks.activations.clear()
                    hooks.gradients.clear()

        print(f"\n完成：成功 {success_count} 张，失败 {len(failures)} 张")
        print(f"输出目录：{save_path}")

    def close(self):
        if self.method is not None:
            self.method.activations_and_grads.release()


def get_params():
    return {
        "weight": (
            "/mnt/DATA/quabot/work/github/HSCD/model-weight/SCCNet_files-seg4-1-wvd11-1-omht-scb6/train/weights/best.pt"
        ),
        "device": "cuda:0",
        "method": "GradCAMPlusPlus",
        "layer": [10, 12, 14, 16, 18],
        "backward_type": "class",
        "conf_threshold": 0.5,
        "ratio": 0.02,
        "show_box": False,
        "renormalize": True,
        "img_size": 640,
        # True：打印第一张图片的模型输出结构，然后退出。
        # 完成分割头输出适配后，再改为 False 批量生成热力图。
        "debug_only": False,
    }


if __name__ == "__main__":
    input_path = "/mnt/DATA/quabot/work/HRSC/images/val"
    output_path = "heatmap-wolte"

    heatmap = YOLOHeatmap(**get_params())

    try:
        heatmap(input_path, output_path)
    finally:
        heatmap.close()
