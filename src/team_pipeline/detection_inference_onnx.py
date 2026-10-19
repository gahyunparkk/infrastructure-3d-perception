# detection_inference_onnx.py  (CPU 전용)

import argparse
import glob
import os.path as osp
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
import torchvision
import numpy as np
import cv2
import onnx
import onnxruntime as ort

# Constants
DEFAULT_LETTERBOX_SIZE = (832, 1440)

# Runtime constants (CPU 고정)
device = torch.device('cpu')


class LetterBox:
    def __init__(
        self,
        shape: Tuple[int, int] = DEFAULT_LETTERBOX_SIZE,
        auto: bool = False,
        scale_fill: bool = False,
        scale_up: bool = True,
        center: bool = True,
        stride: int = 32,
    ):
        self.new_shape = shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scale_up
        self.stride = stride
        self.center = center

    def __call__(self, image: np.ndarray, labels: Dict[str, Any] = None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scale_fill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])

        if self.center:
            dw /= 2
            dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = (int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1)))
        left, right = (int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1)))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return img


class Inferencer:
    def __init__(
        self,
        model_path: str,
        letterbox_size: Tuple[int, int] = DEFAULT_LETTERBOX_SIZE,
        is_half: bool = True,
    ):
        # Load model & session (CPU)
        self.model = onnx.load(model_path)
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.is_half = is_half
        self.letterbox_size = letterbox_size

    def run(self, image: np.ndarray) -> torch.Tensor:
        """
        Runs the inference to predict list of object boxes (parallelograms).

        Parameters:
            image (numpy.ndarray): BGR image to run inference on

        Returns:
            results (torch.Tensor): (n, 6) -> (cx, cy, v1x, v1y, v2x, v2y)
        """
        original_width, original_height = (image.shape[1], image.shape[0])

        # 전처리: torch -> numpy(FP16/32) 변환 후 CPU 실행
        x: torch.Tensor = self._preprocess(image)
        if self.is_half:
            x = x.half()

        x_np = x.cpu().numpy().astype(np.float16 if self.is_half else np.float32)
        out = self.sess.run(None, {"x": x_np})  # CPU forward
        y = torch.from_numpy(out[0])  # (1, 7, 24570) tensor (CPU)

        y = self._non_max_suppression(y)[0]
        results = self._scale_triangles(
            y[:, :6], self.letterbox_size, (original_height, original_width, 3)
        )
        return results

    def _nms_bboxes(self, triangles: torch.Tensor, confidences: torch.Tensor, iou_threshold):
        bboxes = self._compute_bounding_boxes(triangles)
        return torchvision.ops.nms(bboxes, confidences, iou_threshold)

    def _compute_bounding_boxes(self, triangles: torch.Tensor):
        """
        Compute axis-aligned bboxes for the parallelograms (mirror v1,v2 across center).
        """
        x1, y1 = triangles[:, 0], triangles[:, 1]  # center
        x2, y2 = triangles[:, 2], triangles[:, 3]  # v1
        x3, y3 = triangles[:, 4], triangles[:, 5]  # v2

        # mirrored points
        x2_m, y2_m = 2 * x1 - x2, 2 * y1 - y2
        x3_m, y3_m = 2 * x1 - x3, 2 * y1 - y3

        x_coords = torch.stack([x1, x2, x3, x2_m, x3_m], dim=1)
        y_coords = torch.stack([y1, y2, y3, y2_m, y3_m], dim=1)

        min_x, _ = torch.min(x_coords, dim=1)
        max_x, _ = torch.max(x_coords, dim=1)
        min_y, _ = torch.min(y_coords, dim=1)
        max_y, _ = torch.max(y_coords, dim=1)
        return torch.stack([min_x, min_y, max_x, max_y], dim=1)

    def _non_max_suppression(
        self,
        prediction: torch.Tensor,
        conf_thres: float = 0.1,
        iou_thres: float = 0.5,
        classes: torch.Tensor = None,
        agnostic: bool = False,
        multi_label: bool = False,
        max_det: int = 300,
        nc: int = 0,
        max_nms: int = 30000,
        max_wh: int = 7680,
    ):
        assert 0 <= conf_thres <= 1
        assert 0 <= iou_thres <= 1
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        if classes is not None:
            classes = torch.tensor(classes, device=prediction.device)

        bs = prediction.shape[0]
        nc = nc or (prediction.shape[1] - 6)
        nm = prediction.shape[1] - nc - 6
        mi = 6 + nc
        xc = prediction[:, 6:mi].amax(1) > conf_thres

        prediction = prediction.transpose(-1, -2)
        output = [torch.zeros((0, 8 + nm), device=prediction.device)] * bs

        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            box, cls, mask = x.split((6, nc, nm), 1)

            if multi_label:
                i, j = torch.where(cls > conf_thres)
                x = torch.cat((box[i], x[i, 6 + j, None], j[:, None].float(), mask[i]), 1)
            else:
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            if classes is not None:
                x = x[(x[:, 7:8] == classes).any(1)]

            n = x.shape[0]
            if n > 0:
                if n > max_nms:
                    x = x[x[:, 6].argsort(descending=True)[:max_nms]]

                c = x[:, 7:8] * (0 if agnostic else max_wh)
                scores = x[:, 6]
                boxes = x[:, :6] + c

                i = self._nms_bboxes(boxes, scores, iou_thres)
                i = i[:max_det]
                output[xi] = x[i]

        return output

    def _scale_triangles(
        self,
        triangles: torch.Tensor,
        src_shape: Tuple[float, float],
        target_shape: Union[Tuple[float, float], Tuple[float, float, float]],
        padding: bool = True,
    ):
        if len(target_shape) == 2:
            target_shape = (target_shape[0], target_shape[1], 3)

        gain = min(src_shape[0] / target_shape[0], src_shape[1] / target_shape[1])
        pad = (
            round((src_shape[1] - target_shape[1] * gain) / 2 - 0.1),
            round((src_shape[0] - target_shape[0] * gain) / 2 - 0.1),
        )

        if padding:
            triangles[..., 0] -= pad[0]
            triangles[..., 1] -= pad[1]
            triangles[..., 2] -= pad[0]
            triangles[..., 3] -= pad[1]
            triangles[..., 4] -= pad[0]
            triangles[..., 5] -= pad[1]

        triangles[..., :6] /= gain
        return triangles

    def _preprocess(self, im: Union[np.ndarray, torch.Tensor]):
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack([LetterBox(self.letterbox_size, auto=False, stride=32)(im)])
            # BGR->RGB, BHWC->BCHW
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)

        im = im.to(device)
        im = im.float()
        if not_tensor:
            im /= 255
        return im


def visualize_detections(
    detections: np.ndarray, image: np.ndarray, save_path: Optional[str] = None
) -> Image:
    for detection in detections:
        x1, y1, x2, y2, x3, y3 = detection[:6]
        x2_mirror = 2 * x1 - x2
        y2_mirror = 2 * y1 - y2
        x3_mirror = 2 * x1 - x3
        y3_mirror = 2 * y1 - y3

        poly = np.array([[x2, y2], [x3, y3], [x2_mirror, y2_mirror], [x3_mirror, y3_mirror]], dtype=int)
        cv2.polylines(image, [poly], isClosed=True, color=(0, 255, 0), thickness=2)

    if save_path is not None:
        cv2.imwrite(save_path, image)
    return Image.fromarray(image)

# [추가] 3D 시각화 --------------------------------------------------------
def visualize_detections_3d(
    detections: np.ndarray,
    image: np.ndarray,
    save_path: Optional[str] = None,
    dy: Optional[int] = 30, #None,      # None이면 자동(아래 height_scale로 결정)
    height_scale: float = 0.25,    # 자동 높이: poly 높이 * 0.25
    min_dy: int = 8, max_dy: int = 80,
    ) -> Image:
    """
    기존 visualize_detections와 동일한 입력/동작: 
    - detections: (N,6) = [cx,cy,f1x,f1y,f2x,f2y]
    - image: BGR
    - save_path: 저장 경로(선택)
    차이: 각 평행사변형 바닥에서 y-방향으로 '뚜껑'을 올린 pseudo 3D를 그립니다.
    """
    img = image.copy()

    # 1) (cx,cy,f1,f2) → 바닥 평행사변형 4점 복원
    polys = []
    for det in detections:
        cx, cy, x2, y2, x3, y3 = det[:6]
        x2m, y2m = 2*cx - x2, 2*cy - y2
        x3m, y3m = 2*cx - x3, 2*cy - y3
        poly = np.array([[x2, y2], [x3, y3], [x2m, y2m], [x3m, y3m]], dtype=np.float32)
        hull = cv2.convexHull(poly).squeeze().reshape(-1, 2)
        polys.append(hull)

    # 2) 아래쪽(큰 y)을 먼저 그려 겹침을 자연스럽게
    polys.sort(key=lambda p: p[:,1].max())

    for poly in polys:
        # 자동 높이(or 고정 dy)
        y_min, y_max = poly[:,1].min(), poly[:,1].max()
        h = max(1, int(y_max - y_min))
        off = dy if dy is not None else int(np.clip(height_scale * h, min_dy, max_dy))

        # 바닥/뚜껑 좌표
        base = poly.astype(int)
        top  = base.copy(); top[:,1] -= off

        # 면 채우기(옅게) + 외곽선 + 수직 모서리
        overlay = img.copy()
        cv2.fillPoly(overlay, [top],  (0, 200, 255))  # 윗면
        cv2.fillPoly(overlay, [base], (0, 170, 240))  # 바닥(살짝 칠해 대비)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        for i in range(len(base)):
            cv2.line(img, tuple(base[i]), tuple(top[i]), (0,255,255), 2)

        cv2.polylines(img, [base], True, (0,255,255), 2)
        cv2.polylines(img, [top],  True, (0,255,255), 2)

    if save_path is not None:
        cv2.imwrite(save_path, img)

    return Image.fromarray(img)
# ------------------------------------------------------- 

def results_to_df(results: torch.Tensor, ts: float = time.time()) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": ts,
            "cx": results[:, 0],
            "cy": results[:, 1],
            "f1x": results[:, 2],
            "f1y": results[:, 3],
            "f2x": results[:, 4],
            "f2y": results[:, 5],
        }
    )


def guess_timestamp(filename: str) -> float:
    if match := re.search(r"(\d{10}-\d+)", filename):
        return float(match.group(1).replace("-", "."))
    return 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for ONNX model (CPU)")
    parser.add_argument("image", type=str, help="Path to image or glob pattern")
    parser.add_argument("--model", type=str, default="checkpoints/model_v2_half.onnx")
    parser.add_argument("--ts", type=float, default=time.time(), help="Timestamp to use in CSV output")
    parser.add_argument("--guess-ts", action="store_false", help="Infer timestamp from filename")
    parser.add_argument("--half", action="store_false", help="Model is FP16 (float16). If 실패 시 이 옵션 빼세요.")
    parser.add_argument("--viz", action="store_true", help="Save visualization images")
    # [추가] 3D 시각화 옵션----------------------------------------------------------
    parser.add_argument("--viz-3d", action="store_true", help="Save pseudo-3D visualization (pixel-offset extrusions)")
    parser.add_argument("--viz-3d-dy", type=int, default=None, help="Fixed vertical offset in pixels for pseudo-3D. If None, auto-scale.")
    # -----------------------------------------------------------------------------

    return parser.parse_args()


def main():
    print("Initialized")
    args = parse_args()

    print("Preparing inferencer (CPU)")
    inferencer = Inferencer(args.model, is_half=args.half)

    dfs: List[pd.DataFrame] = []

    for filename in glob.glob(args.image):
        print(f"Loading image {filename}")
        image = cv2.imread(filename)

        ts = args.ts
        if args.guess_ts:
            ts = guess_timestamp(filename) or ts

        print("Running inference (CPU)")
        t0 = time.monotonic()
        results: np.ndarray = inferencer.run(image).cpu().numpy()
        t1 = time.monotonic()

        if args.viz:
            out_ext = filename.split(".")[-1]
            out_file = f"{osp.basename(filename).rstrip('.' + out_ext)}_out.{out_ext}"
            visualize_detections(results, image, save_path=out_file)

        # [추가] 3D 저장 ----------------------------------------
        if args.viz_3d:
            out_ext = filename.split(".")[-1]
            out_base = osp.basename(filename).rstrip("." + out_ext)

            # 저장 폴더 지정: result/RC
            out_dir = osp.join("result", "RC")
            os.makedirs(out_dir, exist_ok=True)

            # 저장 파일명: result/RC/frame_000123_out3d.jpg
            out_file3d = osp.join(out_dir, f"{out_base}_out3d.{out_ext}")

            visualize_detections_3d(
                results,
                image.copy(),
                save_path=out_file3d,
                dy=args.viz_3d_dy  # None이면 높이 자동
            )
        # ----------------------------------------------------


        print(f"Got {results.shape[0]} object boxes, took {t1 - t0:.3f} sec")
        dfs.append(results_to_df(results, ts))

    print("Saving output")
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("detections.csv", header=True, index=False)


if __name__ == "__main__":
    main()
