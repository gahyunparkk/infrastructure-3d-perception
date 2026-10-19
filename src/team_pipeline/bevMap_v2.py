# bevMap_v2.py (SORT 적용하지 않을 경우)
import os
from typing import Iterable, Dict, Tuple, Optional, Union
import numpy as np
import cv2

class BEVMap:
    """
    월드좌표[m] (x_min,y_min) ~ (x_max,y_max)를
    BEV 이미지 [px] (W,H)로 선형 매핑해서 차량 박스를 그립니다.
    yaw는 라디안, 중심(cx,cy)는 [m].
    """
    def __init__(
        self,
        world_min: Tuple[float, float] = (-100.0, -50.0),
        world_max: Tuple[float, float] = (-30.0, 40.0),
        size_px: Tuple[int, int] = (700, 900),            # (W, H)
        flip_y: bool = False,                             # 이미지 y축(아래 +) 보정
        car_size_m: Tuple[float, float] = (4.5, 1.8),     # (length, width)
        out_dir: Optional[str] = None                     # 저장 폴더 (None이면 저장 안함)
    ):
        self.xmin, self.ymin = world_min
        self.xmax, self.ymax = world_max
        self.W, self.H = size_px
        self.flip_y = flip_y
        self.car_L, self.car_W = car_size_m
        self.bg_color = (255, 255, 255)

        self.sx = self.W / (self.xmax - self.xmin)
        self.sy = self.H / (self.ymax - self.ymin)
        self.out_dir = out_dir
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # --- 좌표 변환: world[m] -> pixel[px] ---
    def world_to_px(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> np.ndarray:
        u = (np.asarray(x) - self.xmin) * self.sx
        v = (np.asarray(y) - self.ymin) * self.sy
        if self.flip_y:
            v = self.H - v
        return np.stack([u, v], axis=-1)

    # --- 차량 폴리곤(회전 사각형) 생성 ---
    def car_corners_world(self, cx: float, cy: float, yaw: float) -> np.ndarray:
        # 중심 기준: +x 전방, +y 좌측
        L2, W2 = self.car_L / 2.0, self.car_W / 2.0
        box_local = np.array([
            [ +L2, +W2],  # front-left
            [ +L2, -W2],  # front-right
            [ -L2, -W2],  # back-right
            [ -L2, +W2],  # back-left
        ], dtype=float)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=float)
        box_world = (box_local @ R.T) + np.array([cx, cy])
        return box_world  # (4,2)

    def draw_frame(self, tracks: Iterable[Dict], frame_id: Optional[int] = None) -> np.ndarray:
        """
        tracks: [{'id': int, 'cx': float, 'cy': float, 'yaw': float, ...}, ...]
                (SORT confirmed만 넣어주면 깔끔)
        """
        canvas = np.full((self.H, self.W, 3), self.bg_color, dtype=np.uint8)

        for t in tracks:
            if t.get("cx") is None or t.get("cy") is None:
                continue
            cx, cy = float(t["cx"]), float(t["cy"])
            yaw = float(t["yaw"]) if t.get("yaw") is not None else 0.0

            # 1) 월드 상 꼭짓점 계산 → 2) 픽셀로 변환
            corners_w = self.car_corners_world(cx, cy, yaw)     # (4,2) [m]
            corners_p = self.world_to_px(corners_w[:, 0], corners_w[:, 1]).astype(int)

            # 중심도 픽셀 변환
            center_p = self.world_to_px(cx, cy).astype(int)

            # 초록색 박스
            cv2.polylines(canvas, [corners_p], isClosed=True, color=(0, 200, 0), thickness=2)

            # 빨간색 id 라벨 (박스 위쪽에 살짝)
            label = str(t.get("id", ""))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # flip_y에 따라 라벨 위치 보정 방향 반전
            offset_px = int(self.car_W * self.sy * (1.2 if not self.flip_y else -1.2))
            p = (int(center_p[0] - tw // 2), int(center_p[1] - offset_px))
            #cv2.rectangle(canvas, (p[0]-4, p[1]-th-4), (p[0]+tw+4, p[1]+4), (0,255,0), 2)
            #cv2.putText(canvas, label, (p[0], p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

        # 테두리
        cv2.rectangle(canvas, (0,0), (self.W-1, self.H-1), (50,50,50), 2)

        # 저장 옵션
        if self.out_dir is not None and frame_id is not None:
            cv2.imwrite(os.path.join(self.out_dir, f"{frame_id:06d}.png"), canvas)

        return canvas
