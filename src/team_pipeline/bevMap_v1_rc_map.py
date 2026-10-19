# # bevMap.py  — draw boxes on top of a background image (e.g., samples/map.png)
# import os
# from typing import Iterable, Dict, Tuple, Optional, Union
# import numpy as np
# import cv2


# class BEVMap:
#     """
#     월드좌표[m] (x_min,y_min) ~ (x_max,y_max)를
#     BEV 이미지 [px] (W,H)로 선형 매핑해서 차량 폴리곤(회전 사각형)을 그리고,
#     차량 id는 '텍스트만' 표시합니다. yaw는 라디안, 중심(cx,cy)는 [m].
#     """

#     def __init__(
#         self,
#         world_min: Tuple[float, float] = (-10.0, -10.0),
#         world_max: Tuple[float, float] = (10.0, 10.0),
#         size_px: Optional[Tuple[int, int]] = (800, 800),   # (W, H). 배경과 다르면 배경을 리사이즈
#         flip_y: bool = False,
#         car_size_m: Tuple[float, float] = (3.2, 1.2),      # (length, width) in meters
#         out_dir: Optional[str] = "results/rc_bev_withSORT_map",
#         bg_image_path: Optional[str] = "samples/map.png"    # ← 배경 이미지 경로(None이면 흰 배경)
#     ):
#         # ----- 좌표 범위/설정 -----
#         self.xmin, self.ymin = world_min
#         self.xmax, self.ymax = world_max
#         self.flip_y = flip_y
#         self.car_L, self.car_W = car_size_m
#         self.bg_color = (255, 255, 255)

#         # ----- 배경 이미지 로드 -----
#         self.bg_img = None
#         if bg_image_path:
#             img = cv2.imread(bg_image_path, cv2.IMREAD_COLOR)
#             if img is None:
#                 raise FileNotFoundError(f"Background image not found: {bg_image_path}")
#             self.bg_img = img

#         # ----- 캔버스 크기 결정 -----
#         if self.bg_img is not None and size_px is None:
#             self.W, self.H = self.bg_img.shape[1], self.bg_img.shape[0]
#         elif self.bg_img is not None and size_px is not None:
#             self.W, self.H = size_px
#             if (self.bg_img.shape[1], self.bg_img.shape[0]) != (self.W, self.H):
#                 self.bg_img = cv2.resize(self.bg_img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
#         else:
#             assert size_px is not None, "size_px must be provided when bg_image_path=None"
#             self.W, self.H = size_px

#         # ----- 월드→픽셀 스케일 -----
#         self.sx = self.W / (self.xmax - self.xmin)
#         self.sy = self.H / (self.ymax - self.ymin)

#         # ----- 출력 폴더 -----
#         self.out_dir = out_dir
#         if out_dir:
#             os.makedirs(out_dir, exist_ok=True)

#     # --- 좌표 변환: world[m] -> pixel[px] ---
#     def world_to_px(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> np.ndarray:
#         u = (np.asarray(x) - self.xmin) * self.sx
#         v = (np.asarray(y) - self.ymin) * self.sy
#         if self.flip_y:
#             v = self.H - v
#         return np.stack([u, v], axis=-1)

#     # --- 차량 폴리곤(회전 사각형) 생성 ---
#     def car_corners_world(self, cx: float, cy: float, yaw: float) -> np.ndarray:
#         # 중심 기준: +x 전방, +y 좌측
#         L2, W2 = self.car_L / 2.0, self.car_W / 2.0
#         box_local = np.array([
#             [ +L2, +W2],  # front-left
#             [ +L2, -W2],  # front-right
#             [ -L2, -W2],  # back-right
#             [ -L2, +W2],  # back-left
#         ], dtype=float)
#         c, s = np.cos(yaw), np.sin(yaw)
#         R = np.array([[c, -s], [s, c]], dtype=float)
#         return (box_local @ R.T) + np.array([cx, cy])  # (4,2)

#     def _get_canvas(self) -> np.ndarray:
#         """배경 이미지를 캔버스로 사용(사본). 없으면 흰 배경 생성."""
#         if self.bg_img is None:
#             return np.full((self.H, self.W, 3), self.bg_color, dtype=np.uint8)
#         return self.bg_img.copy()

#     def draw_frame(self, tracks: Iterable[Dict], frame_id: Optional[int] = None) -> np.ndarray:
#         """
#         tracks: [{'id': int, 'cx': float, 'cy': float, 'yaw': float, 
#                   'vx': float, 'vy': float, ...}, ...]
#         """
#         canvas = self._get_canvas()

#         for t in tracks:
#             if t.get("cx") is None or t.get("cy") is None:
#                 continue
#             cx, cy = float(t["cx"]), float(t["cy"])
#             yaw = float(t.get("yaw", 0.0))

#             # 1) 월드 상 꼭짓점 계산 → 2) 픽셀로 변환
#             corners_w = self.car_corners_world(cx, cy, yaw)
#             corners_p = self.world_to_px(corners_w[:, 0], corners_w[:, 1]).astype(int)
#             center_p  = self.world_to_px(cx, cy).astype(int).ravel()

#             # 차량 폴리곤
#             cv2.polylines(canvas, [corners_p], isClosed=True, color=(0, 200, 0), thickness=2)

#             # --- ID와 속도 텍스트 생성 (변경/추가된 부분) ---
#             car_id = t.get("id", "")
            
#             # 1. vx, vy 값 가져오기
#             vx = t.get("vx", 0.0)
#             vy = t.get("vy", 0.0)

#             # 2. m/s 단위의 속력 계산 후 km/h로 변환
#             speed_ms = np.hypot(float(vx), float(vy))  # 스칼라 속력 (m/s)
#             speed_kmh = speed_ms * 3.6                # km/h 로 변환

#             # 3. ID와 속력을 함께 표시할 레이블 생성
#             # 예: "3 | 45.1kmh"
#             label = f"{car_id} | {speed_kmh:.1f}kmh"
            
#             # --- 기존 텍스트 그리기 로직 (그대로 사용) ---
#             if car_id: # ID가 있을 때만 텍스트 표시
#                 # 텍스트 크기를 계산하여 위치를 잡는 것은 기존 로직과 동일합니다.
#                 # 폰트 크기나 두께는 여기서 조절할 수 있습니다.
#                 font_scale = 0.5
#                 thickness = 1
#                 (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
#                 x, y, w, h = cv2.boundingRect(corners_p)
                
#                 fits_inside = (tw + 4 <= w) and (th + 4 <= h)
#                 if fits_inside:
#                     org = (int(center_p[0] - tw // 2), int(center_p[1] + th // 2))
#                 else:
#                     top_y = int(np.min(corners_p[:, 1]))
#                     org = (x + (w - tw) // 2, max(0, top_y - 6))
                
#                 cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
#                             (0, 0, 255), thickness, cv2.LINE_AA)

#         # 저장
#         if self.out_dir is not None and frame_id is not None:
#             cv2.imwrite(os.path.join(self.out_dir, f"{frame_id:06d}.png"), canvas)

#         return canvas

# if __name__ == "__main__":

#     # quick test on samples/map.png
#     bev = BEVMap(
#         world_min=(-10.0, -10.0),
#         world_max=( 10.0,  10.0),
#         size_px=None,
#         flip_y=False,
#         car_size_m=(3.2, 1.3),
#         out_dir=None,
#         bg_image_path="samples/map.png"
#     )

#     rng = np.random.default_rng(0)
#     tracks = [{
#         "id": i + 1,
#         "cx": float(rng.uniform(-9, 9)),
#         "cy": float(rng.uniform(-9, 9)),
#         "yaw": float(rng.uniform(-np.pi, np.pi)),
#         "vx": float(rng.uniform(-5, 5)),  # 테스트용 vx 추가 (m/s)
#         "vy": float(rng.uniform(-5, 5)),  # 테스트용 vy 추가 (m/s)
#     } for i in range(5)]

#     img = bev.draw_frame(tracks, frame_id=0)
#     cv2.imshow("BEV on map.png with Speed", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# bevMap.py — draw car images on top of a background image
# bevMap.py — draw car images on top of a background image

# bevMap.py — draw car images on top of a background image
# bevMap.py — draw car images on top of a background image




import os
from typing import Iterable, Dict, Tuple, Optional, Union
import numpy as np
import cv2


class BEVMap:
    """
    월드좌표[m]를 BEV 이미지[px]로 매핑하여, 실제 차량 이미지를 회전시켜 그립니다.
    [수정] yaw 값의 회전 방향을 반대로 적용하여 차량의 앞/뒤 방향이 정확히 표시되도록 수정했습니다.
    """

    def __init__(
        self,
        world_min: Tuple[float, float] = (-10.0, -10.0),
        world_max: Tuple[float, float] = (10.0, 10.0),
        size_px: Optional[Tuple[int, int]] = (800, 800),
        flip_y: bool = False,
        car_size_m: Tuple[float, float] = (3.2, 1.2),    # (length, width) in meters
        out_dir: Optional[str] = None,
        bg_image_path: Optional[str] = "samples/map.png",
        # vvvvvvvvvvvvvv ADDED vvvvvvvvvvvvvv
        car_image_path: Optional[str] = None  #車両画像のパスを追加
        # ^^^^^^^^^^^^^^ ADDED ^^^^^^^^^^^^^^
    ):
        # ----- 좌표계 및 크기 설정 -----
        self.xmin, self.ymin = world_min
        self.xmax, self.ymax = world_max
        self.flip_y = flip_y
        self.car_L, self.car_W = car_size_m
        self.bg_color = (255, 255, 255)
        self.out_dir = out_dir

        # ----- 배경 이미지 로드 -----
        # vvvvvvvvvvvvvv CHANGED vvvvvvvvvvvvvv
        # The original code for loading the background image was simpler.
        # This new version is more robust, handling potential file path issues
        # and decoding errors by reading the file as bytes first.
        self.bg_img = None
        if bg_image_path and os.path.exists(bg_image_path):
            try:
                with open(bg_image_path, "rb") as f:
                    file_bytes = np.fromfile(f, np.uint8)
                self.bg_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception:
                self.bg_img = None
        # ^^^^^^^^^^^^^^ CHANGED ^^^^^^^^^^^^^^

        # ----- 캔버스 크기 결정 -----
        # vvvvvvvvvvvvvv CHANGED vvvvvvvvvvvvvv
        # This block was restructured for clarity. It now handles four cases:
        # 1. Background image exists, size_px is not given.
        # 2. size_px is given (overrides background image size).
        # 3. Neither is available (raises an error).
        # 4. If no background image, it creates a new white canvas.
        if self.bg_img is not None and size_px is None:
            self.H, self.W = self.bg_img.shape[:2]
        elif size_px is not None:
            self.W, self.H = size_px
        else:
            raise ValueError("size_px must be provided if bg_image is not available.")
        
        if self.bg_img is None:
            self.bg_img = np.full((self.H, self.W, 3), self.bg_color, dtype=np.uint8)
        elif (self.bg_img.shape[1], self.bg_img.shape[0]) != (self.W, self.H):
            self.bg_img = cv2.resize(self.bg_img, (self.W, self.H))
        # ^^^^^^^^^^^^^^ CHANGED ^^^^^^^^^^^^^^
            
        self.sx = self.W / (self.xmax - self.xmin)
        self.sy = self.H / (self.ymax - self.ymin)

        # ----- 차량 이미지 로드 (안정적인 버전) -----
        # vvvvvvvvvvvvvv ADDED vvvvvvvvvvvvvv
        # This entire block is new. It loads the car image, ensures it has an
        # alpha channel (for transparency), and prints a warning if it fails.
        self.car_img = None
        if car_image_path and os.path.exists(car_image_path):
            try:
                with open(car_image_path, "rb") as f:
                    file_bytes = np.fromfile(f, np.uint8)
                self.car_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

                if self.car_img is not None and self.car_img.shape[2] != 4:
                    self.car_img = cv2.cvtColor(self.car_img, cv2.COLOR_BGR2BGRA)
            except Exception as e:
                self.car_img = None
        
        if self.car_img is None and car_image_path is not None:
             print(f"---[경고]--- 차량 이미지를 로드하지 못했습니다: '{car_image_path}'")
             print("초록색 네모만 그립니다.")
        # ^^^^^^^^^^^^^^ ADDED ^^^^^^^^^^^^^^

        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)

    def world_to_px(self, x, y):
        u = (np.asarray(x) - self.xmin) * self.sx
        v = (np.asarray(y) - self.ymin) * self.sy
        if self.flip_y:
            v = self.H - v
        return np.stack([u, v], axis=-1)

    def car_corners_world(self, cx: float, cy: float, yaw: float) -> np.ndarray:
        # vvvvvvvvvvvvvv CHANGED vvvvvvvvvvvvvv
        # The order of corner points was changed for consistency,
        # although the final shape is the same.
        L2, W2 = self.car_L / 2.0, self.car_W / 2.0
        corners = np.array([[-L2, W2], [L2, W2], [L2, -W2], [-L2, -W2]])
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        return (corners @ R.T) + np.array([cx, cy])
        # ^^^^^^^^^^^^^^ CHANGED ^^^^^^^^^^^^^^

    # vvvvvvvvvvvvvv ADDED vvvvvvvvvvvvvv
    # This new helper function overlays a transparent (RGBA) foreground image
    # onto a background image at a specified center position. It handles
    # boundary checks to prevent errors if the image goes off-screen.
    def _overlay_image(self, bg, fg_rgba, pos):
        """배경(bg) 위에 전경(fg_rgba) 이미지를 중심점(pos)에 합성합니다."""
        x_center, y_center = pos
        h, w = fg_rgba.shape[:2]
        
        y1 = y_center - h // 2
        x1 = x_center - w // 2

        bg_h, bg_w = bg.shape[:2]
        overlap_y1 = max(y1, 0)
        overlap_x1 = max(x1, 0)
        overlap_y2 = min(y1 + h, bg_h)
        overlap_x2 = min(x1 + w, bg_w)

        if overlap_y1 >= overlap_y2 or overlap_x1 >= overlap_x2:
            return

        fg_y1 = overlap_y1 - y1
        fg_x1 = overlap_x1 - x1
        fg_y2 = fg_y1 + (overlap_y2 - overlap_y1)
        fg_x2 = fg_x1 + (overlap_x2 - overlap_x1)

        bg_roi = bg[overlap_y1:overlap_y2, overlap_x1:overlap_x2]
        fg_roi = fg_rgba[fg_y1:fg_y2, fg_x1:fg_x2]

        alpha = fg_roi[:, :, 3:4] / 255.0
        bg_roi[:] = fg_roi[:, :, :3] * alpha + bg_roi * (1. - alpha)
    # ^^^^^^^^^^^^^^ ADDED ^^^^^^^^^^^^^^

    def draw_frame(self, tracks: Iterable[Dict], frame_id: Optional[int] = None) -> np.ndarray:
        canvas = self.bg_img.copy()

        for t in tracks:
            car_id = t.get("id", "")
            cx, cy, yaw = float(t["cx"]), float(t["cy"]), float(t.get("yaw", 0.0))
            center_p = self.world_to_px(cx, cy).astype(int).ravel()

            # vvvvvvvvvvvvvv CHANGED vvvvvvvvvvvvvv
            # The original code only drew a green polygon. This was kept, but
            # now a much more complex block is added to draw a rotated car image.
            corners_w = self.car_corners_world(cx, cy, yaw)
            corners_p = self.world_to_px(corners_w[:, 0], corners_w[:, 1]).astype(int)
            # The color was changed from (0, 200, 0) to (0, 255, 0) for a brighter green.
            cv2.polylines(canvas, [corners_p], isClosed=True, color=(0, 255, 0), thickness=2)

            # vvvvvvvvvvvvvv ADDED vvvvvvvvvvvvvv
            # This entire block is new. It handles resizing the car image to fit the
            # bounding box, rotating it according to the vehicle's yaw, and
            # then calling _overlay_image to draw it on the canvas.
            if self.car_img is not None:
                px_w = int(self.car_W * self.sx)
                px_h = int(self.car_L * self.sy)
                if px_w < 1 or px_h < 1: continue

                img_h_orig, img_w_orig = self.car_img.shape[:2]
                img_ratio = img_w_orig / img_h_orig
                box_ratio = px_w / px_h
                
                if box_ratio > img_ratio:
                    final_h = px_h
                    final_w = int(final_h * img_ratio)
                else:
                    final_w = px_w
                    final_h = int(final_w / img_ratio)

                resized = cv2.resize(self.car_img, (final_w, final_h), interpolation=cv2.INTER_AREA)
                
                # <<<---!!! KEY CHANGE: Rotation direction is inverted !!!--->>>
                # The original (commented out) calculation did not correctly orient the car image.
                # By negating the angle, the image now correctly points in the direction of travel.
                # original: angle_deg = np.degrees(yaw) - 90
                angle_deg = -np.degrees(yaw) - 90
                
                rot_center = (final_w / 2, final_h / 2)
                M = cv2.getRotationMatrix2D(rot_center, angle_deg, 1.0)
                
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int((final_h * sin) + (final_w * cos))
                new_h = int((final_h * cos) + (final_w * sin))
                
                M[0, 2] += (new_w / 2) - rot_center[0]
                M[1, 2] += (new_h / 2) - rot_center[1]

                rotated = cv2.warpAffine(resized, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))
                
                self._overlay_image(canvas, rotated, (center_p[0], center_p[1]))
            # ^^^^^^^^^^^^^^ ADDED ^^^^^^^^^^^^^^

            # vvvvvvvvvvvvvv CHANGED vvvvvvvvvvvvvv
            # The text drawing logic has been completely changed.
            # Instead of trying to fit the text inside or place it above the box,
            # it now calculates a simple offset from the car's center point.
            # This provides a more consistent and cleaner look.
            if car_id:
                vx, vy = t.get("vx", 0.0), t.get("vy", 0.0)
                speed_kmh = np.hypot(float(vx), float(vy)) * 3.6
                
                # 1. 두 개의 레이블 문자열을 정의합니다.
                label1 = f"ID: {car_id}"
                label2 = f"{speed_kmh:.1f}kmh"

                # 2. 폰트 속성을 설정합니다 (ID는 크게, 속도는 작게).
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale_id = 1.0     # ID 폰트 크기
                font_scale_speed = 0.6  # 속도 폰트 크기
                thickness = 2
                padding = 5             # 레이블 사이의 수직 간격

                # 3. 각 레이블의 텍스트 크기(너비, 높이)를 계산합니다.
                (tw1, th1), _ = cv2.getTextSize(label1, font_face, font_scale_id, thickness)
                (tw2, th2), _ = cv2.getTextSize(label2, font_face, font_scale_speed, thickness)

                # 4. 첫 번째 레이블(ID)의 위치를 계산합니다.
                # 차량의 아래쪽에 위치하도록 y 오프셋을 설정합니다.
                base_y_offset = int((self.car_L * self.sy / 2) * 1.1)
                
                # org1은 첫 번째 텍스트의 왼쪽-아래 기준점입니다.
                org1_x = center_p[0] - tw1 // 2
                org1_y = center_p[1] + base_y_offset + th1
                
                # 5. 두 번째 레이블(속도)의 위치를 계산합니다.
                # 첫 번째 레이블 바로 아래에 위치시킵니다.
                org2_x = center_p[0] - tw2 // 2
                org2_y = org1_y + th1 + padding # 첫 번째 레이블 높이(th1)와 패딩만큼 아래로 이동합니다.

                # 6. 두 레이블을 캔버스에 그립니다.
                cv2.putText(canvas, label1, (org1_x, org1_y), font_face, font_scale_id, (0, 255, 255), thickness, cv2.LINE_AA)
                cv2.putText(canvas, label2, (org2_x, org2_y), font_face, font_scale_speed, (0, 255, 255), thickness, cv2.LINE_AA)

            # ^^^^^^^^^^^^^^ CHANGED ^^^^^^^^^^^^^^

        if self.out_dir and frame_id is not None:
            cv2.imwrite(os.path.join(self.out_dir, f"{frame_id:06d}.png"), canvas)

        return canvas








