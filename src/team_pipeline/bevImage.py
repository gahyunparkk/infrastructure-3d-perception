# bev_topdown.py
import argparse
import math
import os
import sys
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd


# 기본 캔버스 1920x1080
DEFAULT_W, DEFAULT_H = 1920, 1080
MARGIN = 30                 # 가장자리 여백(px)
BOX_L, BOX_W = 4.5, 1.8     # 차량 바닥 크기 (월드 단위, 임의 값)  ex) 4.5m x 1.8m
ARROW_LEN = 3.0             # 진행방향 화살표 길이 (월드 단위)


# --- CSV 로딩(인코딩/헤더 유연) -------------------------------------------------
def load_obstacles_csv(path: str) -> pd.DataFrame:
    tried: List[str] = []
    df = None
    for enc in ("utf-8-sig", "utf-16", "cp949", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            tried.append(enc)
    if df is None:
        raise RuntimeError(f"CSV 디코딩 실패 (시도: {tried})")

    # 컬럼 정규화
    df.columns = [c.replace("\ufeff", "").strip().lower() for c in df.columns]
    need = ["cx", "cy", "yaw"]
    if not all(c in df.columns for c in need):
        # 헤더가 없거나 깨졌다면 names로 재시도
        df = pd.read_csv(path, encoding=enc, names=["ts", "cx", "cy", "yaw"], header=None)
        df.columns = ["ts", "cx", "cy", "yaw"]

    # 숫자화/결측 제거
    for c in ["cx", "cy", "yaw"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["cx", "cy", "yaw"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("CSV가 비어있거나 (cx,cy,yaw) 파싱 실패")
    return df


# --- 월드 → 캔버스 선형 매핑(균등 스케일 + Y축 반전) -----------------------------
def make_world_to_canvas(df: pd.DataFrame, canvas_w=DEFAULT_W, canvas_h=DEFAULT_H, margin=MARGIN):
    """
    월드 평면 좌표(x,y)를 2D 이미지 좌표(u,v)로 매핑하는 함수를 만든다.
    - 균등 스케일(가로세로 같은 배율)로 컨텐츠가 캔버스에 맞게 들어오도록
    - 영상 좌표계에 맞추기 위해 Y축 반전 (윗쪽이 0)
    """
    min_x, max_x = df["cx"].min(), df["cx"].max()
    min_y, max_y = df["cy"].min(), df["cy"].max()
    # 데이터가 점 몇 개뿐이어도 0 나눗셈 방지
    rx = max(max_x - min_x, 1e-6)
    ry = max(max_y - min_y, 1e-6)

    sx = (canvas_w - 2 * margin) / rx
    sy = (canvas_h - 2 * margin) / ry
    s = min(sx, sy)  # 균등 스케일

    def f_world_to_canvas(xy: np.ndarray) -> np.ndarray:
        """xy: (N,2) 월드 좌표 → (N,2) 캔버스 px"""
        xy = np.atleast_2d(xy).astype(np.float32)
        # 이동(좌측하단 기준) + 스케일
        u = (xy[:, 0] - min_x) * s + margin
        v = (xy[:, 1] - min_y) * s + margin
        # 영상 좌표계: y는 아래로 증가 → 반전
        v = canvas_h - v
        return np.stack([u, v], axis=1)

    return f_world_to_canvas


# --- 회전 박스(월드) → 캔버스 그리기 --------------------------------------------
def rect_corners_world(center_w: np.ndarray, yaw_rad: float, L: float, W: float) -> np.ndarray:
    """월드 좌표계에서 바닥 직사각형 4점(시계방향)을 생성"""
    hl, hw = L / 2.0, W / 2.0
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    local = np.array([[ hl,  hw],
                      [ hl, -hw],
                      [-hl, -hw],
                      [-hl,  hw]], dtype=np.float32)
    return local @ R.T + center_w  # (4,2)

def draw_vehicle(canvas: np.ndarray,
                 center_w: np.ndarray,
                 yaw_rad: float,
                 world_to_canvas,
                 L=BOX_L, W=BOX_W, arrow_len=ARROW_LEN):
    # 1) 월드에서 코너 계산
    poly_w = rect_corners_world(center_w, yaw_rad, L, W)      # (4,2)
    # 2) 월드→캔버스 매핑 (Y반전 포함)
    poly_px = world_to_canvas(poly_w).astype(int)
    cen_px  = world_to_canvas(center_w.reshape(1, 2)).astype(int)[0]
    head_w  = center_w + np.array([arrow_len * math.cos(yaw_rad),
                                   arrow_len * math.sin(yaw_rad)], dtype=np.float32)
    head_px = world_to_canvas(head_w.reshape(1, 2)).astype(int)[0]

    # 3) 그리기
    cv2.polylines(canvas, [poly_px], True, (0, 255, 0), 2)
    cv2.arrowedLine(canvas, tuple(cen_px), tuple(head_px), (0, 80, 255), 2, tipLength=0.3)
    cv2.circle(canvas, tuple(cen_px), 2, (255, 255, 255), -1)


# --- 메인 -----------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Blank 1920x1080 캔버스에 BEV(월드평면) 박스 그리기")
    ap.add_argument("--csv", required=True, help="obstacles.csv 경로 (ts,cx,cy,yaw)")
    ap.add_argument("--size", type=int, nargs=2, default=(DEFAULT_W, DEFAULT_H), metavar=("W", "H"),
                    help="출력 캔버스 크기 (기본 1920 1080)")
    ap.add_argument("--deg", action="store_true", help="yaw가 degree 단위이면 지정(기본 rad)")
    ap.add_argument("--dims", type=float, nargs=2, default=(BOX_L, BOX_W), metavar=("L", "W"),
                    help="차량 L,W (월드 단위). H는 바닥이므로 무시")
    ap.add_argument("--arrow", type=float, default=ARROW_LEN, help="화살표 길이(월드 단위)")
    ap.add_argument("--out", type=str, default="bev_topdown.png", help="출력 파일명")
    return ap.parse_args()


def main():
    args = parse_args()
    W, H = args.size
    df = load_obstacles_csv(args.csv)

    world_to_canvas = make_world_to_canvas(df, canvas_w=W, canvas_h=H, margin=MARGIN)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for _, r in df.iterrows():
        yaw = math.radians(float(r.yaw)) if args.deg else float(r.yaw)
        center_w = np.array([float(r.cx), float(r.cy)], dtype=np.float32)
        draw_vehicle(canvas, center_w, yaw, world_to_canvas,
                     L=args.dims[0], W=args.dims[1], arrow_len=args.arrow)

    cv2.imwrite(args.out, canvas)
    print(f"[OK] 저장: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
