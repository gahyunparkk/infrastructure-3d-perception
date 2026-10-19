# video_obstacles.py — Video file → ONNX → obstacles (world coords + yaw)
import argparse
import os
import time
import socket
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

#from detection_inference_onnx import Inferencer, visualize_detections, results_to_df
from detection_inference_onnx import Inferencer, visualize_detections, visualize_detections_3d, results_to_df
from geometry import complete_parallelograms, get_heading
from projection import pixel_to_world_plane, rescale_pixels
from calibration import H
# [추가]
from sort_tracker_v2 import sort_update, reset_sort_state
from bevMap_v1 import BEVMap

# ==== UDP 설정 (PC2 수신 IP/포트로 바꿔주세요) ====
UDP_IP = "127.0.0.1"   # PC2 IP
UDP_PORT = 5009           # PC2에서 열 포트
_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 한 번에 보낼 최대 페이로드(UDP MTU 보호용, 넉넉히 1400B)
_MAX_UDP_PAYLOAD = 1400

def _udp_send_lines(lines):
    """
    lines: ["0 -73.68 -61.31 -7.78", "1 ...", ...] 형태 리스트
    MTU 넘치지 않게 적당히 묶어서 전송
    """
    if not lines: 
        return
    buf = ""
    for ln in lines:
        # 다음 줄을 더하면 MTU 넘는지 체크 (대략적으로 길이 기반)
        if len(buf) + len(ln) + 1 > _MAX_UDP_PAYLOAD:
            _udp_sock.sendto(buf.encode(), (UDP_IP, UDP_PORT))
            buf = ln
        else:
            if buf:
                buf += "\n"
            buf += ln
    if buf:
        _udp_sock.sendto(buf.encode(), (UDP_IP, UDP_PORT))


def _send_df_obs_udp(df_obs, frame_id):
    """
    df_obs: (ts, cx, cy, yaw) 컬럼 보유 DataFrame
    frame 내 로우 인덱스 i를 앞에 붙여서 "i cx cy yaw" 형식으로 전송
    소수점은 보기 좋게 2자리로 포맷
    """
    lines = []
    for i, row in enumerate(df_obs.itertuples(index=False)):
        # row 순서: ts, cx, cy, yaw (itertuples 순서는 df 생성 순서와 동일)
        ts, cx, cy, yaw = row
        # 필요하면 frame_id나 ts를 포함하도록 포맷 바꿔도 됨 (예: f"{frame_id} {i} {cx:.2f} {cy:.2f} {yaw:.2f}")
        lines.append(f"{i} {cx:.2f} {cy:.2f} {yaw:.2f}")
    _udp_send_lines(lines)

def _send_tracks_udp(tracks, frame_id=None, yaw_in_deg=True):
    """
    tracks: sort_update() 결과 리스트(confirmed만 넘겨주는 것을 권장)
      각 원소: {"id", "cx", "cy", "yaw", ...}
    포맷: "id cx cy yaw" (yaw는 deg로 전송; None이면 NaN)
    """
    if not tracks:
        return
    lines = []
    for t in tracks:
        cx = float(t["cx"])
        cy = float(t["cy"])
        yaw = t.get("yaw", None)
        if yaw is None or np.isnan(yaw):
            yaw_out = float("nan")
        else:
            yaw_out = math.degrees(float(yaw)) if yaw_in_deg else float(yaw)
        # 필요하면 frame_id까지 보내고 싶다면: f"{frame_id} {t['id']} {cx:.2f} {cy:.2f} {yaw_out:.2f}"
        lines.append(f"{int(t['id'])} {cx:.2f} {cy:.2f} {yaw_out:.2f}")
    _udp_send_lines(lines)

INPUT_IMAGE_SIZE = (1920,1080)

def detections_df_to_obstacles_df(df: pd.DataFrame, camera: str, homography_size=(1920,1080)) -> pd.DataFrame:
    rescale = (lambda x: x) if tuple(homography_size) == INPUT_IMAGE_SIZE \
             else (lambda x: rescale_pixels(x, src=INPUT_IMAGE_SIZE, target=tuple(homography_size)))
    front_1 = np.dstack([df['f1x'], df['f1y']]).reshape(-1, 2)
    front_2 = np.dstack([df['f2x'], df['f2y']]).reshape(-1, 2)
    centers = np.dstack([df['cx'], df['cy']]).reshape(-1, 2)
    H_cam = H[camera]
    f1p = pixel_to_world_plane(front_1, H_cam, rescale=rescale)
    f2p = pixel_to_world_plane(front_2, H_cam, rescale=rescale)
    cp  = pixel_to_world_plane(centers, H_cam, rescale=rescale)
    corners_p = complete_parallelograms(f1p, f2p, cp)
    headings_p = get_heading(corners_p[:, 0], corners_p[:, 2], False)
    return pd.DataFrame({'ts': df['ts'].values,
                         'cx': corners_p[:, -1, 0],
                         'cy': corners_p[:, -1, 1],
                         'yaw': headings_p})

def main():
    ap = argparse.ArgumentParser(description="Video file → ONNX → obstacles (world coords + yaw)")
    ap.add_argument("--video", required=True, help="Path to input video (mp4/avi)")
    ap.add_argument("--model", type=str, default="checkpoints/model_v2_half.onnx")
    ap.add_argument("--camera", choices=H.keys(), default=list(H.keys())[0])
    ap.add_argument("--homography_size", nargs=2, type=int, default=(1920,1080))
    ap.add_argument("--viz", action="store_true", help="Show visualization window")
    ap.add_argument("--save_viz", action="store_true", help="Save frames with overlay into ./image/")
    ap.add_argument("--save_csv", action="store_true", help="Append obstacles rows to CSV")
    ap.add_argument("--csv_path", type=str, default="obstacles_from_video.csv")
    ap.add_argument("--step", type=int, default=1, help="Process every Nth frame (for speed)")
    ap.add_argument("--viz-3d", action="store_true", help="Save pseudo-3D visualization (pixel-offset extrusions)")
    ap.add_argument("--viz-3d-dy", type=int, default=None, help="Fixed vertical offset in pixels for pseudo-3D. If None, auto-scale.")
    args = ap.parse_args()
    args.homography_size = tuple(args.homography_size)

    # [추가] 새 시퀀스 시작 시 SORT 상태 초기화 -----
    reset_sort_state()
    confirmed_tracks = []
    # ------------------------------------------

    # [추가] bevMap -----------------------------
    bev = BEVMap(
        world_min=(-100.0, -50.0),
        world_max=(-30.0, 40.0),
        size_px=(700, 900),      
        flip_y=False,
        car_size_m=(4.5, 1.8),
        out_dir="None"            # 저장하려면 예: "bev_frames"
    )
    # ------------------------------------------

    # paths
    if args.save_viz and not os.path.isdir("./image"):
        os.makedirs("./image", exist_ok=True)
    # robust model path (works from any CWD)
    proj_root = Path(__file__).resolve().parents[0]  # script directory
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (proj_root.parent / model_path).resolve()

    # init
    infer = Inferencer(str(model_path), is_half=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_id = 0
    processed = 0
    t_start = time.time()

    print(f"Video: {args.video} | FPS={fps:.2f} | Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting...")
            break

        # skip frames if step>1
        if (frame_id % args.step) != 0:
            frame_id += 1
            continue

        # timestamp (sec) from video timeline
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        t0 = time.monotonic()
        det = infer.run(frame)                # torch.Tensor (n,6)
        det_np = det.cpu().numpy()
        df_det = results_to_df(det, ts=ts)    # (ts,cx,cy,f1x,f1y,f2x,f2y)
        df_obs = detections_df_to_obstacles_df(df_det, camera=args.camera, homography_size=args.homography_size)
        t1 = time.monotonic()

        # [추가] SORT 업데이트 ----------
        if len(df_obs):
            dets_xyyaw = df_obs[["cx","cy","yaw"]].to_numpy(dtype=float)
            tracks = sort_update(
                dets_xyyaw,
                timestamp=float(ts),
                dist_threshold=2.0,   # 오차 임계값
                max_age=8,           # 객체 사라져도 id 유지하는 횟수
                min_hits=2
            )
            confirmed_tracks = [t for t in tracks if t["confirmed"]]

            # >>> 여기에서 정렬된(확정) 트랙을 UDP로 전송 <<<
            _send_tracks_udp(confirmed_tracks, frame_id)

            # SORT 추적 결과 저장
            if args.save_csv and len(confirmed_tracks):
                tracks_df = pd.DataFrame(confirmed_tracks)
                tracks_df.insert(0, "frame_id", int(frame_id))  # 맨 앞에 프레임 번호 컬럼 추가
                out_csv = args.csv_path.replace(".csv", "_tracks.csv")
                tracks_df.to_csv(
                    out_csv,
                    mode="a",
                    index=False,
                    header=not os.path.exists(out_csv)
                )

            # BEV viz
            if len(confirmed_tracks):
                bev_img = bev.draw_frame(confirmed_tracks, frame_id=frame_id)
                cv2.imshow("BEV Map", bev_img)
        # ------------------------------

        # viz
        if args.viz or args.save_viz or args.viz_3d:
            disp = frame.copy()
            #disp1 = np.array(visualize_detections(det_np, disp, save_path=None))
            disp2 = np.array(visualize_detections_3d(det_np, disp, save_path=None))
            if args.viz or args.viz_3d:
                #cv2.imshow("Video + 2.5D → obstacles", disp)
                #cv2.imshow("Video + 2.5D → obstacles1", disp1)
                cv2.imshow("Video + 2.5D → obstacles2", disp2)
            if args.save_viz or args.viz_3d:
                cv2.imwrite(f"./image/{frame_id:06}_viz.png", disp2)
        

        # BEV 좌표 저장 (수정함)
        if args.save_csv and len(df_obs):
            df_obs_with_frame = df_obs.copy()
            df_obs_with_frame.insert(0, "frame_id", int(frame_id))
            df_obs_with_frame.to_csv(args.csv_path, mode="a", index=False, header=not os.path.exists(args.csv_path))

        processed += 1
        if processed % 30 == 0:
            elapsed = time.time() - t_start
            print(f"[{frame_id:06}] proc={processed}, fps~={processed/max(elapsed,1e-6):.2f}, det={len(df_det)}, obs={len(df_obs)}, infer={(t1-t0)*1000:.2f}ms")

        frame_id += 1
        key = cv2.waitKey(1) & 0xFF  # 키 입력 대기
        if key == ord('q'):  # q 눌리면 종료
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
