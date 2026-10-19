import argparse
import os
import time
from typing import List

import cv2
import depthai as dai
import numpy as np
import pandas as pd

import math
import socket
import threading
import queue
import torch

from detection_inference_onnx_v2 import Inferencer, visualize_detections, visualize_detections_3d, results_to_df
from geometry import complete_parallelograms, get_heading
from projection import pixel_to_world_plane, rescale_pixels
from calibration import H
from sort_tracker_v4 import sort_update, reset_sort_state
from bevMap_v1_rc_map import BEVMap

# device 변수 추가
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== UDP 설정 (PC2 수신 IP/포트로 바꿔주세요) ====
UDP_IP = "127.0.0.1" 
UDP_PORT = 5009
_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_MAX_UDP_PAYLOAD = 1400

# UDP 전송 및 기타 유틸리티 함수
def _udp_send_lines(lines):
    if not lines: return
    buf = ""
    for ln in lines:
        if len(buf) + len(ln) + 1 > _MAX_UDP_PAYLOAD:
            _udp_sock.sendto(buf.encode(), (UDP_IP, UDP_PORT))
            buf = ln
        else:
            if buf: buf += "\n"
            buf += ln
    if buf: _udp_sock.sendto(buf.encode(), (UDP_IP, UDP_PORT))

def _send_tracks_udp(tracks, frame_id=None, yaw_in_deg=True):
    if not tracks: return
    lines = []
    for t in tracks:
        cx = float(t["cx"])
        cy = float(t["cy"])
        yaw = t.get("yaw", None)
        if yaw is None or np.isnan(yaw): yaw_out = float("nan")
        else: yaw_out = math.degrees(float(yaw)) if yaw_in_deg else float(yaw)
        lines.append(f"{int(t['id'])} {cx:.2f} {cy:.2f} {yaw_out:.2f}")
    _udp_send_lines(lines)

INPUT_IMAGE_SIZE = (1440, 810)

# H 행렬을 PyTorch 텐서로 변환하고 GPU로 옮김
H_tensors = {
    key: torch.from_numpy(val).to(device, dtype=torch.float32) 
    for key, val in H.items()
}

def detections_df_to_obstacles_df(df: pd.DataFrame, camera: str, homography_size=(1920,1080)) -> pd.DataFrame:
    rescale = (lambda x: x) if homography_size == INPUT_IMAGE_SIZE else (lambda x: rescale_pixels(x, src=INPUT_IMAGE_SIZE, target=homography_size))

    front_1 = np.dstack([df['f1x'], df['f1y']]).reshape(-1, 2)
    front_2 = np.dstack([df['f2x'], df['f2y']]).reshape(-1, 2)
    centers = np.dstack([df['cx'], df['cy']]).reshape(-1, 2)

    H_cam = H[camera]
    front_1_p = pixel_to_world_plane(front_1, H_cam, rescale=rescale)
    front_2_p = pixel_to_world_plane(front_2, H_cam, rescale=rescale)
    centers_p = pixel_to_world_plane(centers, H_cam, rescale=rescale)

    corners_p = complete_parallelograms(front_1_p, front_2_p, centers_p)
    headings_p = get_heading(corners_p[:, 0], corners_p[:, 2], False)
    out = pd.DataFrame({
        'ts': df['ts'].values,
        'cx': corners_p[:, -1, 0],
        'cy': corners_p[:, -1, 1],
        'yaw': headings_p
    })
    return out

def main():
    parser = argparse.ArgumentParser(description="DepthAI live → ONNX → obstacles (world coords + yaw)")
    parser.add_argument("--model", type=str, default="checkpoints/model_v2_half.onnx")
    parser.add_argument("--camera", choices=H.keys(), default=list(H.keys())[0])
    parser.add_argument("--homography_size", nargs=2, type=int, default=(1920,1080))
    parser.add_argument("--iso", type=int, default=800)
    parser.add_argument("--exposure_us", type=int, default=20000)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--csv_path", type=str, default="obstacles_live.csv")
    parser.add_argument("--frame_width", type=int, default=1440)
    parser.add_argument("--frame_height", type=int, default=810)
    parser.add_argument("--viz-3d", action="store_true", help="Save pseudo-3D visualization (pixel-offset extrusions)")
    parser.add_argument("--viz-3d-dy", type=int, default=None, help="Fixed vertical offset in pixels for pseudo-3D. If None, auto-scale.")
    
    parser.add_argument("--video", type=str, help="Path to the input video file (for video mode)")
    
    args = parser.parse_args()
    args.homography_size = tuple(args.homography_size)

    reset_sort_state()
    confirmed_tracks = []

    bev = BEVMap(
        world_min=(-10.0, -10.0),
        world_max=(10.0, 10.0),
        size_px=(800, 800),
        flip_y=False,
        car_size_m=(3.2, 1.2),
        out_dir="results/rc_bev_withSORT_map"
    )

    if args.save_viz and not os.path.isdir("./image"):
        os.makedirs("./image", exist_ok=True)
    
    infer = Inferencer(args.model, is_half=True)
    
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return

        frame_id = 0
        print(f"Video processing started from: {args.video}. Press 'q' to quit.")
        
        while True:
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                print("Video stream ended.")
                break
                
            rgbFrame = cv2.resize(frame, (args.frame_width, args.frame_height))
            device_ts = time.time()

            t2 = time.monotonic()
            det = infer.run(rgbFrame)
            t3 = time.monotonic()
            infer_ms = (t3 - t2) * 1000.0

            t5 = time.monotonic()
            
            # --- 수정된 부분 ---
            # GPU 텐서인 'det'를 CPU로 옮기고 NumPy 배열로 변환
            det_np = det.cpu().numpy()
            
            df_det = results_to_df(det_np, ts=device_ts)
            df_obs = detections_df_to_obstacles_df(
                df_det, camera=args.camera, homography_size=args.homography_size
            )
            # --- 수정 끝 ---
            
            if len(df_obs):
                dets_xyyaw = df_obs[["cx","cy","yaw"]].to_numpy(dtype=float)
                tracks = sort_update(
                    dets_xyyaw,
                    timestamp=float(device_ts),
                    dist_threshold=4.0,
                    max_age=8,
                    min_hits=3,
                    q_pos=0.1, q_vel=2.0, r_pos=1.5,
                    use_mahalanobis=True, mahalanobis_thresh=16.0,
                    yaw_cost_weight=0.1, iou_cost_weight=0.8,
                    motion_cost_weight=0.6, min_speed_for_motion_cost=0.5,
                    stop_speed_thresh=0.4, stop_damping=0.4,
                )
                confirmed_tracks = [t for t in tracks if t["confirmed"]]
                _send_tracks_udp(confirmed_tracks, frame_id)

                if args.save_csv and len(confirmed_tracks):
                    tracks_df = pd.DataFrame(confirmed_tracks)
                    tracks_df.insert(0, "frame_id", int(frame_id))
                    out_csv = args.csv_path.replace(".csv", "_tracks.csv")
                    tracks_df.to_csv(
                        out_csv,
                        mode="a",
                        index=False,
                        header=not os.path.exists(out_csv)
                    )

                if len(confirmed_tracks):
                    bev_img = bev.draw_frame(confirmed_tracks, frame_id=frame_id)
                    cv2.imshow("BEV Map", bev_img)

            if args.viz or args.save_viz or args.viz_3d:
                disp = rgbFrame.copy()
                disp1 = np.array(visualize_detections(det_np, disp, save_path=None))
                disp2 = np.array(visualize_detections_3d(det_np, disp, save_path=None))
                if args.viz or args.viz_3d:
                    cv2.imshow("Video + 2.5D → obstacles (3D overlay)", disp1)
                    cv2.imshow("Video + 3D → obstacles (3D overlay)", disp2)
                if args.save_viz or args.viz_3d:
                    cv2.imwrite(f"./image/{frame_id:06}_viz.png", disp2)

            if args.save_csv:
                df_obs.to_csv(args.csv_path, mode="a", index=False, header=not os.path.exists(args.csv_path))
                if len(df_obs):
                    df_obs_with_frame = df_obs.copy()
                    df_obs_with_frame.insert(0, "frame_id", int(frame_id))
                    df_obs_with_frame.to_csv(args.csv_path, mode="a", index=False, header=not os.path.exists(args.csv_path))

            t6 = time.monotonic()
            total_ms = (t6 - t0) * 1000.0
            post_infer_ms = (t6 - t5) * 1000.0
            
            fps = 1000.0 / total_ms if total_ms > 0 else float('inf')
            
            print(f"[{frame_id:06}] "
                  f"infer={infer_ms:.2f} ms | "
                  f"post_infer={post_infer_ms:.2f} ms | "
                  f"total={total_ms:.2f} ms | "
                  f"FPS={fps:.2f}")

            frame_id += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        pipeline = build_pipeline(args.frame_width, args.frame_height)
        with dai.Device(pipeline) as device:
            rgbQueue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            controlQueue = device.getInputQueue("control")

            ctrl = dai.CameraControl()
            ctrl.setManualExposure(args.exposure_us, args.iso)
            controlQueue.send(ctrl)

            cam_t = threading.Thread(target=camera_thread, args=(rgbQueue, q_frames, stop_event), daemon=True)
            inf_t = threading.Thread(target=inference_thread, args=(infer, q_frames, q_results, stop_event), daemon=True)
            proc_t = threading.Thread(target=processing_thread, args=(args, q_results, q_tracks, q_vis_data, stop_event), daemon=True)
            udp_t = threading.Thread(target=udp_thread, args=(q_tracks, stop_event), daemon=True)

            cam_t.start()
            inf_t.start()
            proc_t.start()
            udp_t.start()

            print("Multi-threaded pipeline started. Press 'q' to quit.")

            try:
                while not stop_event.is_set():
                    if not q_vis_data.empty():
                        bev_img, disp1, disp2 = q_vis_data.get()
                        q_vis_data.task_done()
                        cv2.imshow("BEV Map", bev_img)
                        if args.viz or args.viz_3d:
                            cv2.imshow("Video + 2.5D → obstacles (3D overlay)", disp1)
                            cv2.imshow("Video + 3D → obstacles (3D overlay)", disp2)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quit signal received.")
                        stop_event.set()
            except KeyboardInterrupt:
                print("Keyboard interrupt received.")
                stop_event.set()
            finally:
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()