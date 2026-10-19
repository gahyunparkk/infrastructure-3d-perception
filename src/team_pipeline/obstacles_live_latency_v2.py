# import argparse
# import os
# import time
# from typing import List

# import cv2
# import depthai as dai
# import numpy as np
# import pandas as pd

# import math
# import socket
# import threading
# import queue

# # 다른 파일에서 필요한 모듈 import
# from detection_inference_onnx_v2 import Inferencer, visualize_detections, visualize_detections_3d, results_to_df
# from geometry import complete_parallelograms, get_heading
# from projection import pixel_to_world_plane, rescale_pixels
# from calibration import H

# from sort_tracker_v4 import sort_update, reset_sort_state
# from bevMap_v1_rc_map import BEVMap

# # ==== UDP 설정 (PC2 수신 IP/포트로 바꿔주세요) ====
# UDP_IP = "127.0.0.1" 
# UDP_PORT = 5009
# _udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# _MAX_UDP_PAYLOAD = 1400

# # 스레드 간 데이터 공유를 위한 큐 설정
# q_frames = queue.Queue(maxsize=1)
# q_results = queue.Queue(maxsize=1)
# q_tracks = queue.Queue(maxsize=1)
# q_vis_data = queue.Queue(maxsize=1)

# # 스레드 종료를 위한 이벤트
# stop_event = threading.Event()
# t1 = 0

# # UDP 전송 및 기타 유틸리티 함수
# def _udp_send_lines(lines):
#     if not lines: return
#     buf = ""
#     for ln in lines:
#         if len(buf) + len(ln) + 1 > _MAX_UDP_PAYLOAD:
#             _udp_sock.sendto(buf.encode(), (UDP_IP, UDP_PORT))
#             buf = ln
#         else:
#             if buf: buf += "\n"
#             buf += ln
#     if buf: _udp_sock.sendto(buf.encode(), (UDP_IP, UDP_PORT))

# def _send_tracks_udp(tracks, frame_id=None, yaw_in_deg=True):
#     if not tracks: return
#     lines = []
#     for t in tracks:
#         cx = float(t["cx"])
#         cy = float(t["cy"])
#         yaw = t.get("yaw", None)
#         if yaw is None or np.isnan(yaw): yaw_out = float("nan")
#         else: yaw_out = math.degrees(float(yaw)) if yaw_in_deg else float(yaw)
#         lines.append(f"{int(t['id'])} {cx:.2f} {cy:.2f} {yaw_out:.2f}")
#     _udp_send_lines(lines)

# INPUT_IMAGE_SIZE = (1440, 810)

# def detections_df_to_obstacles_df(df: pd.DataFrame, camera: str, homography_size=(1920,1080)) -> pd.DataFrame:
#     rescale = (lambda x: x) if homography_size == INPUT_IMAGE_SIZE else (lambda x: rescale_pixels(x, src=INPUT_IMAGE_SIZE, target=homography_size))
#     front_1 = np.dstack([df['f1x'], df['f1y']]).reshape(-1, 2)
#     front_2 = np.dstack([df['f2x'], df['f2y']]).reshape(-1, 2)
#     centers = np.dstack([df['cx'], df['cy']]).reshape(-1, 2)
#     H_cam = H[camera]
#     front_1_p = pixel_to_world_plane(front_1, H_cam, rescale=rescale)
#     front_2_p = pixel_to_world_plane(front_2, H_cam, rescale=rescale)
#     centers_p = pixel_to_world_plane(centers, H_cam, rescale=rescale)
#     corners_p = complete_parallelograms(front_1_p, front_2_p, centers_p)
#     headings_p = get_heading(corners_p[:, 0], corners_p[:, 2], False)
#     out = pd.DataFrame({
#         'ts': df['ts'].values,
#         'cx': corners_p[:, -1, 0],
#         'cy': corners_p[:, -1, 1],
#         'yaw': headings_p
#     })
#     return out

# def build_pipeline(width=1920, height=1080):
#     pipeline = dai.Pipeline()
#     camRgb = pipeline.createColorCamera()
#     camRgb.setFps(30)
#     camRgb.setPreviewSize(1280, 720)
#     camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
#     camRgb.setPreviewSize(width, height)
#     camRgb.setIspScale(1, 1)
#     camRgb.setInterleaved(False)
#     xoutRgb = pipeline.createXLinkOut()
#     xoutRgb.setStreamName("rgb")
#     camRgb.preview.link(xoutRgb.input)
#     xinControl = pipeline.createXLinkIn()
#     xinControl.setStreamName("control")
#     xinControl.out.link(camRgb.inputControl)
#     camRgb.inputControl.setQueueSize(1)
#     camRgb.inputControl.setBlocking(False)
#     return pipeline

# # ==== 스레드 함수들 ====

# def camera_thread(rgbQueue, q_frames, stop_event):
#     """카메라 프레임 수신 및 큐에 전달"""
#     while not stop_event.is_set():
#         t_extract_start = time.monotonic()
#         rgbData = rgbQueue.get() 
#         t_extract_end = time.monotonic()
        
#         rgbFrame = rgbData.getCvFrame()
#         device_ts = rgbData.getTimestamp().total_seconds()
        
#         extract_ms = (t_extract_end - t_extract_start) * 1000.0

#         if not q_frames.full():
#             q_frames.put((rgbFrame, device_ts, t_extract_start, t_extract_end))

# def inference_thread(inferencer, q_frames, q_results, stop_event):
#     """ONNX 추론 실행 및 결과 큐에 전달"""
#     while not stop_event.is_set():
#         try:
#             rgbFrame, device_ts, t_extract_start, t_extract_end = q_frames.get(timeout=0.1)
#             q_frames.task_done()
#         except queue.Empty:
#             continue

#         t_infer_start = time.monotonic()
#         det = inferencer.run(rgbFrame)
#         det_np = det.cpu().numpy()
#         t_infer_end = time.monotonic()
        
#         infer_ms = (t_infer_end - t_infer_start) * 1000.0

#         if not q_results.full():
#             q_results.put((det_np, device_ts, t_extract_start, t_extract_end, infer_ms, rgbFrame))

# def processing_thread(args, q_results, q_tracks, q_vis_data, stop_event):
#     """추론 결과 후처리 및 시각화 데이터 전달"""
#     reset_sort_state()
#     bev = BEVMap(
#         world_min=(-10.0, -10.0),
#         world_max=(10.0, 10.0),
#         size_px=(800, 800),      
#         flip_y=False,
#         car_size_m=(3.2, 1.2),
#         out_dir=None            # 저장하려면 예: "bev_frames"
#     )
#     frame_id = 0
#     while not stop_event.is_set():
#         try:
#             det_np, device_ts, t_extract_start, t_extract_end, infer_ms, rgbFrame = q_results.get(timeout=0.1)
#             q_results.task_done()
#         except queue.Empty:
#             continue

#         t_post_start = time.monotonic()
        
#         df_det = results_to_df(det_np, ts=device_ts)
#         df_obs = detections_df_to_obstacles_df(df_det, camera=args.camera, homography_size=args.homography_size)
        
#         confirmed_tracks = []
#         if len(df_obs):
#             dets_xyyaw = df_obs[["cx", "cy", "yaw"]].to_numpy(dtype=float)
#             tracks = sort_update(dets_xyyaw, timestamp=float(device_ts))
#             confirmed_tracks = [t for t in tracks if t["confirmed"]]
        
#         t_post_end = time.monotonic()
#         post_infer_ms = (t_post_end - t_post_start) * 1000.0
        
#         if not q_tracks.full():
#             q_tracks.put((confirmed_tracks, device_ts, t_extract_start, t_extract_end, infer_ms, post_infer_ms))

#         if not q_vis_data.full():
#             bev_img = bev.draw_frame(confirmed_tracks, frame_id=frame_id)
#             disp1 = np.array(visualize_detections(det_np, rgbFrame.copy(), save_path=None))
#             disp2 = np.array(visualize_detections_3d(det_np, rgbFrame.copy(), save_path=None))
#             q_vis_data.put((bev_img, disp1, disp2))
#             frame_id += 1

# def udp_thread(q_tracks, stop_event):
#     """
#     Tracks results from the processing thread and sends them via UDP.
#     Calculates and prints the final latency and FPS.
#     """
#     frame_id = 0
    
#     # Track time of the last successful frame processed by this thread
#     t_last_frame_processed = time.monotonic()

#     while not stop_event.is_set():
#         try:
#             # Get data and timestamps from the queue
#             confirmed_tracks, device_ts, t_extract_start, t_extract_end, infer_ms, post_infer_ms = q_tracks.get(timeout=0.1)
#             q_tracks.task_done()
#         except queue.Empty:
#             # If the queue is empty, continue to the next iteration without updating the timer
#             continue
        
#         # --- Timings for the current frame ---
#         t_udp_start = time.monotonic()
#         _send_tracks_udp(confirmed_tracks, frame_id)
#         t_udp_end = time.monotonic()

#         # Calculate time for each stage
#         extract_ms = (t_extract_end - t_extract_start) * 1000.0
#         udp_ms = (t_udp_end - t_udp_start) * 1000.0
#         e2e_ms = (t_udp_end - t_extract_start) * 1000.0
        
#         # --- Calculate FPS ---
#         current_time = time.monotonic()
#         # Time since the last frame was processed by this thread
#         time_since_last_frame_ms = (current_time - t_last_frame_processed) * 1000.0
        
#         # Avoid division by zero
#         fps = 1000.0 / time_since_last_frame_ms if time_since_last_frame_ms > 0 else float('inf')

#         # Update the timestamp for the next iteration
#         t_last_frame_processed = current_time

#         print(f"[{frame_id:06}] "
#               f"extract={extract_ms:.2f} ms | "
#               f"infer={infer_ms:.2f} ms | "
#               f"post_infer={post_infer_ms:.2f} ms | "
#               f"udp_send={udp_ms:.2f} ms | "
#               f"e2e={e2e_ms:.2f} ms | "
#               f"FPS={fps:.2f}")
        
#         frame_id += 1
        
# def main():
#     parser = argparse.ArgumentParser(description="DepthAI live → ONNX → obstacles (world coords + yaw)")
#     parser.add_argument("--model", type=str, default="checkpoints/model_v2_half.onnx")
#     parser.add_argument("--camera", choices=H.keys(), default=list(H.keys())[0])
#     parser.add_argument("--homography_size", nargs=2, type=int, default=(1920,1080))
#     parser.add_argument("--iso", type=int, default=800)
#     parser.add_argument("--exposure_us", type=int, default=20000)
#     parser.add_argument("--viz", action="store_true")
#     parser.add_argument("--save_viz", action="store_true")
#     parser.add_argument("--save_csv", action="store_true")
#     parser.add_argument("--csv_path", type=str, default="obstacles_live.csv")
#     parser.add_argument("--frame_width", type=int, default=1440)
#     parser.add_argument("--frame_height", type=int, default=810)
#     parser.add_argument("--viz-3d", action="store_true", help="Save pseudo-3D visualization (pixel-offset extrusions)")
#     parser.add_argument("--viz-3d-dy", type=int, default=None, help="Fixed vertical offset in pixels for pseudo-3D. If None, auto-scale.")
#     args = parser.parse_args()
#     args.homography_size = tuple(args.homography_size)

#     if args.save_viz and not os.path.isdir("./image"):
#         os.makedirs("./image", exist_ok=True)
    
#     infer = Inferencer(args.model, is_half=True)
#     pipeline = build_pipeline(args.frame_width, args.frame_height)

#     with dai.Device(pipeline) as device:
#         rgbQueue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
#         controlQueue = device.getInputQueue("control")

#         ctrl = dai.CameraControl()
#         ctrl.setManualExposure(args.exposure_us, args.iso)
#         controlQueue.send(ctrl)

#         cam_t = threading.Thread(target=camera_thread, args=(rgbQueue, q_frames, stop_event), daemon=True)
#         inf_t = threading.Thread(target=inference_thread, args=(infer, q_frames, q_results, stop_event), daemon=True)
#         proc_t = threading.Thread(target=processing_thread, args=(args, q_results, q_tracks, q_vis_data, stop_event), daemon=True)
#         udp_t = threading.Thread(target=udp_thread, args=(q_tracks, stop_event), daemon=True)

#         cam_t.start()
#         inf_t.start()
#         proc_t.start()
#         udp_t.start()

#         print("Multi-threaded pipeline started. Press 'q' to quit.")

#         try:
#             while not stop_event.is_set():
#                 if not q_vis_data.empty():
#                     bev_img, disp1, disp2 = q_vis_data.get()
#                     q_vis_data.task_done()
#                     cv2.imshow("BEV Map", bev_img)
#                     if args.viz or args.viz_3d:
#                         cv2.imshow("Video + 2.5D → obstacles (3D overlay)", disp1)
#                         cv2.imshow("Video + 3D → obstacles (3D overlay)", disp2)

#                 key = cv2.waitKey(1) & 0xFF
#                 if key == ord('q'):
#                     print("Quit signal received.")
#                     stop_event.set()
#         except KeyboardInterrupt:
#             print("Keyboard interrupt received.")
#             stop_event.set()
#         finally:
#             cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

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

# 다른 파일에서 필요한 모듈 import
from detection_inference_onnx_v2 import Inferencer, visualize_detections, visualize_detections_3d, results_to_df
from geometry import complete_parallelograms, get_heading
from projection import pixel_to_world_plane, rescale_pixels
from calibration import H

from sort_tracker_v4 import sort_update, reset_sort_state
from bevMap_v1_rc_map import BEVMap

# ==== UDP 설정 (PC2 수신 IP/포트로 바꿔주세요) ====
UDP_IP = "127.0.0.1" 
UDP_PORT = 5009
_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_MAX_UDP_PAYLOAD = 1400

# 스레드 간 데이터 공유를 위한 큐 설정
q_frames = queue.Queue(maxsize=1)
q_results = queue.Queue(maxsize=1)
q_tracks = queue.Queue(maxsize=1)
q_vis_data = queue.Queue(maxsize=1)

# 스레드 종료를 위한 이벤트
stop_event = threading.Event()
t1 = 0

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

def build_pipeline(width=1920, height=1080):
    pipeline = dai.Pipeline()
    camRgb = pipeline.createColorCamera()
    camRgb.setFps(30)
    camRgb.setPreviewSize(1280, 720)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setPreviewSize(width, height)
    camRgb.setIspScale(1, 1)
    camRgb.setInterleaved(False)
    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)
    xinControl = pipeline.createXLinkIn()
    xinControl.setStreamName("control")
    xinControl.out.link(camRgb.inputControl)
    camRgb.inputControl.setQueueSize(1)
    camRgb.inputControl.setBlocking(False)
    return pipeline

# ==== 스레드 함수들 ====

def camera_thread(rgbQueue, q_frames, stop_event):
    """카메라 프레임 수신 및 큐에 전달"""
    while not stop_event.is_set():
        t_extract_start = time.monotonic()
        rgbData = rgbQueue.get() 
        t_extract_end = time.monotonic()
        
        rgbFrame = rgbData.getCvFrame()
        device_ts = rgbData.getTimestamp().total_seconds()
        
        extract_ms = (t_extract_end - t_extract_start) * 1000.0

        if not q_frames.full():
            q_frames.put((rgbFrame, device_ts, t_extract_start, t_extract_end))

def inference_thread(inferencer, q_frames, q_results, stop_event):
    """ONNX 추론 실행 및 결과 큐에 전달"""
    while not stop_event.is_set():
        try:
            rgbFrame, device_ts, t_extract_start, t_extract_end = q_frames.get(timeout=0.1)
            q_frames.task_done()
        except queue.Empty:
            continue

        t_infer_start = time.monotonic()
        det = inferencer.run(rgbFrame)
        det_np = det.cpu().numpy()
        t_infer_end = time.monotonic()
        
        infer_ms = (t_infer_end - t_infer_start) * 1000.0

        if not q_results.full():
            q_results.put((det_np, device_ts, t_extract_start, t_extract_end, infer_ms, rgbFrame))

def processing_thread(args, q_results, q_tracks, q_vis_data, stop_event):
    """추론 결과 후처리 및 시각화 데이터 전달"""
    reset_sort_state()
    
    # vvvvvvvvvvvvvv ADDED vvvvvvvvvvvvvv
    # This block defines the absolute path to the car image.
    # The 'r' prefix creates a raw string, which is useful for Windows paths
    # that contain backslashes.
    # <<<---!!! 중요: 여기에 차량 이미지의 절대 경로를 지정합니다 !!!--->>>
    # r"..."는 Windows의 역슬래시(\)를 그대로 인식하게 해주는 Raw String 표기법입니다.
    CAR_IMAGE_ABSOLUTE_PATH = r"C:\Users\USER\Desktop\Capstone\2.5d-object-detection\src\car_logo.png"
    # ^^^^^^^^^^^^^^ ADDED ^^^^^^^^^^^^^^

    bev = BEVMap(
        world_min=(-10.0, -10.0),
        world_max=(10.0, 10.0),
        size_px=(800, 800),     
        flip_y=False,
        car_size_m=(2.6, 1.3),
        out_dir=None,           # 저장하려면 예: "bev_frames"
        # vvvvvvvvvvvvvv CHANGED vvvvvvvvvvvvvv
        # The 'car_image_path' argument is now provided with the variable
        # defined above, allowing the BEVMap class to load and draw the car image.
        car_image_path=CAR_IMAGE_ABSOLUTE_PATH  # <-- 수정된 부분: 이미지 경로 전달
        # ^^^^^^^^^^^^^^ CHANGED ^^^^^^^^^^^^^^
    )
    frame_id = 0
    while not stop_event.is_set():
        try:
            det_np, device_ts, t_extract_start, t_extract_end, infer_ms, rgbFrame = q_results.get(timeout=0.1)
            q_results.task_done()
        except queue.Empty:
            continue

        t_post_start = time.monotonic()
        
        df_det = results_to_df(det_np, ts=device_ts)
        df_obs = detections_df_to_obstacles_df(df_det, camera=args.camera, homography_size=args.homography_size)
        
        confirmed_tracks = []
        if len(df_obs):
            dets_xyyaw = df_obs[["cx", "cy", "yaw"]].to_numpy(dtype=float)
            tracks = sort_update(dets_xyyaw, timestamp=float(device_ts))
            confirmed_tracks = [t for t in tracks if t["confirmed"]]
        
        t_post_end = time.monotonic()
        post_infer_ms = (t_post_end - t_post_start) * 1000.0
        
        if not q_tracks.full():
            q_tracks.put((confirmed_tracks, device_ts, t_extract_start, t_extract_end, infer_ms, post_infer_ms))

        if not q_vis_data.full():
            bev_img = bev.draw_frame(confirmed_tracks, frame_id=frame_id)
            disp1 = np.array(visualize_detections(det_np, rgbFrame.copy(), save_path=None))
            disp2 = np.array(visualize_detections_3d(det_np, rgbFrame.copy(), save_path=None))
            q_vis_data.put((bev_img, disp1, disp2))
            frame_id += 1

def udp_thread(q_tracks, stop_event):
    """
    Tracks results from the processing thread and sends them via UDP.
    Calculates and prints the final latency and FPS.
    """
    frame_id = 0
    
    # Track time of the last successful frame processed by this thread
    t_last_frame_processed = time.monotonic()

    while not stop_event.is_set():
        try:
            # Get data and timestamps from the queue
            confirmed_tracks, device_ts, t_extract_start, t_extract_end, infer_ms, post_infer_ms = q_tracks.get(timeout=0.1)
            q_tracks.task_done()
        except queue.Empty:
            # If the queue is empty, continue to the next iteration without updating the timer
            continue
        
        # --- Timings for the current frame ---
        t_udp_start = time.monotonic()
        _send_tracks_udp(confirmed_tracks, frame_id)
        t_udp_end = time.monotonic()

        # Calculate time for each stage
        extract_ms = (t_extract_end - t_extract_start) * 1000.0
        udp_ms = (t_udp_end - t_udp_start) * 1000.0
        e2e_ms = (t_udp_end - t_extract_start) * 1000.0
        
        # --- Calculate FPS ---
        current_time = time.monotonic()
        # Time since the last frame was processed by this thread
        time_since_last_frame_ms = (current_time - t_last_frame_processed) * 1000.0
        
        # Avoid division by zero
        fps = 1000.0 / time_since_last_frame_ms if time_since_last_frame_ms > 0 else float('inf')

        # Update the timestamp for the next iteration
        t_last_frame_processed = current_time

        print(f"[{frame_id:06}] "
              f"extract={extract_ms:.2f} ms | "
              f"infer={infer_ms:.2f} ms | "
              f"post_infer={post_infer_ms:.2f} ms | "
              f"udp_send={udp_ms:.2f} ms | "
              f"e2e={e2e_ms:.2f} ms | "
              f"FPS={fps:.2f}")
        
        frame_id += 1
        
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
    args = parser.parse_args()
    args.homography_size = tuple(args.homography_size)

    if args.save_viz and not os.path.isdir("./image"):
        os.makedirs("./image", exist_ok=True)
    
    infer = Inferencer(args.model, is_half=True)
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