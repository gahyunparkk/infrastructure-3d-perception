# video_obstacles.py — Video file → ONNX → obstacles (world coords + yaw)
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

#from detection_inference_onnx import Inferencer, visualize_detections, results_to_df
from detection_inference_onnx import Inferencer, visualize_detections, visualize_detections_3d, results_to_df
from geometry import complete_parallelograms, get_heading
from projection import pixel_to_world_plane, rescale_pixels
from calibration import H

INPUT_IMAGE_SIZE = (1440, 810)

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

        # viz
        if args.viz or args.save_viz or args.viz_3d:
            disp = frame.copy()
            disp1 = np.array(visualize_detections(det_np, disp, save_path=None))
            disp2 = np.array(visualize_detections_3d(det_np, disp, save_path=None))
            if args.viz or args.viz_3d:
                #cv2.imshow("Video + 2.5D → obstacles", disp)
                #cv2.imshow("Video + 2.5D → obstacles1", disp1)
                cv2.imshow("Video + 2.5D → obstacles2", disp2)
            if args.save_viz or args.viz_3d:
                cv2.imwrite(f"./image/{frame_id:06}_viz.png", disp2)
        

        # CSV
        if args.save_csv and len(df_obs):
            df_obs.to_csv(args.csv_path, mode="a", index=False, header=not os.path.exists(args.csv_path))

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
