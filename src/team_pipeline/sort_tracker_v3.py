# sort_tracker.py
# Kalman Filter + Hungarian (linear_sum_assignment) 기반 SORT (월드좌표[m])
from __future__ import annotations
from typing import List, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment

def _ensure_state():
    if not hasattr(sort_update, "_state"):
        sort_update._state = {"tracks": [], "next_id": 1}

def _kf_predict(trk: dict, dt: float, q_pos: float, q_vel: float):
    # State: x = [cx, cy, vx, vy]
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]], dtype=float)
    Q = np.diag([q_pos, q_pos, q_vel, q_vel]) * max(dt, 1e-3)
    trk["x"] = F @ trk["x"]
    trk["P"] = F @ trk["P"] @ F.T + Q

def _kf_update(trk: dict, z: np.ndarray, r_pos: float):
    # Measurement: z = [cx, cy]
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)
    R = np.array([[r_pos, 0],
                  [0, r_pos]], dtype=float)
    y = z - (H @ trk["x"])
    S = H @ trk["P"] @ H.T + R
    K = trk["P"] @ H.T @ np.linalg.inv(S)
    trk["x"] = trk["x"] + (K @ y)
    trk["P"] = (np.eye(4) - K @ H) @ trk["P"]

def sort_update(
    detections: np.ndarray,
    timestamp: float,
    dist_threshold: float = 2.0,   # 매칭 게이팅 거리 [m]
    max_age: int = 8,              # 연속 미검출 허용 프레임 수
    min_hits: int = 2,             # 확정까지 필요한 히트 수
    q_pos: float = 0.01,            # 프로세스 잡음(위치)
    q_vel: float = 1.0,            # 프로세스 잡음(속도)
    r_pos: float = 0.5,            # 측정 잡음(위치)
    max_dt: float = 0.5,           # dt 상한(초) - 드랍 프레임 견딤
    default_dt: float = 1.0/30.0,  # 첫 프레임 등 dt 미상시
    yaw_alpha: float = 1.0,        # yaw 지수평활 파라미터 (yaw 그래도 쓸거면 1.0)
) -> List[Dict]:
    """
    detections: (N,2)=[cx,cy] or (N,3)=[cx,cy,yaw] in meters/rad
    timestamp : float seconds
    returns   : List[Dict] with keys:
                id, ts, cx, cy, yaw, vx, vy, age, hits, time_since_update, confirmed
    """
    _ensure_state()
    S = sort_update._state

    # --- 입력 정리
    if detections is None:
        dets = np.empty((0, 2), dtype=float)
        det_yaw = None
    else:
        dets = np.asarray(detections, dtype=float)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        det_yaw = dets[:, 2] if dets.shape[1] >= 3 else None
        dets = dets[:, :2] if dets.size else np.empty((0, 2), dtype=float)

    # --- 예측 단계 (각 트랙 KF predict + 에이징)
    for trk in S["tracks"]:
        last_ts = trk.get("ts", None)
        dt = (timestamp - last_ts) if last_ts is not None else default_dt
        dt = float(np.clip(dt, 1e-3, max_dt))
        _kf_predict(trk, dt, q_pos, q_vel)
        trk["age"] += 1
        trk["time_since_update"] += 1

    # --- 비용행렬 (예측 위치 vs 검출) : 유클리드 거리
    m, n = len(S["tracks"]), len(dets)
    if m > 0 and n > 0:
        pred_xy = np.array([[t["x"][0], t["x"][1]] for t in S["tracks"]])  # (m,2)
        dists = np.linalg.norm(pred_xy[:, None, :] - dets[None, :, :], axis=2)  # (m,n)
        # 게이팅: 임계 초과는 큰 비용
        cost = np.where(dists <= dist_threshold, dists, dist_threshold + 1e6)
        rows, cols = linear_sum_assignment(cost)
        matches = []
        unmatched_tracks = set(range(m))
        unmatched_dets = set(range(n))
        for r, c in zip(rows, cols):
            if cost[r, c] <= dist_threshold:  # 유효 매칭만 채택
                matches.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)
        unmatched_tracks = list(unmatched_tracks)
        unmatched_dets = list(unmatched_dets)
    else:
        matches = []
        unmatched_tracks = list(range(m))
        unmatched_dets = list(range(n))

    # --- 업데이트(매칭된 트랙)
    for i, j in matches:
        trk = S["tracks"][i]
        _kf_update(trk, dets[j], r_pos)
        # yaw 지수평활
        if det_yaw is not None and not np.isnan(det_yaw[j]):
            meas_yaw = float(det_yaw[j])
            if trk["yaw"] is None or np.isnan(trk["yaw"]):
                trk["yaw"] = meas_yaw
            else:
                trk["yaw"] = yaw_alpha * meas_yaw + (1 - yaw_alpha) * trk["yaw"]
        # 관리 필드
        trk["hits"] += 1
        trk["time_since_update"] = 0
        trk["ts"] = float(timestamp)

    # --- 새 트랙 생성(미매칭 검출)
    for j in unmatched_dets:
        cx, cy = dets[j]
        x0 = np.array([cx, cy, 0.0, 0.0], dtype=float)  # 초기 속도 0
        P0 = np.diag([1.0, 1.0, 10.0, 10.0]).astype(float)
        new_trk = {
            "id": S["next_id"],
            "x": x0, "P": P0,
            "yaw": (None if det_yaw is None or np.isnan(det_yaw[j]) else float(det_yaw[j])),
            "age": 1, "hits": 1, "time_since_update": 0,
            "ts": float(timestamp),
        }
        S["tracks"].append(new_trk)
        S["next_id"] += 1

    # --- 오래된 트랙 제거(미업데이트 초과)
    S["tracks"] = [t for t in S["tracks"] if t["time_since_update"] <= max_age]

    # --- 결과 패키징
    out: List[Dict] = []
    for trk in S["tracks"]:
        confirmed = trk["hits"] >= min_hits
        out.append({
            "id": int(trk["id"]),
            "ts": float(trk["ts"]),
            "cx": float(trk["x"][0]),
            "cy": float(trk["x"][1]),
            "vx": float(trk["x"][2]),
            "vy": float(trk["x"][3]),
            "yaw": (None if trk["yaw"] is None else float(trk["yaw"])),
            "age": int(trk["age"]),
            "hits": int(trk["hits"]),
            "time_since_update": int(trk["time_since_update"]),
            "confirmed": bool(confirmed),
        })
    return out

def reset_sort_state():
    """시퀀스를 새로 시작할 때 내부 상태 초기화."""
    if hasattr(sort_update, "_state"):
        delattr(sort_update, "_state")
