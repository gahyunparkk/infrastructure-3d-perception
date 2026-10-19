# sort_tracker.py
# Kalman Filter + Hungarian (linear_sum_assignment) 기반 SORT (월드좌표[m])
from __future__ import annotations
from typing import List, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi

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

def _box_corners(cx: float, cy: float, yaw: float, L: float, W: float) -> np.ndarray:
    """Compute oriented car box corners (4,2) in world [m]."""
    L2, W2 = L * 0.5, W * 0.5
    box_local = np.array([
        [ +L2, +W2],
        [ +L2, -W2],
        [ -L2, -W2],
        [ -L2, +W2],
    ], dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return (box_local @ R.T) + np.array([cx, cy])

def _oriented_box_iou(cx1: float, cy1: float, yaw1: float,
                      cx2: float, cy2: float, yaw2: float,
                      L: float, W: float) -> float:
    """IoU between two oriented car rectangles defined by center/yaw/size.
    Uses cv2.intersectConvexConvex on 4-pt convex polygons.
    """
    poly1 = _box_corners(cx1, cy1, yaw1, L, W).astype(np.float32)
    poly2 = _box_corners(cx2, cy2, yaw2, L, W).astype(np.float32)
    area1 = abs(cv2.contourArea(poly1))
    area2 = abs(cv2.contourArea(poly2))
    if area1 <= 1e-6 or area2 <= 1e-6:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(poly1, poly2)
    inter_area = max(0.0, float(inter_area))
    union = area1 + area2 - inter_area
    if union <= 1e-6:
        return 0.0
    return inter_area / union

def sort_update(
    detections: np.ndarray,
    timestamp: float,
    dist_threshold: float = 4.0,   # 매칭 게이팅 거리 [m]
    max_age: int = 8,              # 연속 미검출 허용 프레임 수
    min_hits: int = 3,             # 확정까지 필요한 히트 수
    q_pos: float = 0.1,            # 프로세스 잡음(위치)
    q_vel: float = 2.0,            # 프로세스 잡음(속도)
    r_pos: float = 1.5,            # 측정 잡음(위치)
    max_dt: float = 0.5,           # dt 상한(초) - 드랍 프레임 견딤
    default_dt: float = 1.0/30.0,  # 첫 프레임 등 dt 미상시
    yaw_alpha: float = 1.0,        # yaw 지수평활 파라미터 (yaw 그래도 쓸거면 1.0)
    use_mahalanobis: bool = True,  # Mahalanobis 거리 기반 매칭
    mahalanobis_thresh: float = 16.0,  # DOF=2, 99% chi-square ≈ 9.21
    yaw_cost_weight: float = 0.1,  # yaw 차이 가중치(라디안). 0이면 무시
    iou_cost_weight: float = 0.8,  # 박스 IoU 비용 가중치 (1-IoU)
    car_length: float = 4.5,
    car_width: float = 1.8,
    motion_cost_weight: float = 0.6,     # 운동방향 일관성 비용(수직편차/역행)
    min_speed_for_motion_cost: float = 0.5,  # m/s 이상에서만 적용
    stop_speed_thresh: float = 0.4,      # 정지 근처 속도 임계
    stop_damping: float = 0.4,           # 정지 근처 속도 감소 비율
    # --- 예측 최소화/측정 우선 옵션 ---
    freeze_on_miss: bool = False,        # 미검출 프레임: 기본은 예측 유지(Freeze 원하면 True)
    force_meas_position: bool = True,    # 매칭 시 최종 위치를 측정값(cx,cy)에 강제로 맞춤
    miss_vel_damping: float = 0.8,       # 미검출 동안 속도 감쇠 계수(작을수록 빨리 감쇠)
    miss_blend_alpha: float = 0.15,      # 미검출 시 위치 = (1-a)*마지막측정 + a*예측
    max_miss_disp_per_frame: float = 0.5,# 미검출 프레임당 최대 이동(m) 캡
    freeze_after_misses: int = 3,        # 이 이상 연속 미검출이면 완전 고정
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
        # 예측 시각을 갱신하여 미매칭 기간에도 dt 누적 과다를 방지
        trk["ts"] = float(timestamp)

    # --- 비용행렬 (예측 위치 vs 검출)
    m, n = len(S["tracks"]), len(dets)
    if m > 0 and n > 0:
        pred_xy = np.array([[t["x"][0], t["x"][1]] for t in S["tracks"]], dtype=float)  # (m,2)

        if use_mahalanobis:
            # Mahalanobis d^2 = (z - Hx)^T S^{-1} (z - Hx)
            # H = [[1,0,0,0],[0,1,0,0]]; S = H P H^T + R
            Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
            Rm = np.array([[r_pos, 0], [0, r_pos]], dtype=float)
            cost = np.full((m, n), fill_value=mahalanobis_thresh + 1e6, dtype=float)
            for i, trk in enumerate(S["tracks"]):
                S_i = Hm @ trk["P"] @ Hm.T + Rm
                try:
                    S_inv = np.linalg.inv(S_i)
                except np.linalg.LinAlgError:
                    # fallback: pseudo-inverse
                    S_inv = np.linalg.pinv(S_i)
                diff = dets - pred_xy[i]  # (n,2)
                # d2 for all detections vs track i
                d2 = np.einsum('ij,jk,ik->i', diff, S_inv, diff)
                # 기본 비용은 d2 (작을수록 좋음)
                c_row = d2.copy()
                # yaw 비용 추가(선택)
                if yaw_cost_weight > 0.0 and det_yaw is not None:
                    trk_yaw = trk.get("yaw", None)
                    if trk_yaw is not None and not np.isnan(trk_yaw):
                        dyaw = np.abs(_wrap_angle(det_yaw - float(trk_yaw)))  # (n,)
                        c_row = c_row + yaw_cost_weight * dyaw
                # IoU 비용 추가(선택, 각도/차크기 사용)
                if iou_cost_weight > 0.0 and det_yaw is not None:
                    tyaw = trk.get("yaw", None)
                    if tyaw is not None and not np.isnan(tyaw):
                        cx_t, cy_t = float(trk["x"][0]), float(trk["x"][1])
                        iou_terms = np.zeros(n, dtype=float)
                        # 계산 비용을 줄이기 위해 근접한 것들만 IoU 계산(엘립스 게이트 안쪽 후보)
                        valid = d2 <= mahalanobis_thresh
                        if np.any(valid):
                            for j in np.where(valid)[0]:
                                cx_d, cy_d = float(dets[j, 0]), float(dets[j, 1])
                                dyaw_d = float(det_yaw[j])
                                if np.isnan(dyaw_d):
                                    continue
                                # 박스 꼭짓점 → IoU
                                iou = _oriented_box_iou(cx_t, cy_t, float(tyaw), cx_d, cy_d, dyaw_d, car_length, car_width)
                                iou_terms[j] = (1.0 - iou)
                            c_row = c_row + iou_cost_weight * iou_terms
                # 운동 방향 비용 추가(선택)
                if motion_cost_weight > 0.0:
                    vx, vy = float(trk["x"][2]), float(trk["x"][3])
                    spd = float(np.hypot(vx, vy))
                    if spd >= min_speed_for_motion_cost:
                        vhat = np.array([vx, vy], dtype=float) / max(spd, 1e-6)
                        # dets - pred
                        delta = dets - pred_xy[i]
                        along = delta @ vhat  # (n,)
                        perp = delta - (along[:, None] * vhat[None, :])
                        perp_err = np.linalg.norm(perp, axis=1)
                        c_row = c_row + motion_cost_weight * (perp_err + np.maximum(0.0, -along))
                # 게이팅: 임계^2 초과는 큰 비용으로
                c_row = np.where(d2 <= mahalanobis_thresh, c_row, mahalanobis_thresh + 1e6)
                cost[i, :] = c_row
        else:
            # 유클리드 거리 + 게이팅
            dists = np.linalg.norm(pred_xy[:, None, :] - dets[None, :, :], axis=2)  # (m,n)
            # yaw 비용 추가(선택)
            if yaw_cost_weight > 0.0 and det_yaw is not None:
                yaw_term = np.zeros_like(dists)
                for i, trk in enumerate(S["tracks"]):
                    trk_yaw = trk.get("yaw", None)
                    if trk_yaw is not None and not np.isnan(trk_yaw):
                        dyaw = np.abs(_wrap_angle(det_yaw - float(trk_yaw)))  # (n,)
                        yaw_term[i, :] = yaw_cost_weight * dyaw
                dists = dists + yaw_term
            # IoU 비용 추가(선택)
            if iou_cost_weight > 0.0 and det_yaw is not None:
                iou_term = np.zeros_like(dists)
                for i, trk in enumerate(S["tracks"]):
                    tyaw = trk.get("yaw", None)
                    if tyaw is not None and not np.isnan(tyaw):
                        cx_t, cy_t = float(trk["x"][0]), float(trk["x"][1])
                        for j in range(n):
                            cx_d, cy_d = float(dets[j, 0]), float(dets[j, 1])
                            dyaw_d = float(det_yaw[j])
                            if np.isnan(dyaw_d):
                                continue
                            iou = _oriented_box_iou(cx_t, cy_t, float(tyaw), cx_d, cy_d, dyaw_d, car_length, car_width)
                            iou_term[i, j] = iou_cost_weight * (1.0 - iou)
                dists = dists + iou_term
            # 운동 방향 비용 추가(선택)
            if motion_cost_weight > 0.0:
                motion_term = np.zeros_like(dists)
                for i, trk in enumerate(S["tracks"]):
                    vx, vy = float(trk["x"][2]), float(trk["x"][3])
                    spd = float(np.hypot(vx, vy))
                    if spd >= min_speed_for_motion_cost:
                        vhat = np.array([vx, vy], dtype=float) / max(spd, 1e-6)
                        delta = dets - pred_xy[i]
                        along = delta @ vhat  # (n,)
                        perp = delta - (along[:, None] * vhat[None, :])
                        perp_err = np.linalg.norm(perp, axis=1)
                        motion_term[i, :] = motion_cost_weight * (perp_err + np.maximum(0.0, -along))
                dists = dists + motion_term
            # 게이팅: 임계 초과는 큰 비용
            cost = np.where(dists <= dist_threshold, dists, dist_threshold + 1e6)
        rows, cols = linear_sum_assignment(cost)
        matches = []
        unmatched_tracks = set(range(m))
        unmatched_dets = set(range(n))
        gate_thresh = mahalanobis_thresh if use_mahalanobis else dist_threshold
        for r, c in zip(rows, cols):
            # 게이트 이내만 유효 매칭으로 인정
            if cost[r, c] <= gate_thresh:
                matches.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)
        # Fallback: Mahalanobis로 한 개도 매칭 못하면 유클리드로 재시도
        if use_mahalanobis and len(matches) == 0 and m > 0 and n > 0:
            dists_fb = np.linalg.norm(pred_xy[:, None, :] - dets[None, :, :], axis=2)
            cost_fb = np.where(dists_fb <= dist_threshold, dists_fb, dist_threshold + 1e6)
            rows, cols = linear_sum_assignment(cost_fb)
            for r, c in zip(rows, cols):
                if cost_fb[r, c] <= dist_threshold:
                    matches.append((r, c))
                    if r in unmatched_tracks:
                        unmatched_tracks.remove(r)
                    if c in unmatched_dets:
                        unmatched_dets.remove(c)
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
        # 필요시 위치를 측정값으로 강제(측정 우선)
        if force_meas_position:
            trk["x"][0] = float(dets[j, 0])
            trk["x"][1] = float(dets[j, 1])
        # yaw 지수평활 (각도 래핑으로 경계(-pi,pi)에서 튐 방지)
        if det_yaw is not None and not np.isnan(det_yaw[j]):
            meas_yaw = float(det_yaw[j])
            if trk["yaw"] is None or np.isnan(trk["yaw"]):
                trk["yaw"] = meas_yaw
            else:
                prev = float(trk["yaw"])
                err = _wrap_angle(meas_yaw - prev)
                trk["yaw"] = _wrap_angle(prev + yaw_alpha * err)
        # 관리 필드
        trk["hits"] += 1
        trk["time_since_update"] = 0
        trk["ts"] = float(timestamp)
        # 마지막 측정값 기록(미검출 시 블렌딩/복원용)
        trk["_last_meas_xy"] = np.array([float(dets[j, 0]), float(dets[j, 1])], dtype=float)
        trk["_last_meas_ts"] = float(timestamp)
        # 정지 근처에서는 속도 과도/역행 억제 위해 감쇠
        spd = float(np.hypot(trk["x"][2], trk["x"][3]))
        if spd < stop_speed_thresh:
            trk["x"][2] *= stop_damping
            trk["x"][3] *= stop_damping

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
        # 마지막 측정값 기록
        new_trk["_last_meas_xy"] = np.array([float(cx), float(cy)], dtype=float)
        new_trk["_last_meas_ts"] = float(timestamp)
        S["tracks"].append(new_trk)
        S["next_id"] += 1

    # --- 미매칭 트랙 처리: Freeze 또는 예측 유지(+속도 감쇠/블렌딩/이동량 캡)
    if len(S["tracks"]) > 0:
        matched_track_indices = {i for i, _ in matches}
        for i, trk in enumerate(S["tracks"]):
            if i in matched_track_indices:
                continue
            tsu = int(trk.get("time_since_update", 0))
            if tsu <= 0:
                continue
            if freeze_on_miss or tsu >= int(freeze_after_misses):
                # 마지막 측정 위치로 고정 + 속도 0
                last_xy = trk.get("_last_meas_xy", None)
                if last_xy is not None and np.all(np.isfinite(last_xy)):
                    trk["x"][0] = float(last_xy[0])
                    trk["x"][1] = float(last_xy[1])
                trk["x"][2] = 0.0
                trk["x"][3] = 0.0
                if isinstance(trk.get("P", None), np.ndarray) and trk["P"].shape == (4, 4):
                    trk["P"][0, 0] = min(trk["P"][0, 0] + r_pos, 100.0)
                    trk["P"][1, 1] = min(trk["P"][1, 1] + r_pos, 100.0)
                    trk["P"][2, 2] = max(trk["P"][2, 2] * 0.5, 1e-3)
                    trk["P"][3, 3] = max(trk["P"][3, 3] * 0.5, 1e-3)
            else:
                # 예측 유지: 속도 감쇠 + 위치 블렌딩 + 이동량 캡
                trk["x"][2] *= float(miss_vel_damping)
                trk["x"][3] *= float(miss_vel_damping)
                pred_xy = np.array([float(trk["x"][0]), float(trk["x"][1])], dtype=float)
                last_xy = trk.get("_last_meas_xy", None)
                if last_xy is None or not np.all(np.isfinite(last_xy)):
                    last_xy = pred_xy.copy()
                # 블렌딩 (측정 쪽으로 강하게 끌어당김)
                a = float(np.clip(miss_blend_alpha, 0.0, 1.0))
                blended = (1.0 - a) * last_xy + a * pred_xy
                # 프레임당 최대 이동량 캡 (누적 미스 수에 비례하여 느슨화)
                cap = max(0.0, float(max_miss_disp_per_frame)) * max(1, tsu)
                delta = blended - last_xy
                d = float(np.hypot(delta[0], delta[1]))
                if d > cap and cap > 0.0:
                    blended = last_xy + (delta * (cap / (d + 1e-6)))
                trk["x"][0] = float(blended[0])
                trk["x"][1] = float(blended[1])

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