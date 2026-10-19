from typing import Tuple
import numpy as np

def get_heading(p1: np.ndarray, p2: np.ndarray, degrees: bool=False) -> np.ndarray:
    p1 = p1.reshape((1, 2)) if p1.ndim == 1 else p1
    p2 = p2.reshape((1, 2)) if p2.ndim == 1 else p2
    dir = p1 - p2
    radians = np.arctan2(dir[:,1], dir[:,0])
    return np.degrees(radians) if degrees else radians

def mirror_points(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    assert points.shape == centers.shape
    return centers - (points - centers)

def complete_parallelograms(corners1: np.ndarray, corners2: np.ndarray, centers: np.ndarray, include_centers: bool = True) -> np.ndarray:
    corners3 = mirror_points(corners1, centers)
    corners4 = mirror_points(corners2, centers)
    corners = np.array([corners1, corners2, corners4, corners3])  # e.g. fl, fr, bl, br
    corners = np.vstack([corners, centers.reshape(1, -1, 2)]) if include_centers else corners
    return np.moveaxis(corners, 0, 1)  # (N, 4, 2)  or (N, 5, 2)

def span_bbox(center: np.ndarray, heading: float, dimensions: Tuple[float, float, float]) -> np.ndarray:
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    if center.shape[0] == 2:
        center = np.hstack([center, [dimensions[2] / 2]])
    
    l, w, h = dimensions[0]/2, dimensions[1]/2, dimensions[2]/2
    
    corners_local = np.array([
        [ l,  w,  h],  # ftr
        [ l,  w, -h],  # fbr
        [ l, -w,  h],  # ftl
        [ l, -w, -h],  # fbl
        [-l,  w,  h],  # btr
        [-l,  w, -h],  # bbr
        [-l, -w,  h],  # btl
        [-l, -w, -h],  # bbl
    ])
    
    corners = (R @ corners_local.T).T + center
    return corners