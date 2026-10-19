from typing import Callable, Optional
import numpy as np

from calibration import H_CAM_EXAMPLE1

def rescale_pixels(pixels: np.ndarray, src=(1920,1080), target=(1920, 1080), inv=False) -> np.ndarray:
    pixels = pixels.reshape(1, 2) if pixels.ndim == 1 else pixels
    src, target = (target, src) if inv else (src, target)
    scale_x = target[0] / src[0]
    scale_y = target[1] / src[1]
    return np.dstack([pixels[:,0] * scale_x, pixels[:,1] * scale_y]).reshape(-1, 2)

# (N, 3) -> (N, 3)
def norm_homogeneous2(x: np.ndarray) -> np.ndarray:
    return x / x[:, 2:3] 

# (N, 2) -> (N, 3)
def homogeneous(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def pixel_to_world_plane(x: np.ndarray, H: np.ndarray, rescale: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    x = x.reshape(1, 2) if x.ndim == 1 else x
    x = rescale(x) if rescale is not None else x
    hom = homogeneous(x)
    res = norm_homogeneous2(hom @ H.T)
    return res[:, :2]

def world_plane_to_pixel(x: np.ndarray, H: np.ndarray, rescale: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    H_inv = np.linalg.inv(H)
    hom = homogeneous(x)
    res = norm_homogeneous2(hom @ H_inv.T)
    res = res[:, :2]
    return rescale(res) if rescale is not None else res

def project_to_image(x: np.ndarray, P: np.ndarray, rescale: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    assert x.shape[1] == 3
    assert P.shape == (3, 4)
    
    x = np.hstack([x, np.ones((x.shape[0], 1))])
    x_t = x @ P.T
    x_t = x_t[:, :2] / x_t[:, 2:3]
    
    return rescale(x_t) if rescale is not None else x_t
    

def transform(x: np.ndarray, T: np.ndarray) -> np.ndarray:
    x = np.hstack([x, np.zeros((x.shape[0], 1))]) if x.shape[1] == 2 else x
    
    assert x.shape[1] == 3
    assert T.shape == (4, 4)
    
    x = np.hstack([x, np.ones((x.shape[0], 1))])
    x_t = x @ T.T
    return x_t[:,:3] / x[:,3:4]

#if __name__ == '__main__':
    # Example

    # pixel_0 = np.array([1290, 550])  # in 1440 x 810 image
    # print(pixel_0)

    # world_0 = pixel_to_world_plane(rescale_pixels(pixel_0), H_CAM_EXAMPLE1)
    # print(world_0)

    # reverse_0 = rescale_pixels(world_plane_to_pixel(world_0, H_CAM_EXAMPLE1), inv=True)
    # print(reverse_0) 
