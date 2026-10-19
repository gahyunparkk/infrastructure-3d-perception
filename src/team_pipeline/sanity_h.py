# sanity_h.py
import numpy as np
from projection import pixel_to_world_plane, world_plane_to_pixel, rescale_pixels

# 1) 너의 픽셀→월드 H(1440x810 기준)를 그대로 넣어줘
H = np.array([
    [-0.00000000, -0.01978737, 14.61354790],
    [-0.00381594, -0.02218584, 17.54673110],
    [ 0.00000000,  0.00029981, -0.19998991]
], dtype=float)

# 2) CSV 좌표 해상도 / H가 계산된 해상도
input_image_size = (1440, 810)   # CSV 좌표가 1440x810이면 이렇게
homography_size  = (1440, 810)   # H를 만든 기준 해상도

# 3) rescale은 '함수'가 아니라 '배열을 리턴하는 람다'여야 함
if input_image_size != homography_size:
    rescale = lambda arr: rescale_pixels(arr, src=input_image_size, target=homography_size)
else:
    rescale = lambda arr: arr

# 4) 픽셀 몇 점을 찍어서 왕복 테스트
pix = np.array([[720., 405.],
                [820., 405.],
                [720., 505.]], dtype=float)

print("types:", type(pix), type(rescale))
w = pixel_to_world_plane(pix, H, rescale)  # pixel -> world
pix_back = world_plane_to_pixel(w, H, None)
# world_plane_to_pixel()은 rescale이 없으니, 되돌릴 때 직접 반대 스케일 적용
if input_image_size != homography_size:
    pix_back = rescale_pixels(pix_back, src=homography_size, target=input_image_size)

print("pix:\n", pix)
print("world:\n", w)
print("pix_back:\n", pix_back)  # pix와 거의 같아야 정상
