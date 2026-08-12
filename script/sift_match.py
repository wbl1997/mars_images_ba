import cv2
import numpy as np

# 读取两张tif影像
img1 = cv2.imread('/media/wbl/Elements/paper_experiments/Mars/new/ESP_069731_2055/downsample/0/mosaic_ds4.tif', cv2.IMREAD_GRAYSCALE)  # 第一张影像（可替换为tif路径）
img2 = cv2.imread('/media/wbl/Elements/paper_experiments/Mars/new/ESP_075559_2055/downsample/0/mosaic_ds4.tif', cv2.IMREAD_GRAYSCALE)  # 第二张影像

if img1 is None or img2 is None:
    print('影像读取失败，请检查路径！')
    exit(1)

# 创建SIFT对象
sift = cv2.SIFT_create()

# 检测特征点和描述子
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 使用FLANN进行特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

# 用cv2绘制并保存匹配结果

# 拼接两张图像
h1, w1 = img1.shape
h2, w2 = img2.shape
out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
out_img[:h1, :w1, 0] = img1
out_img[:h2, w1:w1 + w2, 0] = img2
out_img[:h1, :w1, 1] = img1
out_img[:h2, w1:w1 + w2, 1] = img2
out_img[:h1, :w1, 2] = img1
out_img[:h2, w1:w1 + w2, 2] = img2

# 绘制关键点
for kp in kp1:
    x, y = map(int, kp.pt)
    cv2.circle(out_img, (x, y), 2, (0, 255, 0), -1)
for kp in kp2:
    x, y = map(int, kp.pt)
    cv2.circle(out_img, (int(x + w1), int(y)), 2, (0, 255, 0), -1)

# 绘制匹配线（线宽为3）
for m in good:
    pt1 = tuple(map(int, kp1[m.queryIdx].pt))
    pt2 = tuple(map(int, kp2[m.trainIdx].pt))
    pt2 = (int(pt2[0] + w1), int(pt2[1]))
    cv2.line(out_img, pt1, pt2, (255, 0, 0), 3)

cv2.imwrite('/home/wbl/code/MARS/Mars_new/out/sift_match_result.png', out_img)
print(f'SIFT匹配结果: {len(good)} 对，已保存到 /home/wbl/code/MARS/Mars_new/out/sift_match_result.png')

# --- SIFT自动配准与镶嵌 ---
if len(good) >= 4:
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is not None:
        # 计算变换后img2的四个角坐标
        h2, w2 = img2.shape
        corners_img2 = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
        warped_corners = cv2.perspectiveTransform(corners_img2, H)
        all_corners = np.vstack((warped_corners, np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)))
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        tx, ty = -xmin, -ymin
        out_w, out_h = xmax - xmin, ymax - ymin
        # 变换img2到新坐标系
        H_translate = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        img2_warp = cv2.warpPerspective(img2, H_translate @ H, (out_w, out_h))
        # 将img1放到新坐标系
        mosaic = img2_warp.copy()
        mosaic[ty:ty+h1, tx:tx+w1][img1>0] = img1[img1>0]
        cv2.imwrite('/home/wbl/code/MARS/Mars_new/out/sift_mosaic.png', mosaic)
        print('自动配准镶嵌结果已保存到 /home/wbl/code/MARS/Mars_new/out/sift_mosaic.png')
    else:
        print('单应性矩阵计算失败，无法自动镶嵌。')
else:
    print('匹配点不足，无法自动镶嵌。')

# --- 稠密视差图计算（无需外参）---
# 建议两张影像已配准且为立体对
window_size = 5
min_disp = 0
num_disp = 64  # 必须是16的倍数
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 1 * window_size ** 2,
    P2=32 * 1 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)
# 保证输入影像尺寸和类型一致
h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
img1_rect = cv2.resize(img1, (w, h)).astype(np.uint8)
img2_rect = cv2.resize(img2, (w, h)).astype(np.uint8)
disparity = stereo.compute(img1_rect, img2_rect).astype(np.float32) / 16.0
cv2.imwrite('/home/wbl/code/MARS/Mars_new/out/sift_disparity.png', ((disparity - min_disp) / num_disp * 255).astype(np.uint8))

# 归一化视差图
disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disp_norm = disp_norm.astype(np.uint8)
# 渲染为伪彩色
disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
cv2.imwrite('/home/wbl/code/MARS/Mars_new/out/sift_disparity_color.png', disp_color)
print('稠密视差图已保存到 /home/wbl/code/MARS/Mars_new/out/sift_disparity.png')
print('伪彩色视差图已保存到 /home/wbl/code/MARS/Mars_new/out/sift_disparity_color.png')
