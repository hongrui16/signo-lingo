import numpy as np
from scipy.spatial.transform import Rotation as R

def rotate_keypoints_sequence(keypoints, angle_range=(-30, 30)):
    ## if random chance is less than 0.6, then return the original keypoints
    if np.random.rand() < 0.7:
        return keypoints
    # 生成随机旋转角度
    angles = np.random.uniform(*angle_range, size=3)
    # 创建旋转对象
    rotation = R.from_euler('xyz', angles, degrees=True)
    # 旋转关键点
    rotated_keypoints = rotation.apply(keypoints.reshape(-1, 3)).reshape(keypoints.shape)
    return rotated_keypoints


def scale_keypoints_sequence(keypoints, scale_range=(0.8, 1.2)):
    ## if random chance is less than 0.6, then return the original keypoints
    if np.random.rand() < 0.7:
        return keypoints
    # 生成随机缩放因子
    scale_factor = np.random.uniform(*scale_range)
    # 缩放关键点
    scaled_keypoints = keypoints * scale_factor
    return scaled_keypoints

def translate_keypoints_sequence(keypoints, translation_range=(-0.1, 0.1)):
    ## if random chance is less than 0.6, then return the original keypoints
    if np.random.rand() < 0.7:
        return keypoints
    # 生成随机平移向量
    translation_vector = np.random.uniform(*translation_range, size=3)
    # 平移关键点
    translated_keypoints = keypoints + translation_vector
    return translated_keypoints

def add_noise_to_keypoints_sequence(keypoints, noise_level=0.01):
    ## if random chance is less than 0.6, then return the original keypoints
    if np.random.rand() < 0.7:
        return keypoints
    # 添加高斯噪声
    noise = np.random.normal(scale=noise_level, size=keypoints.shape)
    noisy_keypoints = keypoints + noise
    return noisy_keypoints


if __name__ == '__main__':
    # 测试代码
    keypoints = np.random.rand(3, 4, 3)  # 假设有10帧，每帧15个关键点
    rotated_keypoints = rotate_keypoints_sequence(keypoints)
    scaled_keypoints = scale_keypoints_sequence(keypoints)
    translated_keypoints = translate_keypoints_sequence(keypoints)
    noisy_keypoints = add_noise_to_keypoints_sequence(keypoints)
    print(rotated_keypoints)
    print(scaled_keypoints)
    print(translated_keypoints)
    print(noisy_keypoints)
