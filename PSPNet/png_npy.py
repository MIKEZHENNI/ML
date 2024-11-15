import os
from PIL import Image
import numpy as np

# 定义源文件夹和目标文件夹路径
source_folder_path = 'A_duibi/c_predict'  # 替换为你的源文件夹路径
target_folder_path = 'A_duibi/c'  # 替换为你的目标文件夹路径

# 确保目标文件夹存在
if not os.path.exists(target_folder_path):
    os.makedirs(target_folder_path)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder_path):
    if filename.endswith('.png'):  # 筛选出.png文件
        # 构造源文件的完整路径
        source_file_path = os.path.join(source_folder_path, filename)

        # 使用PIL打开图像
        image = Image.open(source_file_path)

        # 将图像转换为NumPy数组
        image_array = np.array(image)

        # 构造新的文件名，例如：ground_truth_000001.npy
        new_filename = os.path.splitext(filename)[0]  # 去掉原文件名的扩展名
        #new_filename = 'c_prediction_' + new_filename + '.npy'
        #new_filename = 'ground_truth_' + new_filename + '.npy'
        new_filename = 'prediction_' + new_filename + '.npy'
        #prediction_000001.npy
        # 构造目标文件夹中的.npy文件的完整路径
        target_file_path = os.path.join(target_folder_path, new_filename)

        # 保存NumPy数组为.npy文件
        np.save(target_file_path, image_array)
        print(f'Saved {target_file_path}')