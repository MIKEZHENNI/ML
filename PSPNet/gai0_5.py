
import os
import shutil

# 指定源文件夹和目标文件夹的路径
source_folder_path = 'test_data/VOC2007/SegmentationClass'
target_folder_path = 'xin'

# 如果目标文件夹不存在，则创建它
if not os.path.exists(target_folder_path):
    os.makedirs(target_folder_path)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder_path):
    # 检查文件是否以.jpg结尾
    if filename.endswith(".png"):
        # 提取文件名中的数字部分
        file_number = filename[:-4]  # 去掉.jpg后缀
        # 构造新的文件名
        new_filename = f"500{file_number[3:]}.png"
        # 构造完整的文件路径
        old_file = os.path.join(source_folder_path, filename)
        new_file = os.path.join(target_folder_path, new_filename)
        # 复制并重命名文件到目标文件夹
        shutil.copy2(old_file, new_file)
        print(f"Copied and renamed '{filename}' to '{new_filename}' in '{target_folder_path}'")