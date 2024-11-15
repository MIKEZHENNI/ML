from PIL import Image
import os

# 源文件夹路径
source_folder = 'VOCdevkit/VOC2007/JPEGImages'
# 目标文件夹路径
target_folder = 'VOCdevkit/VOC2007/SegmentationClass'

# 如果目标文件夹不存在，则创建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith(".jpg"):  # 检查文件扩展名是否为.jpg
        # 构造完整的文件路径
        file_path = os.path.join(source_folder, filename)
        # 打开图片
        with Image.open(file_path) as img:
            # 构造新的文件名，将.jpg替换为.png
            new_filename = os.path.splitext(filename)[0] + '.png'
            # 构造目标文件的完整路径
            target_path = os.path.join(target_folder, new_filename)
            # 保存为PNG格式
            img.save(target_path, 'PNG')
            print(f'Converted {filename} to {new_filename}')