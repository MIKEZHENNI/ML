
import numpy as np
from PIL import Image

# 打开图片文件
img = Image.open("./img/test/000001.jpg")

# 将图片转换为NumPy数组
img_array = np.array(img)

# 保存NumPy数组为.npy文件
# np.save("image.npy", img_array)

# # 加载.npy文件
# loaded_img_array = np.load("image.npy")

# 获取并打印.npy文件的形状
shape = img_array.shape
print(f"图片的形状是：{shape}")