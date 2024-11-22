import cv2
import numpy as np
import os
import pandas as pd


def color_cast_index(img):
    # 转换为Lab颜色空间
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # 分离通道
    L, a, b = cv2.split(lab)
    # 计算a*和b*的平均值，中心点为128
    mean_a = np.mean(a) - 128
    mean_b = np.mean(b) - 128
    # 计算色彩偏移量（CCI）
    cci = np.sqrt(mean_a ** 2 + mean_b ** 2)
    return cci

def low_light_amount(img):
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    N_total = gray.size
    # 计算暗像素比例（DPP）
    threshold_dark = 50
    N_dark = np.sum(gray < threshold_dark)
    DPP = (N_dark / N_total) * 100
    # 计算平均亮度和标准差
    mean_L = np.mean(gray)
    std_L = np.std(gray)
    # 计算信息熵
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    hist_norm = hist_norm + 1e-7  # 避免log(0)
    entropy = -np.sum(hist_norm * np.log2(hist_norm))
    # 判断光线不足
    low_light_flag = False
    if DPP > 40 and mean_L < 80 and entropy < 7:
        low_light_flag = True
    return {
        '暗像素比例 (%)': DPP,
        '平均亮度': mean_L,
        '亮度标准差': std_L,
        '信息熵': entropy,
        '光线不足': '是' if low_light_flag else '否'
    }


def tenengrad(img):
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 计算Sobel梯度
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # 计算梯度幅值平方和
    FM = Gx ** 2 + Gy ** 2
    # 计算Tenengrad值（平均值）
    S = np.mean(FM)
    return S


def process_images(folder_path, output_excel):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图像：{filename}")
                continue

            print(f"正在处理：{filename} ...")
            # 计算色彩偏移量
            cci = color_cast_index(img)
            # 计算光线不足量
            low_light_metrics = low_light_amount(img)
            # 计算模糊量（Tenengrad值）
            blur = tenengrad(img)
            # 判断模糊程度
            # 设定模糊阈值（根据实际情况调整），这里假设阈值为600
            blur_threshold = 600
            blur_flag = '是' if blur < blur_threshold else '否'
            # 将结果添加到列表
            result = {
                '图片名': filename,
                '色彩偏移量': cci,
                '暗像素比例 (%)': low_light_metrics['暗像素比例 (%)'],
                '平均亮度': low_light_metrics['平均亮度'],
                '亮度标准差': low_light_metrics['亮度标准差'],
                '信息熵': low_light_metrics['信息熵'],
                '光线不足': low_light_metrics['光线不足'],
                '模糊量': blur,
                '过于模糊': blur_flag
            }
            results.append(result)
    # 保存到Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"结果已保存到 {output_excel}")

folder_path = 'resources/a/'  # 替换为您的图像文件夹路径
output_excel = 'result2.xlsx'  # 输出的xlsx文件名
process_images(folder_path, output_excel)
