# scripts/myshow.py
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import torch

def get_torch_info():
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "devices": []
    }

    if info["cuda_available"]:
        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_MB": round(props.total_memory / (1024 ** 2), 2),
                "compute_capability": f"{props.major}.{props.minor}"
            })

    return info

def get_image_shape(image_array):
    print("Python: received image with shape", image_array.shape)
    # 保存图像进行验证
    cv2.imwrite("output_from_cpp.jpg", image_array)
    return image_array.shape

def show_image_basic():
    """
    基础图像显示函数
    
    参数:
        image_path: 图像文件路径(支持jpg/png/bmp等格式)
    """
    try:
        image_path = "1.jpg"
        # 读取图像（默认BGR格式）
        img = cv2.imread(image_path) 
        if img is None:
            raise FileNotFoundError(f"图像加载失败，请检查路径: {image_path}")
            
        # 显示图像 
        cv2.imshow('Image  Viewer', img)
        cv2.waitKey(0)   # 等待任意按键 
        cv2.destroyAllWindows() 
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
 


