import torch
import numpy as np
import cv2

def resize_align_multi_scale(img, scale=1.0, image_size=(256, 256)):
    # Resize ảnh theo tỉ lệ scale
    h, w, _ = img.shape
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Căn giữa ảnh vào khung vuông (256 x 256)
    pad_x = (image_size[0] - new_w) // 2
    pad_y = (image_size[1] - new_h) // 2
    canvas = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w, :] = resized_img

    # Chuyển sang Tensor [C, H, W], chuẩn hóa giá trị về [0, 1]
    img_tensor = torch.from_numpy(canvas).float().permute(2, 0, 1) / 255.0

    # Normalize giống ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor
