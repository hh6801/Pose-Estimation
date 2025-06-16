import cv2
import numpy as np
from ultralytics import YOLO
import torch
from mmcv import Config
from mmpose.models import build_posenet
from mmcv.runner import load_checkpoint
from mmpose.datasets import DatasetInfo
from models import *

def draw_pose_and_skeleton(image, keypoints, skeleton, color):
    for x, y in keypoints:
        if x >= 0 and y >= 0 and x < image.shape[1] and y < image.shape[0]:
            cv2.circle(image, (x, y), 3, color, -1)
    for i, j in skeleton:
        if i < len(keypoints) and j < len(keypoints):
            x1, y1 = keypoints[i]
            x2, y2 = keypoints[j]
            if all(0 <= v < max(image.shape[:2]) for v in [x1, y1, x2, y2]):
                cv2.line(image, (x1, y1), (x2, y2), color, 2)
def letterbox(img, size=256):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (nw, nh))
    pad_top = (size - nh) // 2
    pad_bottom = size - nh - pad_top
    pad_left = (size - nw) // 2
    pad_right = size - nw - pad_left
    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return img_padded, scale, pad_left, pad_top

# Đường dẫn file
IMG_PATH = '/Users/nhh6801/Documents/CDNC1-2-3/CD3/Models/pct/PCT/c.jpg'
YOLO_MODEL_PATH = '/Users/nhh6801/Documents/CDNC1-2-3/CD3/Models/pct/PCT/yolo11l.pt'
CONFIG_PATH = '/Users/nhh6801/Documents/CDNC1-2-3/CD3/Models/pct/PCT/configs/pct_base_classifier.py'
CHECKPOINT_PATH = '/Users/nhh6801/Documents/CDNC1-2-3/CD3/Models/pct/PCT.pth'

# Thiết bị
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load ảnh
image = cv2.imread(IMG_PATH)
orig_h, orig_w = image.shape[:2]

# Dùng YOLO để detect người
det_model = YOLO(YOLO_MODEL_PATH)
yolo_results = det_model(image,stream=True)  
pose_results = []
# for result in yolo_results:
#     boxes = result.boxes

#     if boxes is not None:
#         xyxy = boxes.xyxy.cpu().numpy()
#         conf = boxes.conf.cpu().numpy()
#         cls = boxes.cls.cpu().numpy()

#         for i in range(len(xyxy)):
#             print(int(cls[1]))
#             if int(cls[i]) == 0 and conf[i]>=0.5:  # class 0 là 'persons' trong COCO
#                 x1, y1, x2, y2 = map(int, xyxy[i])
#                 label = f"Person: {conf[i]:.2f}"

#                 # Vẽ hộp
#                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 # Ghi nhãn
#                 cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                             0.5, (0, 255, 0), 2)

# # Hiển thị ảnh kết quả
# cv2.imshow("Detected People", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Load mô hình PCT
config = Config.fromfile(CONFIG_PATH)
model = build_posenet(config.model)
_ = load_checkpoint(model, CHECKPOINT_PATH, map_location=DEVICE)
model.to(DEVICE)
model.eval()

# Lấy info dataset nếu có
dataset_info = config.get('dataset_info', None)
if dataset_info is not None:
    dataset_info = DatasetInfo(dataset_info)

# Resize image để vẽ box
resized_image = cv2.resize(image, (256, 256))
scale_x = 256 / orig_w
scale_y = 256 / orig_h

# Xử lý từng người
for result in yolo_results:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        continue

    print(f"🔍 Tổng số box: {len(boxes)}")

    for box, score, cls_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
        if int(cls_id.item()) != 0 or score.item() < 0.7:
            continue

        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        w, h = x2 - x1, y2 - y1
        person_img = image[y1:y2, x1:x2]

        if person_img.size == 0:
            print("⚠️ Bỏ qua bbox rỗng")
            continue

        
        input_img, scale, pad_left, pad_top = letterbox(person_img)

        cv2.imshow("Detected People", input_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        input_img = input_img / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_img = (input_img - mean) / std
        input_img = input_img.transpose(2, 0, 1)
        img_tensor = torch.tensor(input_img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        print(f"➡️ Tensor input shape: {img_tensor.shape}")

        with torch.no_grad():
            feat = model.backbone(img_tensor)
            feat = feat[0]  # Lấy output cuối cùng từ backbone: [1, 1024, 8, 8]
            print(f"✅ Backbone output shape: {feat.shape}")

            # ⬇️ Truyền qua conv_trans → [1, 256, 8, 8]
            cls_feat = model.keypoint_head.conv_trans(feat)
            print(f"🔹 After conv_trans shape: {cls_feat.shape}")

            # ⬇️ Flatten toàn bộ → [1, 256*8*8] = [1, 16384]
            cls_feat = cls_feat.view(cls_feat.size(0), -1)
            print(f"🔹 Flattened cls_feat shape: {cls_feat.shape}")

            # ⬇️ Truyền vào FCBlock mixer_trans → [1, 2176]
            cls_feat = model.keypoint_head.mixer_trans(cls_feat)
            print(f"🔹 After mixer_trans shape: {cls_feat.shape}")

            # ⬇️ Reshape về [B, token_num, hidden_dim] = [1, 34, 64]
            cls_feat = cls_feat.view(cls_feat.size(0),
                                    model.keypoint_head.token_num,
                                    model.keypoint_head.hidden_dim)
            print(f"🔹 Reshaped for mixer_head: {cls_feat.shape}")

            # ⬇️ Truyền qua các lớp mixer_head
            for mixer_layer in model.keypoint_head.mixer_head:
                cls_feat = mixer_layer(cls_feat)

            # ⬇️ Normal hóa
            cls_feat = model.keypoint_head.mixer_norm_layer(cls_feat)

            # ⬇️ Dự đoán logits: [1, 34, 2048]
            cls_logits = model.keypoint_head.cls_pred_layer(cls_feat)
            print(f"✅ cls_logits shape: {cls_logits.shape}")


            # Dự đoán keypoints
            # Đầu vào backbone đã xong:
            with torch.no_grad():
                pose, _ = model.keypoint_head([feat], [feat], joints=None, train=False)
            print(f"✅ Pose output shape: {pose.shape}")
            print(f"🔹 Pose example:\n{pose[0]}")

            # 🎯 BẮT ĐẦU THÊM Ở ĐÂY
            keypoints = pose.squeeze(0).cpu().numpy()
            scaled_keypoints = []

            for kp in keypoints:
                x, y = kp
                x = (x - pad_left) / scale
                y = (y - pad_top) / scale
                x_orig = x + x1
                y_orig = y + y1
                scaled_keypoints.append((int(x_orig), int(y_orig)))


            pose_results.append(scaled_keypoints)
            # 🎯 KẾT THÚC THÊM



skeleton = [
    (5, 7), (7, 9),        # Left arm
    (6, 8), (8, 10),       # Right arm
    (5, 6),                # Shoulders
    (5, 11), (6, 12),      # Torso
    (11, 13), (13, 15),    # Left leg
    (12, 14), (14, 16),    # Right leg
    (11, 12),              # Hips
    (0, 1), (0, 2),        # Nose to eyes
    (1, 3), (2, 4)         # Eyes to ears
]
colors = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 0, 128)   # purple
]

# Vẽ keypoints và skeleton lên ảnh
for idx, keypoints in enumerate(pose_results):
    color = colors[idx % len(colors)]  # chọn màu khác nhau
    draw_pose_and_skeleton(image, keypoints, skeleton, color)



cv2.imwrite("pose_result.jpg", image)
