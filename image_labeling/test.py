import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt

# MobileNetV2 입력 크기
IMG_SIZE = 224

# 카테고리 맵핑
category_mapping = {
    '자연': ['tree', 'grass', 'river', 'mountain', 'sea', 'cloud', 'forest', 'plant'],
    '인물': ['person'],
    '동물': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
    '교통 수단': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'],
    '가전 제품': ['microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone'],
    '음식': ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
    '가구': ['chair', 'couch', 'bed', 'dining table', 'toilet'],
    '일상': ['traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'backpack', 'umbrella', 'handbag', 'tie', 
             'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
             'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
             'bowl', 'potted plant', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
}

# COCO 카테고리 이름 리스트 (카테고리 맵핑에서 고유한 이름 추출)
category_names = list(set([item for sublist in category_mapping.values() for item in sublist]))

# 모델 로드
model_path = './image_labeling/model/best_mobile_net.h5'  # 모델 파일 경로 수정
model = tf.keras.models.load_model(model_path)

# 이미지 전처리 함수
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found at path: {img_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

# 테스트 이미지에 대한 예측 함수
def predict_and_display(model, img_path):
    img = preprocess_image(img_path)
    img_batch = np.expand_dims(img, axis=0)
    preds = model.predict(img_batch)
    print(preds)
    pred_classes = np.where(preds[0] > 0.5)[0]
    pred_categories = {category: [] for category in category_mapping.keys()}
    
    for pred_class in pred_classes:
        cat_name = category_names[pred_class]
        for category, items in category_mapping.items():
            if cat_name in items:
                pred_categories[category].append(cat_name)
                break
    
    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted Categories: {pred_categories}")
    plt.axis('off')
    plt.show()

# 테스트 이미지 경로
test_img_path = './image_labeling/dataset/train2017/train2017/000000000034.jpg'

# 예측 및 시각화
predict_and_display(model, test_img_path)
