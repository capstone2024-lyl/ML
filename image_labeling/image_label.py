import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
import cv2
import os
from collections import defaultdict

# COCO 데이터셋 경로
dataDir = './image_labeling/dataset/train2017/train2017'
annFile = './image_labeling/dataset/annotations_trainval2017/annotations/instances_train2017.json'

current_dir = os.path.abspath(os.getcwd())
print("Current Directory:", current_dir)

# COCO API 초기화
coco = COCO(annFile)

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

# 역 맵핑 생성
inverse_category_mapping = {}
for category, items in category_mapping.items():
    for item in items:
        inverse_category_mapping[item] = category

# 이미지 전처리 함수
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found at path: {img_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

# 데이터셋 생성 함수
def load_coco_dataset(coco, dataDir, img_size=IMG_SIZE, num_samples=5000):
    img_ids = coco.getImgIds()
    X, y = [], []

    for img_id in img_ids[:num_samples]:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(dataDir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        try:
            img = preprocess_image(img_path)
        except FileNotFoundError as e:
            print(e)
            continue
        
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        
        labels = np.zeros(8)  # 자연, 인물, 동물, 교통 수단, 가전 제품, 음식, 가구, 기타
        for ann in anns:
            cat_name = coco.loadCats(ann['category_id'])[0]['name']
            if cat_name in inverse_category_mapping:
                category = inverse_category_mapping[cat_name]
                if category == '자연':
                    labels[0] = 1
                elif category == '인물':
                    labels[1] = 1
                elif category == '동물':
                    labels[2] = 1
                elif category == '교통 수단':
                    labels[3] = 1
                elif category == '가전 제품':
                    labels[4] = 1
                elif category == '음식':
                    labels[5] = 1
                elif category == '가구':
                    labels[6] = 1
                elif category == '기타':
                    labels[7] = 1
        
        X.append(img)
        y.append(labels)
    
    return np.array(X), np.array(y)

print("Loading COCO dataset...")
X, y = load_coco_dataset(coco, dataDir)
print(f"COCO dataset loaded with {len(X)} samples.")

# MobileNetV2 모델 정의 및 학습
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# MobileNetV2 모델 불러오기 (사전 학습된 가중치 사용)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# 커스텀 출력 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(8, activation='sigmoid')(x)  # 자연, 인물, 동물, 교통 수단, 가전 제품, 음식, 가구, 기타

# 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 일부 레이어를 고정하여 학습하지 않도록 설정
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

print("Starting model training...")
# 모델 학습 (배치 크기를 줄여 메모리 사용량을 낮춥니다)
model.fit(X, y, epochs=30, batch_size=16, validation_split=0.2)
print("Model training completed.")

model.save('./image_labeling/model/mobile_net.h5')
