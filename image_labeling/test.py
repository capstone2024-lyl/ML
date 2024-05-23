import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
import cv2
import os
import matplotlib.pyplot as plt

# COCO 데이터셋 경로
dataDir = './dataset/train2017/train2017'
annFile = './dataset/annotations_trainval2017/annotations/instances_train2017.json'

# COCO API 초기화
coco = COCO(annFile)

# MobileNetV2 입력 크기
IMG_SIZE = 224

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
            print(f"Image file not found: {img_path}")
            continue
        try:
            img = preprocess_image(img_path)
        except FileNotFoundError as e:
            print(e)
            continue
        
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        
        labels = np.zeros(80)
        for ann in anns:
            cat_id = ann['category_id'] - 1
            if cat_id < 80:  # Ensure the category ID is within bounds
                labels[cat_id] = 1  # 카테고리 ID를 One-hot encoding
        
        X.append(img)
        y.append(labels)
    
    return np.array(X), np.array(y)

# COCO 데이터셋 로드 및 전처리
X, y = load_coco_dataset(coco, dataDir)

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
predictions = Dense(80, activation='sigmoid')(x)

# 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 일부 레이어를 고정하여 학습하지 않도록 설정
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

# 모델 저장
model.save('./model/mobilenet_coco.h5')

# 테스트 이미지에 대한 예측
def predict_and_display(model, img_path):
    img = preprocess_image(img_path)
    img_batch = np.expand_dims(img, axis=0)
    preds = model.predict(img_batch)
    pred_classes = np.where(preds[0] > 0.5)[0]
    
    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted Categories: {pred_classes}")
    plt.show()

# 테스트 이미지 경로
test_img_path = './dataset/train2017/train2017/000000000009.jpg'

# 예측 및 시각화
predict_and_display(model, test_img_path)