from pycocotools.coco import COCO
import os
from collections import defaultdict

# COCO 데이터셋 경로 설정
current_dir = os.path.abspath(os.getcwd())
dataDir = './image_labeling/dataset/train2017/train2017'
annFile = './image_labeling/dataset/annotations_trainval2017/annotations/instances_train2017.json'

# COCO API 초기화
coco = COCO(annFile)

# 카테고리 ID와 이름 추출
categories = coco.loadCats(coco.getCatIds())
category_names = [cat['name'] for cat in categories]
category_ids = [cat['id'] for cat in categories]

# 카테고리 ID와 이름 출력
for cat_id, cat_name in zip(category_ids, category_names):
    print(f"ID: {cat_id}, Name: {cat_name}")

# 카테고리 이름 리스트 출력
print("Category Names:", category_names)


# COCO API 초기화
coco = COCO(annFile)

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

# 이미지 분류 및 카운트 함수
def classify_and_count_images(coco, dataDir, num_samples=5000):
    img_ids = coco.getImgIds()
    label_counts = defaultdict(int)
    
    for img_id in img_ids[:num_samples]:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(dataDir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        
        labels = set()
        for ann in anns:
            cat_name = coco.loadCats(ann['category_id'])[0]['name']
            if cat_name in inverse_category_mapping:
                labels.add(inverse_category_mapping[cat_name])
        
        for label in labels:
            label_counts[label] += 1
    
    return label_counts

print("Classifying and counting images...")
label_counts = classify_and_count_images(coco, dataDir)
print("Label counts:", label_counts)