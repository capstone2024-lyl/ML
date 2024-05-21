from pycocotools.coco import COCO
import os

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