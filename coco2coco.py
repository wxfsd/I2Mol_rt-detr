import json
import os

dataset = 'acs'

with open(f'/home/jovyan/data/real_processed/merge_with_chiral+charge_resample/annotations_val.json', 'r') as f:
    data = json.load(f)

with open(f'/home/jovyan/data/real_processed/merge_with_chiral+charge_resample/labels.json', 'r') as file:
    categories_data = json.load(file)

save_path = f'/home/jovyan/data/real_processed/merge_with_chiral+charge_resample/annotations'
if not os.path.exists(save_path):
    os.makedirs(save_path)


coco_data = {
    "info": [],
    "licenses": [],
    "categories": [],
    "images": [],
    "annotations": []
}


# category = {
#     'id': 0,
#     'name': "other",
#     'supercategory': ''
# }
# coco_data["categories"].append(category)

for category_name, category_id in categories_data.items():
    category = {
        'id': category_id,
        'name': category_name,
        'supercategory': ''
    }
    coco_data["categories"].append(category)


def calculate_bbox_area_coco(bbox):
    x_min, y_min, width, height = bbox
    area = width * height
    return area

annotation_id = 0
image_id = 0

for item in data:
    image_data = {
        "file_name": item["file_name"],
        "height": item["height"],
        "width": item["width"],
        "id": image_id
    }
    coco_data["images"].append(image_data)

    

    if "annotations" in item:
        for annotation in item["annotations"]:
            area = calculate_bbox_area_coco(annotation["bbox"])
            # if annotation["category_id"] == 0:
            #     print('+++++++++++++++++++++++++++++++')
            annotation_data = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": annotation["category_id"],
                "bbox": annotation["bbox"],
                "iscrowd": 0,
                "area":area,
                "ignore":0
            }
            coco_data["annotations"].append(annotation_data)
            annotation_id += 1

    image_id += 1
    print('data {}  conversion complete '.format(image_id))


with open(f'{save_path}/val.json', 'w') as f:
    json.dump(coco_data, f)

print("转换结束")

