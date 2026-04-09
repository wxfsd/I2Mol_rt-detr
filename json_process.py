# import json

# # 读取包含重复项的 JSON 文件
# with open('/home/jovyan/data/OCR_SMILES/annotations_train.json', 'r') as file:
#     data = json.load(file)

# print(len(data))

# # 提取所有的 file_name
# file_names = [item['file_name'] for item in data]

# # 保留非重复的 file_name
# unique_file_names = list(set(file_names))

# print(len(unique_file_names))

# # 创建新的 JSON 数据只包含非重复的 file_name
# unique_data = []
# seen_file_names = set()
# for item in data:
#     if item['file_name'] not in seen_file_names:
#         unique_data.append(item)
#         seen_file_names.add(item['file_name'])

# print(len(unique_data))

# # 将新数据写入新的 JSON 文件
# with open('/home/jovyan/data/OCR_SMILES/unique_data.json', 'w') as file:
#     json.dump(unique_data, file, indent=4)

# print("Non-duplicate file_names extracted and saved in unique_data.json")


# import json

# # 读取包含重复项的 JSON 文件
# with open('/home/jovyan/data/OCR_SMILES/annotations_val.json', 'r') as file:
#     data = json.load(file)

# category_list = []

# for i in data:
#     for j in i['annotations']:
#         if j['category_id'] not in category_list:
#             category_list.append(j['category_id'])


# print(category_list)


import pickle

with open('/home/jovyan/data/OCR_SMILES/annotations_val.pkl', 'rb') as file:
    data = pickle.load(file)


category_list = []

for i in data:
    for j in i['annotations']:
        if j['category_id'] not in category_list:
            category_list.append(j['category_id'])


print(category_list)



