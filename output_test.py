# import json
# import os
# import numpy as np
# import torch
# from src.solver.utils import output_to_smiles

# with open('/home/jovyan/data/OCR_SMILES/annotations/val.json', 'r') as file:
#         data = json.load(file)

# image_id_to_name = {}

# for image_data in data['images']:
#     image_id = image_data['id']
#     image_path = image_data['file_name']
#     image_name = os.path.basename(image_path)
#     image_id_to_name[image_id] = image_name

# res_smiles = []

# # load label and assigned idx
# unique_labels = json.load(open('/home/jovyan/data/OCR_SMILES/labels.json', 'r'))
# unique_labels['other'] = 0
# labels = list(unique_labels.keys())
# labels.insert(0, labels.pop())  # need "other" first in the list

# # idx to labels for inference
# bond_labels = [unique_labels[b] for b in ['-', '=', '#']]
# idx_to_labels = {v: k for k, v in unique_labels.items()}
# for l, b in zip(bond_labels, ['SINGLE', 'DOUBLE', 'TRIPLE']):
#     idx_to_labels[l] = b



# res = {
#     0:{
#         'labels':torch.randint(0, 11, (300,)),
#         'boxes':torch.rand((300, 4)) * 240,
#         'scores':torch.rand(300)
#     }
# }

# for key,value in res.items():
#     res_smiles.append(output_to_smiles(value,idx_to_labels,bond_labels))


# print(res_smiles)


import pandas as pd


data = pd.read_csv('/home/jovyan/rt-detr/output/output.csv')
print(data)