# # 删除重名的和mol=None的SMILES
# import pandas as pd
# from rdkit import Chem

# data = pd.read_csv("/home/jovyan/data/real/acs.csv")
# data = data.drop_duplicates(subset='SMILES')

# data_unsuccess = pd.DataFrame({'file_name': [],'SMILES':[]})
# data_new = pd.DataFrame({'file_name': [],'SMILES':[]})
# list = []
# for index, row in data.iterrows():
#     file_name = row["file_path"]
#     smiles = row["SMILES"]
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         new_row = {'file_name':file_name,
#                     'SMILES':smiles}
#         data_unsuccess = data_unsuccess._append(new_row, ignore_index=True)

#     if mol is not None:
#         new_row = {'file_name':file_name,
#                     'SMILES':smiles}
#         data_new = data_new._append(new_row, ignore_index=True)
    

# data_new.to_csv('acs_new.csv', index=False)
# data_unsuccess.to_csv('acs_error.csv', index=False)

# import rdkit
# from rdkit.Chem import Draw
# from xml.dom import minidom
# import os
# import re
# import cv2
# import numpy as np
# from PIL import Image

# image = Image.open('/home/jovyan/rt-detr/output/test.png').convert('RGB')
# image = np.array(image)
# predict_boxes = [[109.99147058823525, 158.3529411764706, 124.99147058823525, 173.3529411764706], [105.46299201529466, 136.35410059604138, 120.46299201529466, 151.35410059604138], [97.59147058823532, 115.33235294117648, 112.59147058823532, 130.33235294117648]]
# for bbox in predict_boxes:
#     print(int(bbox[0]), int(bbox[1]),int(bbox[2]), int(bbox[3]))
#     cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,255), )
# cv2.imwrite("image.jpg", image)




# import json


# with open(f'/home/jovyan/data/real_processed/new_aug_data/annotations_val.json', 'r') as f:
#     data = json.load(f)
# count = 0
# list_ = [0]*24

# for d in data:
#     for ann in d['annotations']:
#         list_[ann['category_id']] += 1

# print(list_)


# import time
# from tqdm import tqdm
# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import Draw, AllChem

# df = pd.read_csv('/home/jovyan/rt-detr/aug/new_data.csv')
# data = df.drop_duplicates(subset='SMILES')

# file_name_list = data['file_name'].tolist()
# smiles_list = data['SMILES'].tolist()

# for file_name, smiles in tqdm(zip(file_name_list,smiles_list), desc='Processing data', unit='item',dynamic_ncols=True):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         Chem.Draw.MolToImageFile(mol, f'/home/jovyan/data/real_processed/new_aug_data/all_images/{file_name}')
#     except Exception as e:
#         print(file_name)
# print(len(smiles_list))



# import pandas as pd
# from rdkit import Chem

# df = pd.read_csv("/home/jovyan/data/real_processed/new_aug_data/new_data_withH.csv")
# file_name_list = df['file_name'].tolist()
# smiles_list = df['SMILES'].tolist()
# print(len(smiles_list))

# smiles_data = pd.DataFrame({'file_name': [],
#                         'SMILES':[]})


# for file_name, smiles in zip(file_name_list, smiles_list):

#     split_strings = smiles.split('.')
#     longest_part = max(split_strings, key=len)

#     new_row = {'file_name':file_name,
#                 'SMILES':longest_part}
#     smiles_data = smiles_data._append(new_row, ignore_index=True)


# smiles_data.to_csv(f'/home/jovyan/data/real_processed/new_aug_data/new_data_without_dot.csv', index=False)



# from rdkit import Chem
# from xml.dom import minidom
# from rdkit.Chem import Draw

# def _get_svg_doc(mol):
#     """
#     Draws molecule a generates SVG string.
#     :param mol:
#     :return:
#     """
#     dm = Draw.PrepareMolForDrawing(mol)
#     d2d = Draw.MolDraw2DSVG(300, 300)
#     d2d.DrawMolecule(dm)
#     d2d.AddMoleculeMetadata(dm)
#     d2d.FinishDrawing()
#     svg = d2d.GetDrawingText()

#     doc = minidom.parseString(svg)
#     return doc,svg



# params = Chem.SmilesParserParams()
# params.removeHs = False
# smiles = '[H][C@@]1(C(=O)O)[C@]([H])(C(C(N)=O)c2cc(N=[N+]=[N-])c3c(c2)C(=O)[C-](c2cnc4ccccc4c2)C3=O)[C@]2([H])C=C[C@@]1([H])C2'
# mol = Chem.MolFromSmiles(smiles,params)
# # mol = Chem.MolFromSmiles(smiles)
# # Chem.AddHs(mol, explicitOnly=True)
# # print(Chem.MolToSmiles(mol))
# Chem.Draw.MolToImageFile(mol, 'test.png')
# # mol_no_hydrogens = Chem.RemoveHs(mol)
# # smiles_no_hydrogens = Chem.MolToSmiles(mol_no_hydrogens)
# # print("去除显式氢原子后的SMILES表示:")
# # print(smiles_no_hydrogens)

# doc,svg = _get_svg_doc(mol)
# file_name = "output.xml"
# with open(file_name, "w") as f:
#     f.write(svg)
# print(f"SVG内容已保存到XML文件 {file_name}")

# atoms_data = [{'x':    int(round(float(path.getAttribute('drawing-x')), 0)),
#                 'y':    int(round(float(path.getAttribute('drawing-y')), 0)),
#                 'type': ''.join([a.GetSymbol(), str(a.GetFormalCharge())])} for path, a in
#                 zip(doc.getElementsByTagName('rdkit:atom'), mol.GetAtoms())]


# print(len(atoms_data))
# if len(mol.GetAtoms()) < len(doc.getElementsByTagName('rdkit:atom')):
#     len_ = len(doc.getElementsByTagName('rdkit:atom')) - len(mol.GetAtoms())


#     last_n_atom_tags = doc.getElementsByTagName('rdkit:atom')[-len_:]


#     for atom_tag, a in zip(last_n_atom_tags, mol.GetAtoms()):
#         atom_data = {
#             'x': int(round(float(atom_tag.getAttribute('drawing-x')), 0)),
#             'y': int(round(float(atom_tag.getAttribute('drawing-y')), 0)),
#             'type': 'H0'}
#         atoms_data.append(atom_data)

# print(len(atoms_data))
# print(atoms_data)


# Chem.Draw.MolToImageFile(mol, 'test.png')


# from rdkit import Chem
# params = Chem.SmilesParserParams()
# params.removeHs = False

# mol = Chem.MolFromSmiles("C[H]", params)

# for at in mol.GetAtoms():
#     print(at.GetSymbol())




# from rdkit import Chem

# mol = Chem.MolFromSmiles('O=C([C@@H]1[C@H](/C=C/C2=CC=CC=C2)C[C@@H]1/C=C/C3=CC=CC=C3)NC4=C5C(C=CC=N5)=CC=C4')
# for bond in mol.GetBonds():
#     bond_type = str(bond.GetBondType()).split('.')[-1]
#     bond_dir = str(bond.GetBondDir()).split('.')[-1]
#     if bond_type == 'SINGLE':
#         if bond_dir =='BEGINWEDGE' or bond_dir == 'BEGINDASH': 
#             print('+')


# from rdkit import Chem
# from rdkit.Chem import Draw
# from xml.dom import minidom


# def _get_svg_doc(mol):
#     """
#     Draws molecule a generates SVG string.
#     :param mol:
#     :return:
#     """
#     dm = Draw.PrepareMolForDrawing(mol)
#     d2d = Draw.MolDraw2DSVG(300, 300)
#     d2d.DrawMolecule(dm)
#     d2d.AddMoleculeMetadata(dm)
#     d2d.FinishDrawing()
#     svg = d2d.GetDrawingText()

#     doc = minidom.parseString(svg)
#     return doc,svg


# smiles = '[H][C@]1(CC(N)=O)[C@@]([H])(C(=O)O)[C@@]2([H])C=C(c3nc(N)nc(-c4ccco4)c3-c3ccc(=O)n(CC(C)CC)c3)[C@]1([H])C2[N+](=O)[O-]'

# mol = Chem.MolFromSmiles(smiles)
# # params = Chem.SmilesParserParams()
# # params.removeHs = False
# # mol = Chem.MolFromSmiles(smiles,params)

# doc,svg = _get_svg_doc(mol)

# file_name = "output.xml"
# with open(file_name, "w") as f:
#     f.write(svg)
# print(f"SVG内容已保存到XML文件 {file_name}")

# atoms_data = [{'x':    int(round(float(path.getAttribute('drawing-x')), 0)),
#                 'y':    int(round(float(path.getAttribute('drawing-y')), 0)),
#                 'type': ''.join([a.GetSymbol(), str(a.GetFormalCharge())])} for path, a in
#                 zip(doc.getElementsByTagName('rdkit:atom'), mol.GetAtoms())]


# for atom in atoms_data:
#     print([atom['x'],atom['y']],)
# print(atoms_data)





# import rdkit
# from rdkit.Chem import Draw
# from xml.dom import minidom
# import os
# import re
# import cv2
# import numpy as np
# from PIL import Image

# image = Image.open('/home/jovyan/data/real_processed/new_aug_data/images/val/1000.png').convert('RGB')
# image = np.array(image)
# # predict_boxes = [[208, 105],[200, 130],[226, 135],[244, 115],[270, 120],
# # [235, 90],[198, 157],[211, 162],[201, 183],[225, 194],[183, 200],[172, 163],
# # [174, 187],[147, 152],[149, 126],[129, 108],[134, 82],[114, 64],[119, 38],
# # [89, 73],[84, 99],[58, 108],[50, 134],[24, 134],[15, 109],[36, 93],[104, 117],[99, 143],
# # [73, 152],[68, 178],[88, 196],[83, 222],[114, 187],[134, 204],[129, 231],[149, 248],
# # [103, 239],[98, 266],[119, 161],[175, 120],[186, 95],[189, 143],[216, 145],
# # [223, 157],[236, 128]]
# predict_boxes = [[214, 136],
# [240, 123],
# [242, 94],
# [268, 81],
# [218, 78],
# [210, 165],
# [230, 186],
# [222, 214],
# [245, 182],
# [181, 169],
# [155, 156],
# [160, 127],
# [139, 106],
# [147, 78],
# [127, 57],
# [134, 29],
# [98, 64],
# [90, 93],
# [62, 100],
# [51, 127],
# [22, 125],
# [15, 97],
# [40, 81],
# [111, 114],
# [103, 142],
# [75, 149],
# [67, 177],
# [87, 198],
# [80, 226],
# [116, 191],
# [136, 212],
# [128, 240],
# [149, 260],
# [100, 247],
# [92, 275],
# [124, 162],
# [188, 123],
# [202, 149],
# [230, 154],
# [239, 167],
# [254, 137]]
# predict_boxes = np.array(predict_boxes)
# for bbox in predict_boxes:
#     print(int(bbox[0]), int(bbox[1]))
#     cv2.circle(image, (int(bbox[0]), int(bbox[1])),3,(0,0,255),-1)
# cv2.imwrite("image.jpg", image)




# from rdkit import Chem
# from rdkit.Chem import Draw

# smiles = '[H][C@]1(CC(N)=O)[C@@]([H])(C(=O)O)[C@@]2([H])C=C(c3nc(N)nc(-c4ccco4)c3-c3ccc(=O)n(CC(C)CC)c3)[C@]1([H])C2[N+](=O)[O-]'
# params = Chem.SmilesParserParams()
# params.removeHs = False
# mol = Chem.MolFromSmiles(smiles,params)

# Chem.Draw.MolToImageFile(mol, 'image.png',size=(1000,1000))



# import json


# with open(f'/home/jovyan/data/real_processed/merge_with_chiral+charge_large/labels.json', 'r') as f:
#     labels = json.load(f)

# for key, value in labels.items():
#     print(f"{value}:'{key}',")



from rdkit import Chem
from rdkit.Chem import Draw
from xml.dom import minidom


def _get_svg_doc(mol):
    """
    Draws molecule a generates SVG string.
    :param mol:
    :return:
    """
    dm = Draw.PrepareMolForDrawing(mol)
    d2d = Draw.MolDraw2DSVG(300, 300)
    d2d.DrawMolecule(dm)
    d2d.AddMoleculeMetadata(dm)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()

    doc = minidom.parseString(svg)
    return doc,svg


smiles = 'CC(C)(C)C1=CC2=C(C=C1)O[Cr]1(Cl)(Cl)N2=CC2C=CC=CC=2P1(C1C=CC=CC=1)C1C=CC=CC=1'

mol = Chem.MolFromSmiles(smiles)

doc,svg = _get_svg_doc(mol)

file_name = "output.xml"
with open(file_name, "w") as f:
    f.write(svg)
print(f"SVG内容已保存到XML文件 {file_name}")
