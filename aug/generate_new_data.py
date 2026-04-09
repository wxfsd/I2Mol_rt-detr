from tqdm import tqdm
from rdkit_AugSmiles import generate_image
import cv2
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import time
from xml.dom import minidom


df = pd.read_csv('/home/jovyan/data/real_processed/new_aug_data/merged_file.csv')
file_name_list = df['file_name'].tolist()
smiles_list = df['SMILES'].tolist()
print(len(smiles_list))
new_smiles_list  = []
new_smiles_count = 0
unsuccess_count = 0

for file_name, smiles in tqdm(zip(file_name_list,smiles_list), desc='Processing data', unit='item',dynamic_ncols=True):
    try:
        new_smiles = "ThisIsNotAValidSMILESString"
        count = 0

        while Chem.MolFromSmiles(new_smiles) is None and count < 5 :
            img, new_smiles, graph, success = generate_image(idx = file_name, smiles = smiles)
            count += 1

        if Chem.MolFromSmiles(new_smiles) is not None:
            new_smiles_list.append(new_smiles)
            new_smiles_count += 1
        else:
            raise ValueError("error")
    except:
        unsuccess_count += 1
print(new_smiles_count)
print(unsuccess_count)




file_names = [f"{i}.png" for i in range(new_smiles_count+1)]
new_df = pd.DataFrame(list(zip(file_names, new_smiles_list)), columns=['file_name', 'SMILES'])
csv_file = "new_data_0823.csv"
new_df.to_csv(csv_file, index=False)

print(f"数据已成功写入到 {csv_file} 文件中。")


    

