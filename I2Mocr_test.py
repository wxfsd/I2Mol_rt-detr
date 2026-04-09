# from DECIMER import predict_SMILES
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Only GPU 0 will be visible
# os.chdir('/cadd_data/samba_share/bowen/workspace/DECIMER-Image-to-SMILES/Network')
import pandas as pd 
import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import rdkit.Chem.MolStandardize
import os,sys
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

# 关闭 RDKit 日志
os.environ["CUDA_VISIBLE_DEVICES"]='2'
RDLogger.DisableLog('rdApp.*')
sys.path.append("/home/jovyan/rt-detr/rt-detr")
os.chdir('/home/jovyan/rt-detr/rt-detr')
from src.solver.utils import output_to_smiles
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
from src.data import get_coco_api_from_dataset
from src.solver.utils import bbox_to_graph_with_charge,mol_from_graph_with_chiral
from draw_box_utils import draw_objs,STANDARD_COLORS,draw_text
import draw_box_utils
from PIL import ImageColor
import PIL.ImageDraw as ImageDraw
import numpy as np
import src
from rdkit.Chem import Draw, AllChem
import rdkit
from rdkit import Chem
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True,use_gpu =False,
    rec_algorithm='SVTR_LCNet', rec_model_dir='/home/jovyan/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer',
    lang="en")  # need to run only once to download and load model into memory

##################### MolScribe#################################################################################### 
from typing import List

VALENCES = {
    "H": [1], "Li": [1], "Be": [2], "B": [3], "C": [4], "N": [3, 5], "O": [2], "F": [1],
    "Na": [1], "Mg": [2], "Al": [3], "Si": [4], "P": [5, 3], "S": [6, 2, 4], "Cl": [1], "K": [1], "Ca": [2],
    "Br": [1], "I": [1]
}

ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

COLORS = {
    u'c': '0.0,0.75,0.75', u'b': '0.0,0.0,1.0', u'g': '0.0,0.5,0.0', u'y': '0.75,0.75,0',
    u'k': '0.0,0.0,0.0', u'r': '1.0,0.0,0.0', u'm': '0.75,0,0.75'
}
RGROUP_SYMBOLS = ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
                  'Ra', 'Rb', 'Rc', 'Rd', 'X', 'Y', 'Z', 'Q', 'A', 'E', 'Ar']

class Substitution(object):
    '''Define common substitutions for chemical shorthand'''
    def __init__(self, abbrvs, smarts, smiles, probability):
        assert type(abbrvs) is list
        self.abbrvs = abbrvs
        self.smarts = smarts
        self.smiles = smiles
        self.probability = probability

SUBSTITUTIONS: List[Substitution] = [#abbrvs, smarts, smiles
    Substitution(['NO2', 'O2N'], '[N+](=O)[O-]', "[N+](=O)[O-]", 0.5),
    Substitution(['CHO', 'OHC'], '[CH1](=O)', "[CH1](=O)", 0.5),
    Substitution(['CO2Et', 'COOEt'], 'C(=O)[OH0;D2][CH2;D2][CH3]', "[C](=O)OCC", 0.5),

    Substitution(['OAc'], '[OH0;X2]C(=O)[CH3]', "[O]C(=O)C", 0.7),
    Substitution(['NHAc'], '[NH1;D2]C(=O)[CH3]', "[NH]C(=O)C", 0.7),
    Substitution(['Ac'], 'C(=O)[CH3]', "[C](=O)C", 0.1),

    Substitution(['OBz'], '[OH0;D2]C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[O]C(=O)c1ccccc1", 0.7),  # Benzoyl
    Substitution(['Bz'], 'C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[C](=O)c1ccccc1", 0.2),  # Benzoyl

    Substitution(['OBn'], '[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[O]Cc1ccccc1", 0.7),  # Benzyl
    Substitution(['Bn'], '[CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[CH2]c1ccccc1", 0.2),  # Benzyl

    Substitution(['NHBoc'], '[NH1;D2]C(=O)OC([CH3])([CH3])[CH3]', "[NH1]C(=O)OC(C)(C)C", 0.6),
    Substitution(['NBoc'], '[NH0;D3]C(=O)OC([CH3])([CH3])[CH3]', "[NH1]C(=O)OC(C)(C)C", 0.6),
    Substitution(['Boc'], 'C(=O)OC([CH3])([CH3])[CH3]', "[C](=O)OC(C)(C)C", 0.2),

    Substitution(['Cbm'], 'C(=O)[NH2;D1]', "[C](=O)N", 0.2),
    Substitution(['Cbz'], 'C(=O)OC[cH]1[cH][cH][cH1][cH][cH]1', "[C](=O)OCc1ccccc1", 0.4),
    Substitution(['Cy'], '[CH1;X3]1[CH2][CH2][CH2][CH2][CH2]1', "[CH1]1CCCCC1", 0.3),
    Substitution(['Fmoc'], 'C(=O)O[CH2][CH1]1c([cH1][cH1][cH1][cH1]2)c2c3c1[cH1][cH1][cH1][cH1]3',
                 "[C](=O)OCC1c(cccc2)c2c3c1cccc3", 0.6),
    Substitution(['Mes'], '[cH0]1c([CH3])cc([CH3])cc([CH3])1', "[c]1c(C)cc(C)cc(C)1", 0.5),
    Substitution(['OMs'], '[OH0;D2]S(=O)(=O)[CH3]', "[O]S(=O)(=O)C", 0.7),
    Substitution(['Ms'], 'S(=O)(=O)[CH3]', "[S](=O)(=O)C", 0.2),
    Substitution(['Ph'], '[cH0]1[cH][cH][cH1][cH][cH]1', "[c]1ccccc1", 0.5),
    Substitution(['PMB'], '[CH2;D2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[CH2]c1ccc(OC)cc1", 0.2),
    Substitution(['Py'], '[cH0]1[n;+0][cH1][cH1][cH1][cH1]1', "[c]1ncccc1", 0.1),
    Substitution(['SEM'], '[CH2;D2][CH2][Si]([CH3])([CH3])[CH3]', "[CH2]CSi(C)(C)C", 0.2),
    Substitution(['Suc'], 'C(=O)[CH2][CH2]C(=O)[OH]', "[C](=O)CCC(=O)O", 0.2),
    Substitution(['TBS'], '[Si]([CH3])([CH3])C([CH3])([CH3])[CH3]', "[Si](C)(C)C(C)(C)C", 0.5),
    Substitution(['TBZ'], 'C(=S)[cH]1[cH][cH][cH1][cH][cH]1', "[C](=S)c1ccccc1", 0.2),
    Substitution(['OTf'], '[OH0;D2]S(=O)(=O)C(F)(F)F', "[O]S(=O)(=O)C(F)(F)F", 0.7),
    Substitution(['Tf'], 'S(=O)(=O)C(F)(F)F', "[S](=O)(=O)C(F)(F)F", 0.2),
    Substitution(['TFA'], 'C(=O)C(F)(F)F', "[C](=O)C(F)(F)F", 0.3),
    Substitution(['TMS'], '[Si]([CH3])([CH3])[CH3]', "[Si](C)(C)C", 0.5),
    Substitution(['Ts'], 'S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[S](=O)(=O)c1ccc(C)cc1", 0.6),  # Tos

    # Alkyl chains
    Substitution(['OMe', 'MeO'], '[OH0;D2][CH3;D1]', "[O]C", 0.3),
    Substitution(['SMe', 'MeS'], '[SH0;D2][CH3;D1]', "[S]C", 0.3),
    Substitution(['NMe', 'MeN'], '[N;X3][CH3;D1]', "[NH]C", 0.3),
    Substitution(['Me'], '[CH3;D1]', "[CH3]", 0.1),
    Substitution(['OEt', 'EtO'], '[OH0;D2][CH2;D2][CH3]', "[O]CC", 0.5),
    Substitution(['Et', 'C2H5'], '[CH2;D2][CH3]', "[CH2]C", 0.3),
    Substitution(['Pr', 'nPr', 'n-Pr'], '[CH2;D2][CH2;D2][CH3]', "[CH2]CC", 0.3),
    Substitution(['Bu', 'nBu', 'n-Bu'], '[CH2;D2][CH2;D2][CH2;D2][CH3]', "[CH2]CCC", 0.3),

    # Branched
    Substitution(['iPr', 'i-Pr'], '[CH1;D3]([CH3])[CH3]', "[CH1](C)C", 0.2),
    Substitution(['iBu', 'i-Bu'], '[CH2;D2][CH1;D3]([CH3])[CH3]', "[CH2]C(C)C", 0.2),
    Substitution(['OiBu'], '[OH0;D2][CH2;D2][CH1;D3]([CH3])[CH3]', "[O]CC(C)C", 0.2),
    Substitution(['OtBu'], '[OH0;D2][CH0]([CH3])([CH3])[CH3]', "[O]C(C)(C)C", 0.6),
    Substitution(['tBu', 't-Bu'], '[CH0]([CH3])([CH3])[CH3]', "[C](C)(C)C", 0.3),

    # Other shorthands (MIGHT NOT WANT ALL OF THESE)
    Substitution(['CF3', 'F3C'], '[CH0;D4](F)(F)F', "[C](F)(F)F", 0.5),
    Substitution(['NCF3', 'F3CN'], '[N;X3][CH0;D4](F)(F)F', "[NH]C(F)(F)F", 0.5),
    Substitution(['OCF3', 'F3CO'], '[OH0;X2][CH0;D4](F)(F)F', "[O]C(F)(F)F", 0.5),
    Substitution(['CCl3'], '[CH0;D4](Cl)(Cl)Cl', "[C](Cl)(Cl)Cl", 0.5),
    Substitution(['CO2H', 'HO2C', 'COOH'], 'C(=O)[OH]', "[C](=O)O", 0.5),  # COOH
    Substitution(['CN', 'NC'], 'C#[ND1]', "[C]#N", 0.5),
    Substitution(['OCH3', 'H3CO'], '[OH0;D2][CH3]', "[O]C", 0.4),
    Substitution(['SO3H'], 'S(=O)(=O)[OH]', "[S](=O)(=O)O", 0.4),
]
ABBREVIATIONS = {abbrv: sub for sub in SUBSTITUTIONS for abbrv in sub.abbrvs}
def _expand_abbreviation(abbrev):
    """
    Expand abbreviation into its SMILES; also converts [Rn] to [n*]
    Used in `_condensed_formula_list_to_smiles` when encountering abbrev. in condensed formula
    """
    if abbrev in ABBREVIATIONS:
        return ABBREVIATIONS[abbrev].smiles
    if abbrev in RGROUP_SYMBOLS or (abbrev[0] == 'R' and abbrev[1:].isdigit()):
        if abbrev[1:].isdigit():
            return f'[{abbrev[1:]}*]'
        return '*'
    return f'[{abbrev}]'

from rdkit.Chem import rdchem, RWMol, CombineMols

def expandABB(mol,ABBREVIATIONS, placeholder_atoms):
    mols = [mol]
    # **第三步: 替换 * 并合并官能团**
    # 逆序遍历 placeholder_atoms，确保删除后不会影响后续索引
    for idx in sorted(placeholder_atoms.keys(), reverse=True):
        group = placeholder_atoms[idx]  # 获取官能团名称
        # print(idx, group)
        submol = Chem.MolFromSmiles(ABBREVIATIONS[group].smiles)  # 获取官能团的子分子
        submol_rw = RWMol(submol)  # 让 submol 变成可编辑的 RWMol
        anchor_atom_idx = 0  # 选择 `submol` 的第一个原子作为连接点 as defined in ABBREVIATIONS
        # **1. 复制主分子**
        new_mol = RWMol(mol)
        # **2. 计算 `*` 在 `new_mol` 中的索引**
        placeholder_idx = idx
        # **3. 记录 `*` 原子的邻居**
        neighbors = [nb.GetIdx() for nb in new_mol.GetAtomWithIdx(placeholder_idx).GetNeighbors()]
        # **4. 断开 `*` 的所有键**
        bonds_to_remove = []  # 记录要断开的键
        for bond in new_mol.GetBonds():
            if bond.GetBeginAtomIdx() == placeholder_idx or bond.GetEndAtomIdx() == placeholder_idx:
                bonds_to_remove.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for bond in bonds_to_remove:
            new_mol.RemoveBond(bond[0], bond[1])
        # **5. 删除 `*` 原子**
        new_mol.RemoveAtom(placeholder_idx)
        # **6. 重新计算 `neighbors`（删除后索引变化）**
        new_neighbors = []
        for neighbor in neighbors:
            if neighbor < placeholder_idx:
                new_neighbors.append(neighbor)
            else:
                new_neighbors.append(neighbor - 1)  # 因为删除了一个原子，所有索引 -1
        # **7. 合并 `submol`**
        new_mol = RWMol(CombineMols(new_mol, submol_rw))

        # **8. 计算 `submol` 的第一个原子在合并后的位置**
        new_anchor_idx = new_mol.GetNumAtoms() - len(submol_rw.GetAtoms()) + anchor_atom_idx

        # **9. 重新连接官能团**
        for neighbor in new_neighbors:
            # print(neighbor, new_anchor_idx, "!!")
            new_mol.AddBond(neighbor, new_anchor_idx, Chem.BondType.SINGLE)
            a1=new_mol.GetAtomWithIdx(neighbor)
            a2=new_mol.GetAtomWithIdx(new_anchor_idx)
            a1.SetNumRadicalElectrons(0)
            a2.SetNumRadicalElectrons(0)## 将自由基电子数设为 0,as has added new bond
        # **10. 更新主分子**
        mol = new_mol
        mols.append(mol)
    # # 遍历分子中的每个原子
    # for atom in mols[-1].GetAtoms(): NOTE considering original image has the RadicalElectrons
    #     atom_idx = atom.GetIdx()  # 原子索引
    #     radical_electrons = atom.GetNumRadicalElectrons()  # 自由基电子数
    #     if radical_electrons > 0:
    #         # print(f"原子 {atom_idx} 存在自由基，自由基电子数: {radical_electrons}\n current NumExplicitHs: {atom.GetNumExplicitHs()}")
    #         # 消除自由基：通过添加氢原子调整价态
    #         atom.SetNumRadicalElectrons(0)  # 将自由基电子数设为 0,as has added bond
    #         # atom.SetNumExplicitHs(atom.GetNumExplicitHs() + radical_electrons) 
    Chem.SanitizeMol(mols[-1])
    # 输出修改后的分子 SMILES
    modified_smiles = Chem.MolToSmiles(mols[-1])
    # print(f"修改后的分子 SMILES: {modified_smiles}")            
    return mols[-1], modified_smiles

def image_to_tensor(image_path,debug=False):
    # Open the image using PIL
    image = Image.open(image_path)
    # if debug: print("image.mode",image.mode)
    w, h = image.size
    # 判断图片通道数
    if image.mode == "L":  
        if debug: print("检测到灰度图像 (1 通道)，转换为 RGB...")
        image = image.convert("RGB")  # 灰度转换为 RGB（3 通道）
    elif image.mode != "RGB":
        if debug: print(f"检测到 {image.mode} 模式，转换为 RGB...")
        image = image.convert("RGB")  # 其他模式转换为 RGB
    # print("width: {}, height: {}".format(w, h))
    # Define a transform to convert the image to a tensor and normalize it
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
        T.Resize((640, 640)),  # Resize the image to 224x224
        T.ToImageTensor(),  # Convert to Tensor (C x H x W)
        T.ConvertDtype(dtype=torch.float32)
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Optional normalization for pretrained models
    ])
    
    # Apply the transform to the image
    tensor = transform(image)
    
    return tensor,w,h

def show_atom_number(mol, label='molAtomMapNumber'):
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return mol
    
def remove_atom_number(mol, label='molAtomMapNumber'):
    # After performing the desired operation (e.g., showing), you can remove the property
    for atom in mol.GetAtoms():
        atom.ClearProp(label)  # Removes the atom index property
    return mol

# After performing the desired operation (e.g., showing), you can remove the property
    for atom in mol.GetAtoms():
        atom.ClearProp(label)  # Removes the atom index property

def remove_SP(input_string):
    # if "S@SP1" in input_string:
    #     input_string = input_string.replace("S@SP1", "S")
    # elif "S@SP2" in input_string:
    #     input_string = input_string.replace("S@SP2", "S")
    # elif "S@SP3" in input_string:
    #     input_string = input_string.replace("S@SP3", "S")
    input_string = re.sub(r'@SP[1-3]', '', input_string)
    return input_string

import pandas as pd
import math
from scipy.spatial import cKDTree


def assemble_atoms_with_charges(atom_list, charge_list):
    used_charge_indices=set()
    kdt = cKDTree(atom_list[['x','y']])
    for i, charge in charge_list.iterrows():
        if i in used_charge_indices:
            continue
        charge_=charge['charge']
        if charge_=='1':charge_='+'
        dist, idx_atom=kdt.query([charge_list.x[i],charge_list.y[i]], k=1)
        atom_str=atom_list.loc[idx_atom,'atom'] 
        atom_ = re.findall(r'[A-Za-z]+', atom_str)[0] + charge_
        atom_list.loc[idx_atom,'atom']=atom_

    return atom_list
    


import re
pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)

def atomwise_tokenizer(smi, regex=regex,exclusive_tokens = None):
    """
    Tokenize a SMILES molecule at atom-level:
        (1) 'Br' and 'Cl' are two-character tokens
        (2) Symbols with bracket are considered as tokens

    exclusive_tokens: A list of specifical symbols with bracket you want to keep. e.g., ['[C@@H]', '[nH]'].
    Other symbols with bracket will be replaced by '[UNK]'. default is `None`.
    """
    tokens = [token for token in regex.findall(smi)]
    if exclusive_tokens:
        for i, tok in enumerate(tokens):
            if tok.startswith('['):
                if tok not in exclusive_tokens:
                    tokens[i] = '[UNK]'
    return tokens
bond_labels = [13,14,15,16,17]
# idx_to_labels = {0:'other',1:'C',2:'O',3:'N',4:'Cl',5:'Br',6:'S',7:'F',8:'B',
#             9:'I',10:'P',11:'*',12:'Si',13:'NONE',14:'BEGINWEDGE',15:'BEGINDASH',
#             16:'=',17:'#',18:'-4',19:'-2',20:'-1',21:'1',22:'+2',} #NONE is single ?
idx_to_labels23={0:'other',1:'C',2:'O',3:'N',4:'Cl',5:'Br',6:'S',7:'F',8:'B',
                    9:'I',10:'P',11:'*',12:'Si',13:'NONE',14:'BEGINWEDGE',15:'BEGINDASH',
                    16:'=',17:'#',18:'-4',19:'-2',20:'-1',21:'1',22:'2'}
 
idx_to_labels30 = {0:'other',1:'C',2:'O',3:'N',4:'Cl',5:'Br',6:'S',7:'F',8:'B',
                    9:'I',10:'P',11:'H',12:'Si',13:'NONE',14:'BEGINWEDGE',15:'BEGINDASH',
                    16:'=',17:'#',18:'-4',19:'-2',20:'-1',21:'1',22:'2',
                    23:'CF3',#NOTE rdkit get element not supporting group
                    24:'CN',
                    25:'Me',
                    26:'CO2Et',
                    27:'R',
                    28:'Ph',
                    29:'*',
                    }
abrevie={"[23*]":'CF3',
                                    "[24*]":'CN',
                                    "[25*]":'Me',
                                    "[26*]":'CO2Et',
                                    "[27*]":'R',
                                    "[28*]":'Ph',
                                    "[29*]":'3~7UP',
        }

# idx_to_labels=idx_to_labels30
# idx_to_labels=idx_to_labels23

home="/home/jovyan/rt-detr"
pt_outhome='/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output'
pp="/home/jovyan/rt-detr/rt-detr/tools/output/rtdetr_r50vd_6x_coco_real_resample_charge_large/best_checkpoint.pth"
cc="/home/jovyan/rt-detr/rt-detr/tools/output/rtdetr_r50vd_6x_coco_real_resample_adapter_both/checkpoint0068.pth"
tt="/home/jovyan/rt-detr/rt-detr/output/rtdetr_r50vd_6x_coco_real_resample_charge_large_adpter2/best_checkpoint.pth"
diffS='/home/jovyan/rt-detr/rt-detr/output/rtdetr_r50vd_6x_coco_real_resample_charge_large_adpterWithoutJPO_diffSize/checkpoint0071.pth'
# tr1='blured_merged_diff300start11'
# tr1='blured_merged_diff300start12'
# tr1='blured_merged_diff300start12_hand'
# tr1='blured_merged_diff300start12_hand_addedObstac'
# tr1='merged9'
#NOTE need change the class number@coco_detection.yml when change weight as trained in different class number 
bmd= '/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output/merged9/best_checkpoint.pth'#30
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default=f'{home}/rt-detr/configs/rtdetr/rtdetr_r50vd_6x_coco.yml')
# parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/tools/output/rtdetr_r50vd_6x_coco_real_resample_charge_large/checkpoint0032.pth')
# parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/tools/output/rtdetr_r50vd_6x_coco_real_resample/checkpoint0052.pth')
# parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{pp}')
# parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{cc}')
parser.add_argument('--resume', '-r', type=str, default=f'{tt}')
# parser.add_argument('--resume', '-r', type=str, default=f'{diffS}')
# parser.add_argument('--resume', '-r', type=str, default=f'{pt_outhome}/{tr1}/best_checkpoint.pth')
# parser.add_argument('--resume', '-r', type=str, default=f'{bmd}')#last trained



parser.add_argument('--tuning', '-t', type=str,)# default='/home/jovyan/model_checkpoint/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth')
parser.add_argument('--test-only',default=True,)
parser.add_argument('--amp', default=False,)

args, unknown = parser.parse_known_args()#in jupyter
if args.resume in [tt,pp,cc,diffS]:
    idx_to_labels=idx_to_labels23
else:
    idx_to_labels=idx_to_labels30
print(f'number of class {len(idx_to_labels)} !!!!!!!!!! \n {idx_to_labels}')

cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )
args.gpu_device=1
cfg.device=torch.device('cuda', args.gpu_device) if torch.cuda.is_available() else torch.device('cpu') 
_model=cfg.model
#postprocess need the image original size
if torch.cuda.is_available():
    saved_statDict=torch.load(cfg.resume)
else:
    saved_statDict=torch.load(cfg.resume,map_location=torch.device('cpu'))
loaded_state_dict=saved_statDict['model']
#comaparing with pretrained_model
current_model_dict=_model.state_dict()
postprocessor = cfg.postprocessor##RTDETRPostProcessor@@src/zoo/rtertr
from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor



postprocessor2=RTDETRPostProcessor(num_classes=len(idx_to_labels), use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False)
#loaidng trained weights

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#NOTE make sure current_model_dict contating all the loaded_state_dict
key_ori=loaded_state_dict.keys()
key_cur=current_model_dict.keys()
diff_cur=[k for k in key_cur  if k not in key_ori]
diff_ori=[k for k in key_ori if k not in key_cur]#loaded is the original as pretrained
#hehre we only considering new model >= pretrained, not the other case here
same_=[k for k in key_cur if k in key_ori]
# assert len(diff_ori)==0, 
print(f'make sure loaded pretrained model keys: {len(key_ori)} are all included in current build model keys: {len(key_cur)}\n In futhre we considering only part intersection!!')
new_state_dict=dict()
i=0;j=0
for k in same_:
    if loaded_state_dict[k].size()==current_model_dict[k].size():
        new_state_dict[k]=loaded_state_dict[k]
        # print(f'current{k},{current_model_dict[k].size()},ori{loaded_state_dict[k].size()}')
        i+=1
    else:
        new_state_dict[k]=current_model_dict[k] #confg such image_size diff lead
        j+=1
_model.load_state_dict(new_state_dict, strict=False)
print(f'{bcolors.WARNING}have partly load the match paramters>>number of:{len(new_state_dict.keys())}@finalLoad, all:{len(current_model_dict.keys())}@buildModel, {len(loaded_state_dict.keys())}@pretrained. {bcolors.ENDC}')
print(f' {bcolors.WARNING} loaded keys {i}, skip mismatch size keys {j} (same nnName but diff dimensions)')
print("when training use this as valdation dataset::",
    cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'],"\n",
cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'])


def remove_backslash_and_slash(input_string):
    if "\\" in input_string:
        input_string = input_string.replace("\\", "")
    if "/" in input_string:
        input_string = input_string.replace("/", "")

    return input_string


def remove_number_before_star(input_string):
    result = list(input_string) 

    i = 0
    while i < len(result):
        if result[i] == '*' and i!= len(result) -1:  
            #*c1c(*)c(*)c(C(*)(*)C(C)C)c(*)c1* --> *c1c(*)c(*)c(C(*)(*)C(C)C)c(*)c1*
            j = i - 1
            if result[j-1].isalpha(): 
                continue
            while j >= 0 and result[j].isdigit():
                result[j] = ''  
                j -= 1
        i += 1

    return ''.join(result)

def remove_SP(input_string):
    # if "S@SP1" in input_string:
    #     input_string = input_string.replace("S@SP1", "S")
    # elif "S@SP2" in input_string:
    #     input_string = input_string.replace("S@SP2", "S")
    # elif "S@SP3" in input_string:
    #     input_string = input_string.replace("S@SP3", "S")
    input_string = re.sub(r'@SP[1-3]', '', input_string)

    return input_string

def I2M(image_path,debug=False):
    # Example usage: #change thie image
    tensor,w,h = image_to_tensor(image_path,True)
    tensor=tensor.unsqueeze(0)
    # print(tensor.size())  # Output tensor shape (C x H x W)
    # _model.training=False
    _model.eval()#have to uset this
    with torch.no_grad():
        if debug:print("_model.training:",_model.training)
        outputs = _model(tensor)
    ori_size=torch.Tensor([w,h]).long().unsqueeze(0)
    # result_ = postprocessor(outputs, ori_size)
    result_ = postprocessor2(outputs, ori_size)
    # result_ = postprocessor(out_, torch.Tensor([w,h]))
    score_=result_[0]['scores']
    boxe_=result_[0]['boxes']
    label_=result_[0]['labels']
    selected_indices =score_ > 0.5
    # selected_indices =score_ > score_.mean()
    if debug:
        # 统计 True 的数量
        true_count = selected_indices.sum().item()
        print(f"selected_indices 中 True 的数量: {true_count}")

    output={
        'labels': label_[selected_indices],
        'boxes': boxe_[selected_indices],
        'scores': score_[selected_indices]
    }
    # filtered_output_dict={image_path: output  }
    x_center = (output["boxes"][:, 0] + output["boxes"][:, 2]) / 2
    y_center = (output["boxes"][:, 1] + output["boxes"][:, 3]) / 2
    center_coords = torch.stack((x_center, y_center), dim=1)
    output = {'bbox':         output["boxes"].to("cpu").numpy(),
                'bbox_centers': center_coords.to("cpu").numpy(),
                'scores':       output["scores"].to("cpu").numpy(),
                'pred_classes': output["labels"].to("cpu").numpy()}
    atoms_df, bonds_list,charge_list =bbox_to_graph_with_charge(output, idx_to_labels=idx_to_labels,
                                                    bond_labels=bond_labels,  result=[])
    smiles,mol_rebuit=mol_from_graph_with_chiral(atoms_df, bonds_list,charge_list )#
    return smiles,mol_rebuit,output

def I2M(image_path,debug=False):
    # Example usage: #change thie image
    tensor,w,h = image_to_tensor(image_path,True)
    tensor=tensor.unsqueeze(0)
    # print(tensor.size())  # Output tensor shape (C x H x W)
    # _model.training=False
    _model.eval()#have to uset this
    with torch.no_grad():
        if debug:print("_model.training:",_model.training)
        outputs = _model(tensor)
    ori_size=torch.Tensor([w,h]).long().unsqueeze(0)
    # result_ = postprocessor(outputs, ori_size)
    result_ = postprocessor2(outputs, ori_size)
    # result_ = postprocessor(out_, torch.Tensor([w,h]))
    score_=result_[0]['scores']
    boxe_=result_[0]['boxes']
    label_=result_[0]['labels']
    selected_indices =score_ > 0.5
    # selected_indices =score_ > score_.mean()
    if debug:
        # 统计 True 的数量
        true_count = selected_indices.sum().item()
        print(f"selected_indices 中 True 的数量: {true_count}")

    output={
        'labels': label_[selected_indices],
        'boxes': boxe_[selected_indices],
        'scores': score_[selected_indices]
    }
    # filtered_output_dict={image_path: output  }
    x_center = (output["boxes"][:, 0] + output["boxes"][:, 2]) / 2
    y_center = (output["boxes"][:, 1] + output["boxes"][:, 3]) / 2
    center_coords = torch.stack((x_center, y_center), dim=1)
    output = {'bbox':         output["boxes"].to("cpu").numpy(),
                'bbox_centers': center_coords.to("cpu").numpy(),
                'scores':       output["scores"].to("cpu").numpy(),
                'pred_classes': output["labels"].to("cpu").numpy()}
    atoms_df, bonds_list,charge_list =bbox_to_graph_with_charge(output, idx_to_labels=idx_to_labels,
                                                    bond_labels=bond_labels,  result=[])
    smiles,mol_rebuit=mol_from_graph_with_chiral(atoms_df, bonds_list,charge_list )#
    return smiles,mol_rebuit

def ocr(smiles,mol_rebuit,image_path,atoms_df):
    #ocr part
    mol = rdkit.Chem.RWMol(mol_rebuit)
    need_cut=[]
    crops=[]
    index_token=dict()
    ppstr=[]
    ppstr_score=[]
    expan=3#NOTE this control how much the part of bond in crop_Img
    img_ori = Image.open(image_path).convert('RGB')
    img_ori_1k = img_ori.resize((1000,1000))
    # for i_, atom_s in enumerate(atom_df['atom']):
    for i_, row in atoms_df.iterrows():
        if "*" in row.atom or "other" in row.atom:
            need_cut.append(i_)
            a=np.array(row.bbox )+np.array([-expan,-expan,expan,expan])#expand crop
            box=a * 10/3
            cropped_img = img_ori_1k.crop(box)
            crops.append(cropped_img)
            image_np = np.array(cropped_img)
            result = ocr.ocr(image_np, det=False)
            s_, score_ =result[0][0]
            if score_<=0.1:# process cropped_img and try again
                print(s_, "xxx",score_)
                s_='*'
            if s_=='+' or s_=='-':
                s_="*"
            if len(s_)>1:
                s_=re.sub(r'[^a-zA-Z0-9]', '', s_)#remove special chars
                if re.match(r'^\d+$', s_):print(f'why only numbers ?  {s_}')
            index_token[i_]=f'{s_}:{i_}'
            print(f"idx:{i_}, atm:{row.atom}-->[{s_}:{i_}] with score:{score_}")
            mol.GetAtomWithIdx(i_).SetProp("atomLabel", f"{s_}")#TODO check and goon
            ppstr.append(s_)
            ppstr_score.append(score_)
    # TODO ABBRE AND FLAT SMILES BOTH for checking if one ok,acc+1
    return smiles,mol_rebuit,output#ipynb testing two mol and smiles for acc +1


def comparing_smiles(original_smiles,test_smiles):
    original_smiles = remove_backslash_and_slash(original_smiles)
    test_smiles = remove_backslash_and_slash(test_smiles)
    # original_smiles = remove_number_before_star(original_smiles)
    # test_smiles = remove_number_before_star(test_smiles)
    original_smiles = re.sub(r'\[(\d+)\*', '[*',original_smiles)#[1*]-->[*]
    test_smiles = re.sub(r'\[(\d+)\*', '[*',test_smiles)
    original_smiles = remove_SP(original_smiles)
    test_smiles = remove_SP(test_smiles)

    original_mol = Chem.MolFromSmiles(original_smiles)
    test_mol = Chem.MolFromSmiles(test_smiles)
    
    Chem.SanitizeMol(original_mol)
    Chem.SanitizeMol(test_mol)
    try:
        keku_smi_ori=Chem.MolToSmiles(original_mol,kekuleSmiles=True)
        keku_smi=Chem.MolToSmiles(test_mol,kekuleSmiles=True)
        if '*' not in keku_smi:
            keku_inch_ori=  Chem.MolToInchi(Chem.MolFromSmiles(keku_smi_ori))
            keku_inch_test=  Chem.MolToInchi(Chem.MolFromSmiles(keku_smi))
        else:
            keku_inch_ori=  1
            keku_inch_test=  2
    except Exception as e:
        print(f"kekulize problems original_smiles,test_smiles\n{original_smiles}\n{test_smiles}")
        keku_inch_ori=  1
        keku_inch_test=  2
        keku_smi=1
        keku_smi_ori=2
    rd_smi=Chem.MolToSmiles(test_mol)
    rd_smi_ori=Chem.MolToSmiles(original_mol)
    if rd_smi_ori == rd_smi or keku_smi_ori == keku_smi or keku_inch_ori==keku_inch_test :#as orinial smiles may use kekuleSmiles style
        return True
    else:return False

debug=False
das=[
"acs",
# "CLEF",
# "JPO",
# "UOB",
# "USPTO",
# "staker",
# "hand",
]
# ffou=open(f'DecimerV2.log' , 'w')
debug=True

for da in das:
    print(da)
    sums = 0
    diffs=[]
    sums2 = 0
    diffs2=[]
    simRD=0
    sim=0
    mysum=0
    mydiff=[]
    flogout = open(f'{da}I2Mout.txt' , 'w')
    # 添加第3列
    failed=[]
    new_column = []
    rdkit_canconlized=[]
    rdkit_canconlized_diff=[]

    # csv_path =f"/cadd_data/samba_share/from_docker/data/csv_deal/{da}.csv" #279
    # acs_dir = f"/cadd_data/samba_share/from_docker/data/work_space/{da}" #*smi 279
    csv_path =f"/home/jovyan/volume/samba_share/from_docker/data/csv_deal/{da}.csv" #279
    acs_dir = f"/home/jovyan/volume/samba_share/from_docker/data/work_space/{da}" #*smi 279、
    if da=='hand':
        csv_path =f"/home/jovyan/volume/samba_share/from_docker/ocr_data/handDrawLike/DECIMER_HDM/DECIMER_HDM_Dataset_SMILES.tsv.all.csv" #279
        acs_dir = f"/home/jovyan/volume/samba_share/from_docker/ocr_data/handDrawLike/DECIMER_HDM/images/train" #4577
    png_files = [file for file in os.listdir(f'{acs_dir}') if file.endswith('.png')]
    df = pd.read_csv(csv_path,header=None)

    for i, row in df.iterrows():
        png_filename = df.loc[i,0]
        smiles = df.loc[i,1]
        if da=='hand'and png_filename not in png_files:
            continue
        img_path = os.path.join(acs_dir, png_filename)
        if debug:
            print(i, img_path)
        try:
            # image_path = "../Sample_Images/P_16381197.png"
            # smiles_pred = predict_SMILES(img_path)
            smiles_pred, mol_pred,output=I2M(img_path)
            # result = evaluate(img_path)#put in for loop
            # smiles_pred=converter.decode(''.join(result).replace("<start>","").replace("<end>",""))
        except Exception as e:
            print(e)
            # if smiles_pred:
            smiles_pred=None
            print(f'I2M output problems')
        df.loc[i,2]=smiles_pred
        new_column.append(smiles_pred)

        if (type(smiles)!=type('a')) or (type(smiles_pred)!=type('a')):
            print(f"smiles problems\n{smiles}\n{smiles_pred}\n{img_path}")
            failed.append([smiles,smiles_pred,img_path])
            continue
        mol1 = Chem.MolFromSmiles(smiles)
        mol2 = Chem.MolFromSmiles(smiles_pred)
        if (mol2 is None) or (mol1 is None):
            print(f'get rdkit mol None\n{smiles}\n{smiles_pred}\n{img_path}')
            failed.append([smiles,smiles_pred,img_path])
            simRD+=0
            sim+=0
            continue
        try:#fingerprint sim adding
            morganfps1 = AllChem.GetMorganFingerprint(mol1, 3,useChirality=True)
            morganfps2 = AllChem.GetMorganFingerprint(mol2, 3,useChirality=True)
            morgan_tani = DataStructs.DiceSimilarity(morganfps1, morganfps2)
            fp1 = Chem.RDKFingerprint(mol1)
            fp2 = Chem.RDKFingerprint(mol2)
            tanimoto = DataStructs.FingerprintSimilarity(fp1, fp2)
            simRD +=tanimoto
            sim +=morgan_tani
        except Exception as e:
            print(f"mol to fingerprint erros")
            simRD+=0
            sim+=0
            continue
    
        same_diff=comparing_smiles(smiles,smiles_pred)
        rdk_smi1=Chem.MolToSmiles(mol1)
        rdk_smi2=Chem.MolToSmiles(mol2)
        if rdk_smi1==rdk_smi2 or same_diff:
            mysum += 1
            rdkit_canconlized.append([img_path, smiles, smiles_pred])

        else:
            mydiff.append([smiles,smiles_pred,img_path])
            rdkit_canconlized_diff.append([img_path, smiles, smiles_pred])
            #TODO plot here for diff checking with diff dataset dir
            
        # smiles = rdkit.Chem.MolStandardize.canonicalize_tautomer_smiles(smiles)
        # smiles_pred = rdkit.Chem.MolStandardize.canonicalize_tautomer_smiles(smiles_pred)
        smiles = rdMolStandardize.StandardizeSmiles(smiles)
        smiles_pred = rdMolStandardize.StandardizeSmiles(smiles_pred)
        #https://github.com/rdkit/rdkit/blob/master/Docs/Notebooks/MolStandardize.ipynb
        if smiles==smiles_pred:
            sums += 1
        else:
            diffs.append([smiles,smiles_pred,img_path])

        try:
            mol1 = Chem.MolFromSmiles(smiles)
            mol2 = Chem.MolFromSmiles(smiles_pred)
            smiles1 = Chem.MolToSmiles(mol1,canonical=True,isomericSmiles=False)
            smiles2 = Chem.MolToSmiles(mol2,canonical=True,isomericSmiles=False)#after std, smiles maybe none?
        except Exception as e:
            print(e)
            print(f'std make smiles be none or failed!!!!!!!!!')
            print([smiles,smiles_pred,img_path])
            # failed.append([smiles,smiles_pred,img_path])#as diffs2 append latter
        if smiles1==smiles2:
            sums2 += 1
        else:
            diffs2.append([smiles,smiles_pred,img_path])    

    acc = sums/len(df)
    acc2 = sums2/len(df)
    sim_100 = 100*sim/len(df)
    simrd100 = 100*simRD/len(df)
    flogout.write(f"rdkit concanlized==smiles:{100*mysum/len(df)}%\n")
    flogout.write(f"rdMolStandardize.StandardizeSmiles:{acc}\n")
    flogout.write(f"after StandardizeSmiles:{acc2}\n")
    flogout.write(f"DiceSimilarity:{sim}\n")
    # 将新列添加到 DataFrame
    # df[] = new_column
    flogout.write(f"sum,{mysum},{sums},{sums2}\n")
    flogout.write(f"diffs nums:{len(mydiff)},{len(diffs)},{len(diffs2)}\n")
    flogout.write(f"failed:{len(failed)}\n")
    flogout.write(f"ava similarity morgan tanimoto: RDKFp tanimoto:: {sim_100}%,  {simrd100}%  \n")#morgan_tani considering chiraty

    flogout.write(f'I2M@@{da}:: match--{mysum},unmatch--{len(mydiff)},failed--{len(failed)},correct %{100*mysum/len(df)} ')
    flogout.close()
    # if da=='hand':
    rdkit_canconlized_df = pd.DataFrame(rdkit_canconlized, columns=["img_path", "smiles", "smiles_pred"])
    rdkit_canconlized_df_diff = pd.DataFrame(rdkit_canconlized_diff, columns=["img_path", "smiles", "smiles_pred"])
    rdkit_canconlized_df.to_csv(f"{da}_output.csv", index=False)
    rdkit_canconlized_df_diff.to_csv(f"{da}_diff_output.csv", index=False)

    # line_=f'DecimerV2@@{da}:: match--{mysum},unmatch--{len(mydiff)},failed--{len(failed)},correct %{100*mysum/len(df)}'
    # ffou.write(line_)
    # print(f'DecimerV2@@{d}:: match--{mysum},unmatch--{len(mydiff)},failed--{len(failed)},correct %{100*mysum/len(df)}',flush=True)

# ffou.close()

"""
@gpu2
source /cadd_data/samba_share/from_docker/cu12/etc/profile.d/conda.sh
conda activate 
cd /cadd_data/samba_share/bowen/workspace/dv2_test

nohup python /cadd_data/samba_share/bowen/workspace/decimerV2_.py > allv2.out 2>&1 & 
python /home/jovyan/rt-detr/LG_SMILES_1st-main/validation.py


nohup python I2M_test.py > all_rI2M.out 2>&1 & 

"""
