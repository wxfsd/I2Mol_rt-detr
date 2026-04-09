import copy
import json
import math
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import rdDepictor
import matplotlib.pyplot as plt
import re
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
charge_labels = [18,19,20,21,22]




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

def Val_extract_atom_info(error_message):
    """
    从错误信息中提取 atomid, atomType 和 valence。
    :param error_message: 错误信息字符串
    :return: (atomid, atomType, valence) 元组
    """
    # 定义正则表达式来提取原子信息
    pattern = r"Explicit valence for atom # (\d+) (\w), (\d+)"
    pattern2 =r"Explicit valence for atom # (\d+) (\w) "
    # print(type(error_message))
    if not isinstance(error_message, type('strs')):
        error_message=str(error_message)
    match = re.search(pattern, error_message)
    match2 = re.search(pattern2, error_message)
    if match:
        # 提取 atomid, atomType 和 valence
        atomid = int(match.group(1))  # 原子索引
        atomType = match.group(2)     # 原子类型
        valence = int(match.group(3)) # 当前价态
        return atomid, atomType, valence
    elif match2:
        atomid = int(match2.group(1))  # 原子索引
        atomType = match2.group(2)     # 原子类型
        # valence = int(match2.group(3)) # 当前价态
        return atomid, atomType, None
        
    else:
        raise ValueError("无法从错误信息中提取原子信息")
    
def calculate_charge_adjustment(atom_symbol, current_valence):
    """
    计算需要调整的电荷，根据反馈的原子符号和当前价态。
    :param atom_symbol: 原子符号（如 "C"）
    :param current_valence: 当前价态（如 5）
    :return: 需要添加的电荷数（正数表示负电荷，负数表示正电荷）
    """
    if atom_symbol not in VALENCES:
        raise ValueError(f"未知的原子符号: {atom_symbol}")

    # 查找该元素的最大价态
    max_valence = max(VALENCES[atom_symbol])
    if current_valence is None:
        current_valence=max_valence
    # 如果当前价态大于最大允许价态，需要调整电荷
    if current_valence > max_valence:
        # 需要添加的负电荷数
        charge_adjustment = current_valence - max_valence
        return charge_adjustment 
    else:
        # 当前价态已经符合最大允许价态，不需要调整
        return 0

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

################################################################################################################################################################
def output_to_smiles(output,idx_to_labels,bond_labels,result):#this will output * without abbre version
    #only output smiles with *
    x_center = (output["boxes"][:, 0] + output["boxes"][:, 2]) / 2
    y_center = (output["boxes"][:, 1] + output["boxes"][:, 3]) / 2

    center_coords = torch.stack((x_center, y_center), dim=1)

    output = {'bbox':         output["boxes"].to("cpu").numpy(),
                'bbox_centers': center_coords.to("cpu").numpy(),
                'scores':       output["scores"].to("cpu").numpy(),
                'pred_classes': output["labels"].to("cpu").numpy()}
    

    atoms_list, bonds_list,charge = bbox_to_graph_with_charge(output,
                                                idx_to_labels=idx_to_labels,
                                                bond_labels=bond_labels,
                                                result=result)
    smiles, mol= mol_from_graph_with_chiral(atoms_list, bonds_list,charge)
    abc=[atoms_list, bonds_list,charge ]
    
    if isinstance(smiles, type(None)):
        print(f"get atoms_list problems")
        # smiles, mol=None,None
    elif isinstance(atoms_list,type(None)):
        print(f"get atoms_list problems")
        # smiles, mol=None,None
    # else:
    #     smiles, mol=smiles_mol
    return abc,smiles,mol,output


def output_to_smiles2(output,idx_to_labels,bond_labels,result):#this will output * without abbre version
    #only output smiles with *
    x_center = (output["boxes"][:, 0] + output["boxes"][:, 2]) / 2
    y_center = (output["boxes"][:, 1] + output["boxes"][:, 3]) / 2

    center_coords = torch.stack((x_center, y_center), dim=1)

    output = {'bbox':         output["boxes"].to("cpu").numpy(),
                'bbox_centers': center_coords.to("cpu").numpy(),
                'scores':       output["scores"].to("cpu").numpy(),
                'pred_classes': output["labels"].to("cpu").numpy()}
    

    atoms_list, bonds_list,charge = bbox_to_graph_with_charge(output,
                                                idx_to_labels=idx_to_labels,
                                                bond_labels=bond_labels,
                                                result=result)
    smiles, mol= mol_from_graph_with_chiral(atoms_list, bonds_list,charge)
    abc=[atoms_list, bonds_list,charge ]
    if isinstance(smiles, type(None)):
        print(f"get atoms_list problems")
        # smiles, mol=None,None
    elif isinstance(atoms_list,type(None)):
        print(f"get atoms_list problems")
        # smiles, mol=None,None
    # else:
    #     smiles, mol=smiles_mol
    return abc,smiles,mol,output



def bbox_to_graph(output, idx_to_labels, bond_labels,result):
    
    # calculate atoms mask (pred classes that are atoms/bonds)
    atoms_mask = np.array([True if ins not in bond_labels else False for ins in output['pred_classes']])

    # get atom list
    atoms_list = [idx_to_labels[a] for a in output['pred_classes'][atoms_mask]]

    # if len(result) !=0 and 'other' in atoms_list:
    #     new_list = []
    #     replace_index = 0
    #     for item in atoms_list:
    #         if item == 'other':
    #             new_list.append(result[replace_index % len(result)])
    #             replace_index += 1
    #         else:
    #             new_list.append(item)
    #     atoms_list = new_list

    atoms_list = pd.DataFrame({'atom': atoms_list,
                            'x':    output['bbox_centers'][atoms_mask, 0],
                            'y':    output['bbox_centers'][atoms_mask, 1]})

    # in case atoms with sign gets detected two times, keep only the signed one
    for idx, row in atoms_list.iterrows():
        if row.atom[-1] != '0':
            if row.atom[-2] != '-':#assume charge value -9~9
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-1])]
            else:
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-2])]

            kdt = cKDTree(overlapping[['x', 'y']])
            dists, neighbours = kdt.query([row.x, row.y], k=2)
            if dists[1] < 7:
                atoms_list.drop(overlapping.index[neighbours[1]], axis=0, inplace=True)

    bonds_list = []

    # get bonds
    for bbox, bond_type, score in zip(output['bbox'][np.logical_not(atoms_mask)],
                                    output['pred_classes'][np.logical_not(atoms_mask)],
                                    output['scores'][np.logical_not(atoms_mask)]):
         
        # if idx_to_labels[bond_type] == 'SINGLE':
        if idx_to_labels[bond_type] in ['-','SINGLE', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
            _margin = 5
        else:
            _margin = 8

        # anchor positions are _margin distances away from the corners of the bbox.
        anchor_positions = (bbox + [_margin, _margin, -_margin, -_margin]).reshape([2, -1])
        oposite_anchor_positions = anchor_positions.copy()
        oposite_anchor_positions[:, 1] = oposite_anchor_positions[:, 1][::-1]

        # Upper left, lower right, lower left, upper right
        # 0 - 1, 2 - 3
        anchor_positions = np.concatenate([anchor_positions, oposite_anchor_positions])

        # get the closest point to every corner
        atoms_pos = atoms_list[['x', 'y']].values
        kdt = cKDTree(atoms_pos)
        dists, neighbours = kdt.query(anchor_positions, k=1)

        # check corner with the smallest total distance to closest atoms
        if np.argmin((dists[0] + dists[1], dists[2] + dists[3])) == 0:
            # visualize setup
            begin_idx, end_idx = neighbours[:2]
        else:
            # visualize setup
            begin_idx, end_idx = neighbours[2:]

        #NOTE  this proces may lead self-bonding for one atom
        if begin_idx != end_idx:# avoid self-bond
            bonds_list.append((begin_idx, end_idx, idx_to_labels[bond_type], idx_to_labels[bond_type], score))
        else:
            continue
    # return atoms_list.atom.values.tolist(), bonds_list
    return atoms_list, bonds_list


def calculate_distance(coord1, coord2):
    # Calculate Euclidean distance between two coordinates
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def assemble_atoms_with_charges(atom_list, charge_list):
    used_charge_indices=set()
    atom_list = atom_list.reset_index(drop=True)
    # atom_list['atom'] = atom_list['atom'] + '0'
    kdt = cKDTree(atom_list[['x','y']])
    for i, charge in charge_list.iterrows():
        if i in used_charge_indices:
            continue
        charge_=charge['charge']
        # if charge_=='1':charge_='+'
        dist, idx_atom=kdt.query([charge_list.x[i],charge_list.y[i]], k=1)
        # atom_str=atom_list.loc[idx_atom,'atom'] 
        if idx_atom not in atom_list.index:
            print(f"Warning: idx_atom {idx_atom} is out of range for atom_list.")
            continue  # 跳过当前循环迭代
        atom_str = atom_list.iloc[idx_atom]['atom']
        if atom_str=='*':
            atom_=atom_str + charge_
        else:
            try:
                atom_ = re.findall(r'[A-Za-z*]+', atom_str)[0] + charge_
            except Exception as e:
                print(atom_str,charge_,charge_list)
                print(f"@assemble_atoms_with_charges\n {e}\n{atom_list}")
                atom_=atom_str + charge_
        atom_list.loc[idx_atom,'atom']=atom_

    return atom_list



def assemble_atoms_with_charges2(atom_list, charge_list, max_distance=10):
    used_charge_indices = set()

    for idx, atom in atom_list.iterrows():
        atom_coord = atom['x'],atom['y']
        atom_label = atom['atom']
        closest_charge = None
        min_distance = float('inf')

        for i, charge in charge_list.iterrows():
            if i in used_charge_indices:
                continue

            charge_coord = charge['x'],charge['y']
            charge_label = charge['charge']

            distance = calculate_distance(atom_coord, charge_coord)
            #NOTE how t determin this max_distance, dependent on image size??
            if distance <= max_distance and distance < min_distance:
                closest_charge = charge
                min_distance = distance

        
        if closest_charge is not None:
            if closest_charge['charge'] == '1':
                charge_ = '+'
            else:
                charge_ = closest_charge['charge']
            atom_ = atom['atom'] + charge_

            # atom['atom'] = atom_
            atom_list.loc[idx,'atom'] = atom_
            used_charge_indices.add(tuple(charge))

        else:
            # atom['atom'] = atom['atom'] + '0'
            atom_list.loc[idx,'atom'] = atom['atom'] + '0'

    return atom_list

# 计算 IoU（交并比）
def calculate_iou(box1, box2):
    """
    计算两个框的 IoU。
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    # 计算 IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou



def bbox_to_graph_with_charge(output, idx_to_labels, bond_labels,result):
    
    bond_labels_pre=bond_labels
    # charge_labels = [18,19,20,21,22]#make influence
    atoms_mask = np.array([True if ins not in bond_labels and ins not in charge_labels else False for ins in output['pred_classes']])

    try:
        # print(atoms_mask.shape)
        # print(output['pred_classes'].shape)
        atoms_list = [idx_to_labels[a] for a in output['pred_classes'][atoms_mask]]
        if isinstance(atoms_list, pd.Series) and atoms_list.empty:
            return None, None, None
        else:
            atoms_list = pd.DataFrame({'atom': atoms_list,
                                    'x':    output['bbox_centers'][atoms_mask, 0],
                                    'y':    output['bbox_centers'][atoms_mask, 1],
                                    'bbox':  output['bbox'][atoms_mask].tolist() ,#need this for */other converting
                                    'scores': output['scores'][atoms_mask].tolist(),
                                    })
    except Exception as e:
        print(output['pred_classes'][atoms_mask].dtype,output['pred_classes'][atoms_mask])#int64 [ 1  1  1  1  1  2  1 29]
        print(e)
        print(idx_to_labels)
        # print(output['pred_classes'][atoms_mask],"output['pred_classes'][atoms_mask]")
    
        
        # confict_atompaire=[]
        # # 如果你想计算所有边界框之间的IOU，考虑2个原子box 重叠 是否要删掉一个？？ TODO gmy version most box larger then normal mix the rules
        # for i in range(len(atoms_list)):
        #     for j in range(i + 1, len(atoms_list)):
        #         iou_value = calculate_iou(atoms_list.bbox[i], atoms_list.bbox[j])
        #         if iou_value !=0:
        #             # print(f"IOU between box {i} and box {j}: {iou_value}")
        #             if i !=j : confict_atompaire.append([i,j])
        # if len(confict_atompaire)>0:
        #     need_del=[]
        #     for i,j in confict_atompaire:
        #         ij_lab=[atoms_list.loc[i].atom,atoms_list.loc[j].atom ]
        #         ij_score=[atoms_list.loc[i].scores,atoms_list.loc[j].scores]
        #         # print(ij_lab,ij_score)
        #         if ij_lab==['C','N'] or ij_lab==['N','C']:
        #             if atoms_list.loc[i].atom =='C':
        #                 need_del.append(i)
        #             else:
        #                 need_del.append(j)
                # elif atoms_list.loc[i].scores> atoms_list.loc[j].scores:
                #         need_del.append(j)
                # elif atoms_list.loc[j].scores> atoms_list.loc[i].scores:
                #         need_del.append(i)  
            # print(need_del)          
            # atoms_list= atoms_list.drop(need_del)

    charge_mask = np.array([True if ins in charge_labels else False for ins in output['pred_classes']])
    charge_list = [idx_to_labels[a] for a in output['pred_classes'][charge_mask]]
    charge_list = pd.DataFrame({'charge': charge_list,
                        'x':    output['bbox_centers'][charge_mask, 0],
                        'y':    output['bbox_centers'][charge_mask, 1],
                        'scores':    output['scores'][charge_mask],
                        
                        })
    
    # print(charge_list,'\n@bbox_to_graph_with_charge')
    try:
        atoms_list['atom'] = atoms_list['atom']+'0'#add 0 
    except Exception as e:
        print(e)
        print(atoms_list['atom'],'atoms_list["atom"] @@ adding 0 ')
        

    if len(charge_list) > 0:
        atoms_list = assemble_atoms_with_charges(atoms_list,charge_list)
    # else:#Note Most mols are not formal charged 
        # atoms_list['atom'] = atoms_list['atom']+'0'
    # print(atoms_list,"after @@assemble_atoms_with_charges ")
    
    # in case atoms with sign gets detected two times, keep only the signed one
    for idx, row in atoms_list.iterrows():
        if row.atom[-1] != '0':
            try:
                if row.atom[-2] != '-':#assume charge value -9~9
                    overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-1])]
            except Exception as e:
                print(row.atom,"@rin case atoms with sign gets detected two times")
                print(e)
            else:
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-2])]

            kdt = cKDTree(overlapping[['x', 'y']])
            dists, neighbours = kdt.query([row.x, row.y], k=2)
            if dists[1] < 7:
                atoms_list.drop(overlapping.index[neighbours[1]], axis=0, inplace=True)

    bonds_list = []
    # get bonds
    # bond_mask=np.logical_not(np.logical_not(atoms_mask) | np.logical_not(charge_mask))
    bond_mask=np.logical_not(atoms_mask) & np.logical_not(charge_mask)
    for bbox, bond_type, score in zip(output['bbox'][bond_mask],  #NOTE also including the charge part
                                    output['pred_classes'][bond_mask],
                                    output['scores'][bond_mask]):
         
        # if idx_to_labels[bond_type] == 'SINGLE':
        if len(idx_to_labels)==23:
            if idx_to_labels[bond_type] in ['-','SINGLE', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
                _margin = 5
            else:
                _margin = 8
        elif len(idx_to_labels)==30:
            _margin=0#ad this version bond dynamicaly changed
        # anchor positions are _margin distances away from the corners of the bbox.
        anchor_positions = (bbox + [_margin, _margin, -_margin, -_margin]).reshape([2, -1])
        oposite_anchor_positions = anchor_positions.copy()
        oposite_anchor_positions[:, 1] = oposite_anchor_positions[:, 1][::-1]

        # Upper left, lower right, lower left, upper right
        # 0 - 1, 2 - 3
        anchor_positions = np.concatenate([anchor_positions, oposite_anchor_positions])

        # get the closest point to every corner
        atoms_pos = atoms_list[['x', 'y']].values
        kdt = cKDTree(atoms_pos)
        dists, neighbours = kdt.query(anchor_positions, k=1)

        # check corner with the smallest total distance to closest atoms
        if np.argmin((dists[0] + dists[1], dists[2] + dists[3])) == 0:
            # visualize setup
            begin_idx, end_idx = neighbours[:2]
        else:
            # visualize setup
            begin_idx, end_idx = neighbours[2:]

        #NOTE  this proces may lead self-bonding for one atom
        if begin_idx != end_idx: 
            if bond_type in bond_labels:# avoid self-bond
                bonds_list.append((begin_idx, end_idx, idx_to_labels[bond_type], idx_to_labels[bond_type], score))
            else:
                print(f'this box may be charges box not bonds {[bbox, bond_type, score ]}')
        else:
            continue
    # return atoms_list.atom.values.tolist(), bonds_list
    # print(f"@box2graph: atom,bond nums:: {len(atoms_list)}, {len(bonds_list)}")
    return atoms_list, bonds_list,charge_list#dataframe, list

def parse_atom(node):
    s10 = [str(x) for x in range(10)]
    # Determine atom and formal charge
    if 'other' in node:
        a = '*'
        if '-' in node or '+' in node:
            fc = -1 if node[-1] == '-' else 1
        else:
            fc = int(node[-2:]) if node[-2:] in s10 else 0
    elif node[-1] in s10:
        if '-' in node or '+' in node:
            fc = -1 if node[-1] == '-' else 1
            a = node[:-1]
        else:
            a = node[:-1]
            fc = int(node[-1])
    elif node[-1] == '+':
        a = node[:-1]
        fc = 1
    elif node[-1] == '-':
        a = node[:-1]
        fc = -1
    else:
        a = node
        fc = 0
    return a, fc

def mol_from_graph_with_chiral(atoms_list, bonds,charge):

    mol = RWMol()
    nodes_idx = {}
    atoms = atoms_list.atom.values.tolist()
    coords = [(row['x'], 300-row['y'], 0) for index, row in atoms_list.iterrows()]#TODO  fix me with diff size img
    coords = tuple(coords)
    coords = tuple(tuple(num / 100 for num in sub_tuple) for sub_tuple in coords)

    for i in range(len(bonds)):
        idx_1, idx_2, bond_type, bond_dir, score = bonds[i]
        if bond_type in ['-', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
            bonds[i] = (idx_1, idx_2, 'SINGLE', bond_dir, score)
        elif bond_type == '=':
            bonds[i] = (idx_1, idx_2, 'DOUBLE', bond_dir, score)
        elif bond_type == '#':
            bonds[i] = (idx_1, idx_2, 'TRIPLE', bond_dir, score)

            

    bond_types = {'SINGLE':   Chem.rdchem.BondType.SINGLE,
                'DOUBLE':   Chem.rdchem.BondType.DOUBLE,
                'TRIPLE':   Chem.rdchem.BondType.TRIPLE,
                'AROMATIC': Chem.rdchem.BondType.AROMATIC}
    
    bond_dirs = {'NONE':    Chem.rdchem.BondDir.NONE,
            'ENDUPRIGHT':   Chem.rdchem.BondDir.ENDUPRIGHT,
            'BEGINWEDGE':   Chem.rdchem.BondDir.BEGINWEDGE,
            'BEGINDASH':    Chem.rdchem.BondDir.BEGINDASH,
            'ENDDOWNRIGHT': Chem.rdchem.BondDir.ENDDOWNRIGHT,}
    

    debug=True
    if debug:# try:
        placeholder_atoms = {}
        # add nodes
        s10=[str(x) for x in range(10)]
        for idx, node in enumerate(atoms):#NOTE  no formal charge will be X0 here
            # node=node.split(' ')
            # if ('0' in node) or ('1' in node):
            if 'other' in node:#30 idx_lab vesrion OH-->Cl with high 
                a='*'
                if '-' in node or '+' in node:
                    if node[-1] =='-':#in ['-','+']:
                        fc = -1
                    elif [-1] =='+':
                        fc = 1
                    else:      
                        fc = int(node[-2:])
                else:
                    fc = int(node[-1])
            elif node[-1] in s10:
                if '-' in node or '+' in node:
                    if node[-1] =='-':#in ['-','+']:
                        fc = -1
                        a = node[:-1]
                    elif [-1] =='+':
                        fc = 1
                        a = node[:-1]
                    else:      
                        fc = int(node[-2:])
                        a = node[:-2]
                else:
                    a = node[:-1]
                    fc = int(node[-1])
            elif node[-1]=='+':
                a = node[:-1]
                fc = 1
            elif  node[-1]=='-':
                a = node[:-1]
                fc = -1
            # elif ('-1' in node) or ('-' in node):
            #     a = node[:-2]
            #     fc = int(node[-2])
            else:
                a = node
                fc = 0
            #if a in ELEMENTS+['*']:
            if a in ['H', 'C', 'O', 'N', 'Cl', 'Br', 'S', 'F', 'B', 'I', 'P', 'Si']:#  '*', I2M's defined atom types
                ad = Chem.Atom(a)#TODO consider non chemical group and label for using
            elif a in ABBREVIATIONS:
                # Replace with the corresponding RDKit molecule object from ABBREVIATIONS
                smi = ABBREVIATIONS[a].smiles
                # submol = Chem.MolFromSmiles(smi)
                # ad = submol.GetAtomWithIdx(0)  # We get the first atom (usually the central one)
                ad = Chem.Atom("*")
                placeholder_atoms[idx] = a # 记录非标准原但有定义的官能团   类型及其位置,
            else:
                ad = Chem.Atom("*")
                # placeholder_atoms[idx] = a  

            ad.SetFormalCharge(fc)
            atom_idx = mol.AddAtom(ad)
            nodes_idx[idx] = atom_idx

        # add bonds
        existing_bonds = set()
        for idx_1, idx_2, bond_type, bond_dir, score in bonds:
            if (idx_1 in nodes_idx) and (idx_2 in nodes_idx):
                if (idx_1, idx_2) not in existing_bonds and (idx_2, idx_1) not in existing_bonds:
                    try:
                        mol.AddBond(nodes_idx[idx_1], nodes_idx[idx_2], bond_types[bond_type])
                    except Exception as e:
                        print([idx_1, idx_2, bond_type, bond_dir, score],f"erro @add bonds ")
                        print(f"erro@add existing_bonds: {e}\n{bonds}")
                        continue
            existing_bonds.add((idx_1, idx_2))
            if Chem.MolFromSmiles(Chem.MolToSmiles(mol.GetMol())):
                prev_mol = copy.deepcopy(mol)
            else:
                mol = copy.deepcopy(prev_mol)

        # # **第三步: 替换 * 并合并官能团** add in new function
        # mol,smi= expandABB(mol,ABBREVIATIONS, placeholder_atoms)
        
        chiral_centers = Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
        chiral_center_ids = [idx for idx, _ in chiral_centers] 

        for id in chiral_center_ids:
            for index, tup in enumerate(bonds):
                if id == tup[1]:
                    new_tup = tuple([tup[1], tup[0], tup[2], tup[3], tup[4]])#idx_1, idx_2, bond_type, bond_dir, score
                    bonds[index] = new_tup
                    mol.RemoveBond(int(tup[0]), int(tup[1]))
                    try:
                        mol.AddBond(int(tup[1]), int(tup[0]), bond_types[tup[2]])
                    except Exception as e:
                        print( index, tup, id)
                        print(f"bonds: {bonds}")
                        print(f"erro@chiral_center_ids: {e}")
        mol = mol.GetMol()

        # if 'S0' in atoms:
        #     bonds_ = [[row[0], row[1], row[3]] for row in bonds]

        #     n_atoms=len(atoms)
        #     for i in chiral_center_ids:
        #         for j in range(n_atoms):

        #             if [i,j,'BEGINWEDGE'] in bonds_:
        #                 mol.GetBondBetweenAtoms(i, j).SetBondDir(bond_dirs['BEGINWEDGE'])
        #             elif [i,j,'BEGINDASH'] in bonds_:
        #                 mol.GetBondBetweenAtoms(i, j).SetBondDir(bond_dirs['BEGINDASH'])

        #     Chem.SanitizeMol(mol)
        #     AllChem.Compute2DCoords(mol)
        #     Chem.AssignChiralTypesFromBondDirs(mol)
        #     Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

        # else:
        mol.RemoveAllConformers()
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.Set3D(True)
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, (x, y, z))
        mol.AddConformer(conf)
        # Chem.SanitizeMol(mol)
        Chem.AssignStereochemistryFrom3D(mol)

        bonds_ = [[row[0], row[1], row[3]] for row in bonds]

        n_atoms=len(atoms)
        for i in chiral_center_ids:
            for j in range(n_atoms):
                b_=mol.GetBondBetweenAtoms(i, j)
                if [i,j,'BEGINWEDGE'] in bonds_ and b_:
                    b_.SetBondDir(bond_dirs['BEGINWEDGE'])
                elif [i,j,'BEGINDASH'] in bonds_ and b_:
                    b_.SetBondDir(bond_dirs['BEGINDASH'])
        try:
            Chem.SanitizeMol(mol)#NOTE if mol is not rdkit standar will cause the rdkit erros
        except Exception as e:
            problems = Chem.DetectChemistryProblems(mol)
            if len(problems)>0:
                print(F"get problems",len(problems))
                for p in problems:
                    # print(f"try to fixing!!!!!!!! {p.Message()}")
                    # print("GetType",p.GetType())
                    # print("IDX",p.GetAtomIdx())
                    atomid, atomType, valence=Val_extract_atom_info(p.Message())
                    charge_adjustment=calculate_charge_adjustment(atomType, valence)
                    mol.GetAtomWithIdx(atomid).SetFormalCharge(charge_adjustment)
            # Chem.SanitizeMol(mol)#as the valence is right now but still get errors
            
        Chem.DetectBondStereochemistry(mol)
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol)
        # mol.Debug()
        # print('debuged')
        
        # drawing out
        # opts = Draw.MolDrawOptions()
        # opts.addAtomIndices = False
        # opts.addStereoAnnotation = False
        # img = Draw.MolToImage(mol, options=opts,size=(1000, 1000))
        # img.save('tttttttttttttafter.png')
        # Chem.Draw.MolToImageFile(mol, 'tttttttttttttbefore.png')
        # img.save('/home/jovyan/rt-detr/output/test/after.png')
        # Chem.Draw.MolToImageFile(mol, '/home/jovyan/rt-detr/output/test/before.png')
        smiles=Chem.MolToSmiles(mol)
        return smiles,mol
    else:
        try:
            placeholder_atoms = {}
            # add nodes
            s10=[str(x) for x in range(10)]
            for idx, node in enumerate(atoms):#NOTE  no formal charge will be X0 here
                # node=node.split(' ')
                # if ('0' in node) or ('1' in node):
                if 'other' in node:
                    a='*'
                    if '-' in node or '+' in node:
                        if node[-1] =='-':#in ['-','+']:
                            fc = -1
                        elif [-1] =='+':
                            fc = 1
                        else:      
                            fc = int(node[-2:])
                    else:
                        fc = int(node[-1])
                elif node[-1] in s10:
                    if '-' in node or '+' in node:
                        if node[-1] =='-':#in ['-','+']:
                            fc = -1
                            a = node[:-1]
                        elif [-1] =='+':
                            fc = 1
                            a = node[:-1]
                        else:      
                            fc = int(node[-2:])
                            a = node[:-2]
                    else:
                        a = node[:-1]
                        fc = int(node[-1])
                elif node[-1]=='+':
                    a = node[:-1]
                    fc = 1
                elif  node[-1]=='-':
                    a = node[:-1]
                    fc = -1
                # elif ('-1' in node) or ('-' in node):
                #     a = node[:-2]
                #     fc = int(node[-2])
                else:
                    a = node
                    fc = 0
                #if a in ELEMENTS+['*']:
                if a in ['H', 'C', 'O', 'N', 'Cl', 'Br', 'S', 'F', 'B', 'I', 'P', 'Si']:#  '*', I2M's defined atom types
                    ad = Chem.Atom(a)#TODO consider non chemical group and label for using
                elif a in ABBREVIATIONS:
                    # Replace with the corresponding RDKit molecule object from ABBREVIATIONS
                    smi = ABBREVIATIONS[a].smiles
                    submol = Chem.MolFromSmiles(smi)
                    # ad = submol.GetAtomWithIdx(0)  # We get the first atom (usually the central one)
                    ad = Chem.Atom("*")
                    placeholder_atoms[idx] = a # 记录非标准原但有定义的官能团   类型及其位置,
                else:
                    ad = Chem.Atom("*")
                    # placeholder_atoms[idx] = a  

                ad.SetFormalCharge(fc)
                atom_idx = mol.AddAtom(ad)
                nodes_idx[idx] = atom_idx
            # add bonds
            existing_bonds = set()#NOTE important with digui
            for idx_1, idx_2, bond_type, bond_dir, score in bonds:
                if (idx_1 in nodes_idx) and (idx_2 in nodes_idx):
                    if (idx_1, idx_2) not in existing_bonds and (idx_2, idx_1) not in existing_bonds:
                        try:
                            mol.AddBond(nodes_idx[idx_1], nodes_idx[idx_2], bond_types[bond_type])
                        except Exception as e:
                            print([idx_1, idx_2, bond_type, bond_dir, score],f"erro @add bonds ")
                            print(f"erro@add existing_bonds: {e}\n{bonds}")
                            continue
                existing_bonds.add((idx_1, idx_2))
                if Chem.MolFromSmiles(Chem.MolToSmiles(mol.GetMol())):
                    prev_mol = copy.deepcopy(mol)
                else:
                    mol = copy.deepcopy(prev_mol)
            # # **第三步: 替换 * 并合并官能团** add in new function
            # mol,smi= expandABB(mol,ABBREVIATIONS, placeholder_atoms)
            
            chiral_centers = Chem.FindMolChiralCenters(
                mol, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
            chiral_center_ids = [idx for idx, _ in chiral_centers] 

            for id in chiral_center_ids:
                for index, tup in enumerate(bonds):
                    if id == tup[1]:
                        new_tup = tuple([tup[1], tup[0], tup[2], tup[3], tup[4]])#idx_1, idx_2, bond_type, bond_dir, score
                        bonds[index] = new_tup
                        mol.RemoveBond(int(tup[0]), int(tup[1]))
                        try:
                            mol.AddBond(int(tup[1]), int(tup[0]), bond_types[tup[2]])
                        except Exception as e:
                            print( index, tup, id)
                            print(f"bonds: {bonds}")
                            print(f"erro@chiral_center_ids: {e}")
            mol = mol.GetMol()

            mol.RemoveAllConformers()
            conf = Chem.Conformer(mol.GetNumAtoms())
            conf.Set3D(True)
            for i, (x, y, z) in enumerate(coords):
                conf.SetAtomPosition(i, (x, y, z))
            mol.AddConformer(conf)
            # Chem.SanitizeMol(mol)
            Chem.AssignStereochemistryFrom3D(mol)

            bonds_ = [[row[0], row[1], row[3]] for row in bonds]

            n_atoms=len(atoms)
            for i in chiral_center_ids:
                for j in range(n_atoms):
                    if [i,j,'BEGINWEDGE'] in bonds_:
                        mol.GetBondBetweenAtoms(i, j).SetBondDir(bond_dirs['BEGINWEDGE'])
                    elif [i,j,'BEGINDASH'] in bonds_:
                        mol.GetBondBetweenAtoms(i, j).SetBondDir(bond_dirs['BEGINDASH'])
            Chem.SanitizeMol(mol)
            Chem.DetectBondStereochemistry(mol)
            Chem.AssignChiralTypesFromBondDirs(mol)
            Chem.AssignStereochemistry(mol)
            smiles=Chem.MolToSmiles(mol)
            return smiles,mol
        #if debug comment this block of except
        except Chem.rdchem.AtomValenceException as e:
            print(f"捕获到 AtomValenceException 异常@@\n{e}")
            print(atoms,f"idx:{idx},atoms[idx]::{atoms[idx]}")
            # print(atoms_list, bonds,charge)
            return None,mol

        except Exception as e:
            print(f"捕获到   异常@@{e}")
            print(f"Error@@node {node} atom@@ {a} \n")
            print(atoms,f"idx:{idx},atoms[idx]::{atoms[idx]}")
            print("bonds:",bonds)
            print("charge:",charge)
            # print(atoms_list, bonds,charge)
            #TODO fix this here to improve the original real data
            # Error@@node Me0 atom@@ Me 
            # ['N0', 'O0', 'O0', 'Me0', 'O0', 'C0', 'C0', 'Ph0', 'C0', 'C0', 'C0'] idx:3,atoms[idx]::Me0
            # Element 'Me' not found
            # Violation occurred on line 93 in file Code/GraphMol/PeriodicTable.h
            return None,mol



def mol_from_graph_without_chiral(atoms, bonds):

    mol = RWMol()
    nodes_idx = {}

    for i in range(len(bonds)):
        idx_1, idx_2, bond_type, bond_dir, score = bonds[i]
        if bond_type in  ['-', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
            bonds[i] = (idx_1, idx_2, 'SINGLE', bond_dir, score)
        elif bond_type == '=':
            bonds[i] = (idx_1, idx_2, 'DOUBLE', bond_dir, score)
        elif bond_type == '#':
            bonds[i] = (idx_1, idx_2, 'TRIPLE', bond_dir, score)
  

    bond_types = {'SINGLE':   Chem.rdchem.BondType.SINGLE,
                'DOUBLE':   Chem.rdchem.BondType.DOUBLE,
                'TRIPLE':   Chem.rdchem.BondType.TRIPLE,
                'AROMATIC': Chem.rdchem.BondType.AROMATIC}

        
    try:
        # add nodes
        for idx, node in enumerate(atoms):
            if ('0' in node) or ('1' in node):
                a = node[:-1]
                fc = int(node[-1])
            if '-1' in node:
                a = node[:-2]
                fc = -1

            a = Chem.Atom(a)
            a.SetFormalCharge(fc)

            atom_idx = mol.AddAtom(a)
            nodes_idx[idx] = atom_idx

        # add bonds
        existing_bonds = set()#usd only in for loop
        for idx_1, idx_2, bond_type, bond_dir, score in bonds:
            if (idx_1 in nodes_idx) and (idx_2 in nodes_idx):
                if (idx_1, idx_2) not in existing_bonds and (idx_2, idx_1) not in existing_bonds:
                    try:
                        mol.AddBond(nodes_idx[idx_1], nodes_idx[idx_2], bond_types[bond_type])
                    except:
                        continue#failed will not included into existing_bonds
            existing_bonds.add((idx_1, idx_2))
            if Chem.MolFromSmiles(Chem.MolToSmiles(mol.GetMol())):
                prev_mol = copy.deepcopy(mol)
            else:
                mol = copy.deepcopy(prev_mol)

        mol = mol.GetMol()
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        return Chem.MolToSmiles(mol)

    except Chem.rdchem.AtomValenceException as e:
        print("捕获到 AtomValenceException 异常")





