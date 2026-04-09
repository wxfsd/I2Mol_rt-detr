"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by bowen
"""

import json
import math
import os
import sys
import pathlib
from typing import Iterable
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.amp 
from src.data import CocoEvaluator
from PIL import Image
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)
from src.solver.utils import output_to_smiles, output_to_smiles2
from src.solver.utils import bbox_to_graph_with_charge, mol_from_graph_with_chiral

from src.misc.draw_box_utils import draw_objs
from sklearn.metrics import f1_score
# from src.postprocess.abbreviation_detector import get_ocr_recognition_only
# from src.postprocess.utils_dataset import CaptionRemover
from skimage.measure import label
######################################add metric postprocess
import rdkit 
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from typing import List
from rdkit.Chem import rdchem, RWMol, CombineMols
from rdkit import Chem
from rdkit.Chem import rdFMCS
import copy
from paddleocr import PaddleOCR
import re
from rdkit import DataStructs


def MCS_mol(mcs):
    #mcs_smart = mcs.smartsString
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    AllChem.Compute2DCoords(mcs_mol)
    return mcs_mol

def g_atompair_matches(pair,mcs):
    mcs_mol = MCS_mol(mcs)
    matches0 = pair[0].GetSubstructMatches(mcs_mol, useQueryQueryMatches=True,uniquify=False, maxMatches=1000, useChirality=False)
    matches1 = pair[1].GetSubstructMatches(mcs_mol, useQueryQueryMatches=True,uniquify=False, maxMatches=1000, useChirality=False)
    if len(matches0) != len(matches1):
        matches0=list(matches0)
        matches1=list(matches1)
        print( "noted: matcher not equal !!")
        if len(matches0)>len(matches1) and len(matches1) !=0:
            for i in range(0,len(matches0)):
                if i < len(matches1):
                    pass
                else:
                    ii=i % len(matches1)
                    matches1.append(matches1[ii])
        else:
            for i in range(0,len(matches1)):
                if i < len(matches0) and len(matches0):
                    pass
                else:
                    ii=i % len(matches0)
                    matches0.append(matches0[ii])
    # assert len(matches0) == len(matches1), "matcher not equal break!!"
    if len(matches0) != len(matches1):
        atommaping_pairs=[[]]
    else:atommaping_pairs=[list(zip(matches0[i],matches1[i])) for i in range(0,len(matches0))]
    return atommaping_pairs


class CustomError(Exception):
    """A custom exception for specific errors."""
    pass

bond_dirs = {'NONE':    Chem.rdchem.BondDir.NONE,
                'ENDUPRIGHT':   Chem.rdchem.BondDir.ENDUPRIGHT,
                'BEGINWEDGE':   Chem.rdchem.BondDir.BEGINWEDGE,
                'BEGINDASH':    Chem.rdchem.BondDir.BEGINDASH,
            'ENDDOWNRIGHT': Chem.rdchem.BondDir.ENDDOWNRIGHT,}

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
    elif abbrev in RGROUP_SYMBOLS or (abbrev[0] == 'R' and abbrev[1:].isdigit()):
        if abbrev[1:].isdigit():
            return f'[{abbrev[1:]}*]'
    elif abbrev in ELEMENTS:#ocr tool need this
        return f'[{abbrev}]'
    match = re.match(r'^(\d+)?(.*)', abbrev)
    if match:
        numeric_part, remaining_part = match.groups()
        if remaining_part in ELEMENTS:
            return f'[{abbrev}]'
        else:
            if numeric_part:
                abbrev=f'[{numeric_part}*]'
    return '[*]'



def expandABB(mol,ABBREVIATIONS, placeholder_atoms):
    mols = [mol]
    # **第三步: 替换 * 并合并官能团**
    # 逆序遍历 placeholder_atoms，确保删除后不会影响后续索引
    for idx in sorted(placeholder_atoms.keys(), reverse=True):
        group = placeholder_atoms[idx]  # 获取官能团名称
        # print(idx, group)
        group=_expand_abbreviation(group)
        submol = Chem.MolFromSmiles(group)  # 获取官能团的子分子
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
    # 输出修改后的分子 SMILES
    modified_smiles = Chem.MolToSmiles(mols[-1])
    # print(f"修改后的分子 SMILES: {modified_smiles}")            
    return mols[-1], modified_smiles
############################################################################################################################################################
#molscrbe evaluate
from SmilesPE.pretokenizer import atomwise_tokenizer
def canonicalize_smiles(smiles, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True):
    if type(smiles) is not str or smiles == '':
        return '', False
    if ignore_cistrans:
        smiles = smiles.replace('/', '').replace('\\', '')
    if replace_rgroup:
        tokens = atomwise_tokenizer(smiles)
        for j, token in enumerate(tokens):
            if token[0] == '[' and token[-1] == ']':
                symbol = token[1:-1]
                if symbol[0] == 'R' and symbol[1:].isdigit():
                    tokens[j] = f'[{symbol[1:]}*]'
                elif Chem.AtomFromSmiles(token) is None:
                    tokens[j] = '*'
        smiles = ''.join(tokens)
    try:
        canon_smiles = Chem.CanonSmiles(smiles, useChiral=(not ignore_chiral))
        success = True
    except:
        canon_smiles = smiles
        success = False
    return canon_smiles, success


############################################################################################################################################################
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir,
    annot_file=f'/home/jovyan/rt-detr/data/real_processed/CLEF_with_charge/annotations/val.json',
    outcsv_filename=f'/home/jovyan/rt-detr/rt-detr/output/output_charge_CLEF.csv',
    ):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None
    
    # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # home='/home/jovyan/rt-detr'
    # dataset = 'CLEF'
    # annot_file=f'/home/jovyan/rt-detr/data/real_processed/{dataset}_with_charge/annotations/test.json'
    # outcsv_filename/home/jovyan/rt-detr/rt-detr/output/output_charge_{dataset}.csv'


    # annot_file=f'/home/jovyan/rt-detr/data/real_processed/{dataset}_with_charge/annotations/test.json'
    # outcsv_filename=f'/home/jovyan/rt-detr/rt-detr/output/output_charge_{dataset}.csv'
    with open(annot_file, 'r') as file: 
        data = json.load(file)




    image_id_to_name = {}

    for image_data in data['images']:
        image_id = image_data['id']
        image_path = image_data['file_name']
        image_name = os.path.basename(image_path)
        image_id_to_name[image_id] = image_name

    res_smiles = []

    # # bond_labels = [16,17,18,19,20,21,22]
    # # idx_to_labels = {0:'other',1: 'C0', 2: 'O0', 3: 'N0', 4: 'Cl0', 5: 'C-1', 
    # #              6: 'Br0', 7: 'N1', 8: 'O-1', 9: 'S0', 10: 'F0', 11: 'B0', 
    # #              12: 'I0', 13: 'P0', 14: '*0', 15: 'Si0', 16: 'NONE', 
    # #              17: 'ENDUPRIGHT', 18: 'BEGINWEDGE', 19: 'BEGINDASH', 
    # #              20: 'ENDDOWNRIGHT', 21: '=', 22: '#'}

    #NOTE comment based on prepared data----training----
    # # bond_labels = [19,20,21,22,23]
    # # idx_to_labels = {0:'other',1:'H0',2:'C0',3:'O0',4:'N0',5:'Cl0',6:'N1',7:'N-1',
    # #                 8:'C-1',9:'Ar0',10:'O-1',11:'Br0',12:'*0',13:'S0',14:'F0',15:'B0',
    # #                 16:'I0',17:'P0',18:'Si0',19:'NONE',20:'BEGINWEDGE',21:'BEGINDASH',
    # #                 22:'=',23:'#',}
    idx_to_labels23={0:'other',1:'C',2:'O',3:'N',4:'Cl',5:'Br',6:'S',7:'F',8:'B',
                        9:'I',10:'P',11:'*',12:'Si',13:'NONE',14:'BEGINWEDGE',15:'BEGINDASH',
                        16:'=',17:'#',18:'-4',19:'-2',20:'-1',21:'1',22:'2',} 
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
    bond_labels = [13,14,15,16,17]
    # if len(data["categories"])==23:
    if postprocessors.num_classes==23:
        print(data["categories"])
        print(f'usage idx_to_labels23',idx_to_labels23)
        idx_to_labels = idx_to_labels23
                    #NONE is single ?
                    # {"other": 0, "C": 1, "O": 2, "N": 3, "Cl": 4, "Br": 5, "S": 6, "F": 7, "B": 8, "I": 9,
                    #  "P": 10, "*": 11, "Si": 12, "NONE": 13, "BEGINWEDGE": 14, "BEGINDASH": 15, "=": 16,
                    #  "#": 17, "-4": 18, "-2": 19, "-1": 20, "1": 21, "+2": 22}
    elif postprocessors.num_classes==30:
        print(data["categories"])#NOTE 11 is H not * now
        print(f'usage idx_to_labels30',idx_to_labels30)
        idx_to_labels = idx_to_labels30
        # ABBREVIATIONS 
        abrevie={"[23*]":'CF3',
                                    "[24*]":'CN',
                                    "[25*]":'Me',
                                    "[26*]":'CO2Et',
                                    "[27*]":'R',
                                    "[28*]":'Ph',
                                    "[29*]":'3~7UP',
        }
    else:
        print('use custom vocable??!! please add here@@evaluate@det_engine.py in solver directory!!')
        print(len(data["categories"]),'loaded json data not mach will use defualt idx_to_labels')
        idx_to_labels = idx_to_labels30
        # idx_to_labels = idx_to_labels23

    smiles_data = pd.DataFrame({'file_name': [],
                                'SMILES':[]})
    
    output_dict = {}
    target_dict = {}
    filtered_output_dict = {}
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)#RTDETRPostProcessor@@src/zoo/rtertr
        #results: a list of dict  label box score
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        for target, output in zip(targets, results):
            output_dict[target['image_id'].item()] = output

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        # for target in targets:
        #     predict_boxes = target['boxes'].to("cpu").numpy()
        #     predict_classes = target['labels'].to("cpu").numpy()
        #     # predict_scores = target['scores'].to("cpu").numpy()
        #     predict_scores = np.ones(predict_classes.size)
        #     img = Image.open('/home/jovyan/data/real_processed/CLEF_with_charge/images/test/{}'.format(image_id_to_name[target['image_id'].item()])).convert('RGB')
        #     img = img.resize((1000,1000))
        #     predict_boxes = predict_boxes*100/64
        #     plot_img = draw_objs(img,
        #                 predict_boxes,
        #                 predict_classes,
        #                 predict_scores,
        #                 category_index=idx_to_labels,
        #                 box_thresh=0.5,
        #                 line_thickness=1,
        #                 font='arial.ttf',
        #                 font_size=10)
            
        #     plot_img.save('/home/jovyan/rt-detr/output/test_charge_large_label_CLEF/{}'.format(image_id_to_name[target['image_id'].item()]))
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
        # for target, output in zip(targets, results):
        #     target_dict[target['image_id'].item()] = target

        #     predict_boxes = output['boxes'].to("cpu").numpy()
        #     predict_classes = output['labels'].to("cpu").numpy()
        #     predict_scores = output['scores'].to("cpu").numpy()
        #     img = Image.open('/home/jovyan/data/real_processed/CLEF/images/test/{}'.format(image_id_to_name[target['image_id'].item()])).convert('RGB')
        #     img = img.resize((1000,1000))
        #     predict_boxes = predict_boxes*10/3
        #     plot_img = draw_objs(img,
        #                 predict_boxes,
        #                 predict_classes,
        #                 predict_scores,
        #                 category_index=idx_to_labels,
        #                 box_thresh=0.5,
        #                 line_thickness=1,
        #                 font='arial.ttf',
        #                 font_size=10)
            
        #     plot_img.save('/home/jovyan/rt-detr/output/test_charge_large_pred_CLEF/{}'.format(image_id_to_name[target['image_id'].item()]))
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        
    #     if coco_evaluator is not None:
    #         coco_evaluator.update(res)


    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # if coco_evaluator is not None:
    #     coco_evaluator.synchronize_between_processes()
    # if panoptic_evaluator is not None:
    #     panoptic_evaluator.synchronize_between_processes()

    # # accumulate predictions from all images
    # if coco_evaluator is not None:
    #     coco_evaluator.accumulate()
    #     coco_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            # stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()



    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
    # ocr_recognition_only = get_ocr_recognition_only(force_cpu=False)   
    # caption_remover = CaptionRemover(force_cpu=True)
    for key, value in output_dict.items():
        selected_indices = value['scores'] > 0.5#may be >=0.5 cut off, as used the sigmoid?
        if value['labels'][selected_indices].size(0) != 0:#no good prediction
            filtered_output_dict[key] = {
                'labels': value['labels'][selected_indices],# may be selected_indices ==0 as all small than0.5
                'boxes': value['boxes'][selected_indices],
                'scores': value['scores'][selected_indices]
            }
        else:
            ima_name=image_id_to_name[key]
            print(key,"all prediction scores small 0.5!!",len(output_dict),f"{ima_name}")##
        #     target_dict[key] = target
        #     predict_boxes = value['boxes'].to("cpu").numpy()
        #     predict_classes = value['labels'].to("cpu").numpy()
        #     predict_scores = value['scores'].to("cpu").numpy()
        #     img = Image.open(f'{annot_file}.boxedPred.png'.format(image_id_to_name[target['image_id'].item()])).convert('RGB')
        # #     img = img.resize((1000,1000))
        #     predict_boxes = predict_boxes*10/3
        #     plot_img = draw_objs(img,
        #                 predict_boxes,
        #                 predict_classes,
        #                 predict_scores,
        #                 category_index=idx_to_labels,
        #                 box_thresh=0.5,
        #                 line_thickness=1,
        #                 font='arial.ttf',
        #                 font_size=10)

    for i,(key,value) in enumerate(filtered_output_dict.items()):
        result = []
        smi_mol=output_to_smiles(value,idx_to_labels,bond_labels,result)#TODO use the idx_to_labels numer to if --else
        if smi_mol:
            res_smiles.append(smi_mol[0])  #TODO check this erro other0
        else:
            res_smiles.append('')
            
        new_row = {'file_name':image_id_to_name[key], 'SMILES':res_smiles[i]}
        smiles_data = smiles_data._append(new_row, ignore_index=True)
    
    print(f"will save {len(smiles_data)} dataframe into csv") 
    smiles_data.to_csv(outcsv_filename, index=False)

    # for i,(key,value) in enumerate(filtered_output_dict.items()):
    #     result = []
    #     for j, label in enumerate(filtered_output_dict[key]["labels"]):
    #         if label == 0:
    #             bbox = filtered_output_dict[key]["boxes"][j].cpu().numpy()
    #             image = Image.open('/home/jovyan/data/real_processed/merge_with_chiral_resample/images/val/{}'.format(image_id_to_name[key])).convert('RGB')
    #             cropped_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3])).convert('L')
    #             # cropped_image.save(f"cropped_bbox_{image_id_to_name[key]}_{i}.jpg")
    #             cropped_image = caption_remover(cropped_image)
    #             cropped_image = np.array(cropped_image)
    #             # label(cropped_image,connectivity = 1, return_num=True)
    #             result.append(ocr_recognition_only.ocr(cropped_image, det = False, rec = True, cls = False))
    #     result = [tup[0]+'0' for sublist in result for tup in sublist[0]]
    #     print(result)
    #     res_smiles.append(output_to_smiles(value,idx_to_labels,bond_labels,result))
    #     new_row = {'file_name':image_id_to_name[key],
    #                 'SMILES':res_smiles[i]}
    #     smiles_data = smiles_data._append(new_row, ignore_index=True)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return stats, coco_evaluator


#######################################################################################
def molExpanding(mol_rebuit,placeholder_atoms,wdbs,bond_dirs):
    cm=copy.deepcopy(mol_rebuit)
    # print(placeholder_atoms)
    expand_mol, expand_smiles= expandABB(cm,ABBREVIATIONS, placeholder_atoms)
    rdm=copy.deepcopy(expand_mol)
    AllChem.Compute2DCoords(rdm)
    target_mol, ref_mol=rdm, cm
    mcs=rdFMCS.FindMCS([target_mol, ref_mol], # larger,small order
                    atomCompare=rdFMCS.AtomCompare.CompareAny,
                    # bondCompare=rdFMCS.BondCompare.CompareAny,
                    ringCompare=rdFMCS.RingCompare.IgnoreRingFusion,
                    matchChiralTag=False,
    )
    atommaping_pairs=g_atompair_matches([target_mol, ref_mol],mcs)
    atomMap=atommaping_pairs[0]
    try:
        rmsd2=rdkit.Chem.rdMolAlign.AlignMol(prbMol=target_mol, refMol=ref_mol, atomMap=atomMap,maxIters=2000000)
    except Exception as e:
        print(atomMap,"@@@@")
        print(e)
    #after get atomMap
    c2p={cur:pre for cur, pre in atomMap}
    p2c={pre:cur for cur, pre in atomMap}
    for b in wdbs:#add bond direction
        p0,p1=int(b[0]), int(b[1])#may be not in the atomMap as the mcs_sub
        if p0 in p2c.keys() and p1 in p2c.keys():
            c0,c1=p2c[p0],p2c[p1]
            # print("[pre0,pre1]vs[c0,c1]current atom id",[p0,p1],[c0,c1])
            b_=target_mol.GetBondBetweenAtoms(c0,c1)
            if b_:
                b_.SetBondDir(bond_dirs[b[3]])
    expandStero_smi=Chem.MolToSmiles(target_mol)#directly will not add the stero info into smiles, must have the assing steps
    m=target_mol.GetMol()
    # Chem.SanitizeMol(m)
    Chem.DetectBondStereochemistry(m)
    Chem.AssignChiralTypesFromBondDirs(m)
    Chem.AssignStereochemistry(m)#expandStero_smi ,  m 
    return expandStero_smi, m  












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

def rdkit_canonicalize_smiles(smiles):
    tokens = atomwise_tokenizer(smiles)
    for j, token in enumerate(tokens):
        if token[0] == '[' and token[-1] == ']':
            symbol = token[1:-1]
            if symbol[0] == 'R' and symbol[1:].isdigit():
                # tokens[j] = f'[{symbol[1:]}*]'
                tokens[j] = '*'
            elif Chem.AtomFromSmiles(token) is None:
                tokens[j] = '*'
    smiles = ''.join(tokens)
    try:
        canon_smiles = Chem.CanonSmiles(smiles, useChiral=False)
        success = True
    except:
        canon_smiles = smiles
        success = False
    return canon_smiles, success


def comparing_smiles(new_row,test_smiles):#I2M use the coordinates, so 2D coformation should be always
    original_smiles=new_row['SMILESori']
    original_smiles = remove_backslash_and_slash(original_smiles)#c/s 
    test_smiles = remove_backslash_and_slash(test_smiles)
    original_smiles = re.sub(r'\[(\d+)\*', '[*',original_smiles)#[1*]-->[*]
    test_smiles = re.sub(r'\[(\d+)\*', '[*',test_smiles)
    original_smiles = remove_SP(original_smiles)#additional complex space stero from coordinates, most not used
    test_smiles = remove_SP(test_smiles)
    rd_smi_ori, success1=rdkit_canonicalize_smiles(original_smiles)
    rd_smi, success2=rdkit_canonicalize_smiles(test_smiles)
    original_smiles,test_smiles=rd_smi_ori,rd_smi
    try:
        original_mol = Chem.MolFromSmiles(original_smiles)#considering whe nmmet abbrev
        test_mol = Chem.MolFromSmiles(test_smiles,sanitize=False)#as build mol may not sanitized for rdkit
        if original_mol:
            Chem.SanitizeMol(original_mol)
            keku_smi_ori=Chem.MolToSmiles(original_mol,kekuleSmiles=True)
        else:
            keku_smi_ori=original_smiles
        
        if test_mol:
            Chem.SanitizeMol(test_mol)
            keku_smi=Chem.MolToSmiles(test_mol,kekuleSmiles=True)
        else:
            keku_smi=test_smiles
            
        if '*' not in keku_smi:
            keku_inch_ori=  Chem.MolToInchi(Chem.MolFromSmiles(keku_smi_ori))
            keku_inch_test=  Chem.MolToInchi(Chem.MolFromSmiles(keku_smi))
        else:
            keku_inch_ori=  1
            keku_inch_test=  2

        rd_smi=Chem.MolToSmiles(test_mol)#need improve the acc
        rd_smi_ori=Chem.MolToSmiles(original_mol)
    except Exception as e:#TODO fixme here
        print(f"comparing_smiles@@@ kekulize or SanitizeMol problems")# original_smiles,test_smiles\n{original_smiles}\n{test_smiles}")
        print(new_row)
        print(e,"!!!!!!!\n")
        keku_inch_ori=  1
        keku_inch_test=  2
        keku_smi=1
        keku_smi_ori=2
        #add molscribe rules here
        if not success1:#ori smiles still invaild even after * replaced
            rd_smi_ori = rd_smi
        # else:
        #     if canon_smiles1 == canon_smiles2:
        #         rd_smi_ori = rd_smi
            # else:
    if rd_smi_ori == rd_smi or keku_smi_ori == keku_smi or keku_inch_ori==keku_inch_test :#as orinial smiles may use kekuleSmiles style
        return True
    else:return False



#############################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@torch.no_grad()
def evaluate2(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, 
        data_loader, device,
        outcsv_filename=f'/home/jovyan/rt-detr/rt-detr/output/output_charge_CLEF.csv',
        visual_check=False,
        other2ppsocr=True,
        getacc=False,
        ):
    output_directory = os.path.dirname(outcsv_filename)
    prefix_f = os.path.basename(outcsv_filename).split('.')[0]
    if other2ppsocr:
        ocr = PaddleOCR(use_angle_cls=True,use_gpu =False,
            rec_algorithm='SVTR_LCNet', rec_model_dir='/home/jovyan/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer',
            lang="en") 
        outcsv_filename=f"{output_directory}/{prefix_f}_withOCR.csv"

    if visual_check:
        output_directory = os.path.dirname(outcsv_filename)
        prefix_f = os.path.basename(outcsv_filename).split('.')[0]
        ima_checkdir=f"{output_directory}/{prefix_f}_Boxed"
        os.makedirs(ima_checkdir, exist_ok=True)
    
    
    if getacc:
        acc_summary=f"{outcsv_filename}.I2Msummary.txt"
        flogout = open(f'{acc_summary}' , 'w')
        failed=[]
        mydiff=[]
        simRD=0
        sim=0
        mysum=0

    model.eval()
    criterion.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Infering:'

    res_smiles = []
    idx_to_labels23={0:'other',1:'C',2:'O',3:'N',4:'Cl',5:'Br',6:'S',7:'F',8:'B',
                        9:'I',10:'P',11:'*',12:'Si',13:'NONE',14:'BEGINWEDGE',15:'BEGINDASH',
                        16:'=',17:'#',18:'-4',19:'-2',20:'-1',21:'1',22:'2',} 
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
    bond_labels = [13,14,15,16,17]
    if postprocessors.num_classes==23:
        # print(data["categories"])
        print(f'usage idx_to_labels23',idx_to_labels23)
        idx_to_labels = idx_to_labels23
    elif postprocessors.num_classes==30:
        # print(data["categories"])#NOTE 11 is H not * now
        print(f'usage idx_to_labels30',idx_to_labels30)
        idx_to_labels = idx_to_labels30
    else:
        print(f"error unkown ways@@@@@@@@@@@!!!!!!!!!!idx_to_labels::{len(idx_to_labels)}\n{idx_to_labels}")
    abrevie={"[23*]":'CF3',
                                "[24*]":'CN',
                                "[25*]":'Me',
                                "[26*]":'CO2Et',
                                "[27*]":'R',
                                "[28*]":'Ph',
                                "[29*]":'3~7UP',
        }

    # idx_to_labels = idx_to_labels23
    smiles_data = pd.DataFrame({'file_name': [],
                                'SMILESori':[],
                                'SMILESpre':[],
                                'SMILESexp':[],
                                }
                                )
    output_dict = {}
    output_ori={}
    filtered_output_dict = {}
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)#.to(device)    
        orig_target_sizes = targets["orig_size"].to(device)  
        results = postprocessors(outputs, orig_target_sizes)#RTDETRPostProcessor@@src/zoo/rtertr

        for i_, z in enumerate(zip(targets['image_id'], results)):
            ti, output=z
            output_dict[ti.item()] = output
            output_ori[ti.item()] =[ targets['img_path'][i_], 
                                    targets['SMILES'][i_]]
            
    # print(len(output_ori),len(output_dict))     
    for key, value in output_dict.items():
        selected_indices = value['scores'] > 0.5#may be >=0.5 cut off, as used the sigmoid?
        if value['labels'][selected_indices].size(0) != 0:#no good prediction
            filtered_output_dict[key] = {
                'labels': value['labels'][selected_indices],# may be selected_indices ==0 as all small than0.5
                'boxes': value['boxes'][selected_indices],
                'scores': value['scores'][selected_indices]
            }
        else:
            ima_name=output_ori[key][0]
            SMILESori=output_ori[key][1]
            print(key,"all prediction scores small 0.5!!",len(output_dict),f"{ima_name}\n SMILESori:\n{SMILESori}")##
    #process filted with/without OCR 
    # for i,(key,value) in enumerate(filtered_output_dict.items()):
    for key,value in filtered_output_dict.items():
        # smi_mol=output_to_smiles(value,idx_to_labels,bond_labels,result)#TODO use the idx_to_labels numer to if --else
        image_path=output_ori[key][0]
        SMILESori=output_ori[key][1]
        rt = []   
        abc,SMILESpre,mol_rebuit,output=output_to_smiles2(value,idx_to_labels,bond_labels,rt)
        atoms_df, bonds_list,charge =abc
        # if isinstance(mol_rebuit, None):#TODO mol_rebuit awalys be somethings

        # else:
        #     #above for loop is equal output_to_smiles
        #     #set bond direction again
        wdbs=[b for b in bonds_list if b[3]=='BEGINDASH' or  b[3]=='BEGINWEDGE' ]
        for b in wdbs:
            try:
                b_=mol_rebuit.GetBondBetweenAtoms(int(b[0]), int(b[1]))#should try to fix if not have the bond add it
                if b_:
                    b_.SetBondDir(bond_dirs[b[3]])
            except Exception as e:
                print(e,"@@wdbs")
                print(b,f"\n{wdbs}")
                print(f"atoms_df@@{len(atoms_df)}")
                print(f"bonds_list@@{len(bonds_list)}")
                print(f"charge@@{len(charge)}")
                print(image_path)
                print(SMILESori)
                print(key,"@@key",[int(b[0]), int(b[1])])
                # raise CustomError(Exception)
            
        aid_at={}
        aid_at_star={}
        for i_, row in atoms_df.iterrows():
            a_lab=row.atom[:-1]
            aid_at[i_]=a_lab
            #atom types not in defined vocable list
            if a_lab not in ['H', 'C', 'O', 'N', 'Cl', 'Br', 'S', 'F', 'B', 'I', 'P', 'Si']:#  '*', I2M's defined atom types
                aid_at_star[i_]="*"
                    # aid_at[i_]=a_lab
            # print(aid_at,aid_at_star)
        placeholder_atoms={k:aid_at[k] for k,v in aid_at_star.items() if aid_at[k] !='*' }#NOTE when 30IDX2lab get ABB case
        # aid_rest={k:aid_at[k] for k,v in aid_at_star.items() if k not in }
        aidstart_rest={k:v for k,v in aid_at_star.items() if aid_at[k]=='*'}#
        predict_boxes = output['bbox']
        predict_classes = output['pred_classes']
        predict_scores = output['scores']
        img_ori = Image.open(image_path).convert('RGB')
        w_ori, h_ori = img_ori.size  # 获取原始图像的尺寸
        # print(w_ori, h_ori, "orignianl vs 1000,1000")
        # 计算缩放比例
        scale_x = 1000 / w_ori
        scale_y = 1000 / h_ori
        img_ori_1k = img_ori.resize((1000,1000))
        #common part for visul checking and ocr,both use the img_ori_1k
        img = Image.open(image_path).convert('RGB')
        img = img.resize((1000,1000))
        newbox = predict_boxes * [scale_x, scale_y, scale_x, scale_y]
        mol = rdkit.Chem.RWMol(mol_rebuit)
        #TODO insert the OCR extend functions
        if other2ppsocr:
            need_cut=[]
            ppstr=[]
            ppstr_score=[]
            crops=[]
            index_token=dict()
            expan=-2#NOTE this control how much the part of bond in crop_Img
            # mol=deep.copy(mol_rebuit)
            # for i_, v in aidstart_rest.items():
            try:
                for i_, row in atoms_df.iterrows():
                    if "*" in row.atom or "other" in row.atom:
                        need_cut.append(i_)
                        a=np.array(row.bbox )+np.array([-expan,-expan,expan,expan])#expand crop
                        box=a * [scale_x, scale_y, scale_x, scale_y]#TODO need the fix as w h may not equal!!
                        # print(a,box,[scale_x, scale_y, scale_x, scale_y])
                        cropped_img = img_ori_1k.crop(box)
                        if cropped_img.mode != 'RGB':
                            cropped_img = cropped_img.convert('RGB')
                        if cropped_img.size[0] == 0 or cropped_img.size[1] == 0:
                            raise ValueError("Cropped image is empty. Check the `box` coordinates.")
                        crops.append(cropped_img)
                        image_np = np.array(cropped_img)
                        if len(image_np.shape) == 2:
                            image_np = np.stack([image_np] * 3, axis=-1)
                        # 检查裁剪后的图像是否为空
                        try:
                            result_ocr = ocr.ocr(image_np, det=False)
                            # print("OCR result:", result_ocr)
                        except Exception as e:
                            print("OCR failed with error:", e)
                        # result_ocr = ocr.ocr(image_np, det=False)
                        s_, score_ =result_ocr[0][0]
                        # print(f'ocr::idx:{i_}',s_, score_ )
                        if score_<=0.1:# process cropped_img and try again
                            # print(s_, "xxx",score_)
                            s_='*'
                        if s_=='+' or s_=='-':
                            s_="*"
                        if len(s_)>1:
                            s_=re.sub(r'[^a-zA-Z0-9\*\-\+]', '', s_)#remove special chars
                            if re.match(r'^\d+$', s_):
                                s_=f'{s_}*'#number+ *
                                # print(f'why only numbers ?  {s_}')
                        if s_=='L':s_='Li'
                        match = re.match(r'^(\d+)?(.*)', s_)
                        if match:
                            numeric_part, remaining_part = match.groups()
                        if remaining_part in ELEMENTS or remaining_part in ABBREVIATIONS:# can be expanded with placeholder_atoms
                                placeholder_atoms[i_]=s_# such 2Na will be get for rdkit
                            # else:
                            #     if numeric_part:
                            #         s_=f'{numeric_part}*'
                            #     else:
                            #         s_='*'
                        index_token[i_]=f'{s_}:{i_}'
                        # print(f"idx:{i_}, atm:{row.atom}-->[{s_}:{i_}] with score:{score_}")
                        mol.GetAtomWithIdx(i_).SetProp("atomLabel", f"{s_}")# now * will be dipalyed with s_
                        ppstr.append(s_)
                        ppstr_score.append(score_)
                        if s_ in ELEMENTS or s_ in ABBREVIATIONS.keys():
                            placeholder_atoms[i_]=s_
            except Exception as ee:
                print(f'extended OCR problems@@ {image_path}\n{SMILESori}\n{SMILESpre}')
                print(ee)
        # if len(idx_to_labels)==30:
        if len(placeholder_atoms)>0:
            print(f'MOL will be expanded with {placeholder_atoms} !!')
            expandStero_smi,molexp= molExpanding(mol,placeholder_atoms,wdbs,bond_dirs)#TODO fix me whe n multi strings on a atom will missing this ocr infors
        else:
            molexp=mol
            expandStero_smi=SMILESpre #save into csv files, 
        #visual checking
        # TODO #[3H] 2H prpared box for training are too smalled, need adjust
        if visual_check:
            boxed_img = draw_objs(img,
                                newbox,
                                predict_classes,
                                predict_scores,
                                category_index=idx_to_labels,
                                box_thresh=0.5,
                                line_thickness=3,
                                font='arial.ttf',
                                font_size=10)
            opts = Draw.MolDrawOptions()
            opts.addAtomIndices = False
            opts.addStereoAnnotation = False
            img_ori = Image.open(image_path).convert('RGB')
            img_ori_1k = img_ori.resize((1000,1000))
            if other2ppsocr:
                img_rebuit = Draw.MolToImage(molexp, options=opts,size=(1000, 1000))
            else:
                img_rebuit = Draw.MolToImage(molexp, options=opts,size=(1000, 1000))
            combined_img = Image.new('RGB', (img_ori_1k.width + boxed_img.width + img_rebuit.width, img_ori_1k.height))
            combined_img.paste(img_ori_1k, (0, 0))
            combined_img.paste(boxed_img, (img_ori_1k.width, 0))
            combined_img.paste(img_rebuit, (img_ori_1k.width + boxed_img.width, 0))
            imprefix=os.path.basename(image_path).split('.')[0]

            combined_img.save(f"{ima_checkdir}/{imprefix}Boxed.png")
        new_row = {'file_name':image_path, "SMILESori":SMILESori,
                'SMILESpre':SMILESpre,#with *  without expand
                'SMILESexp':expandStero_smi, 
                }
        smiles_data = smiles_data._append(new_row, ignore_index=True)
        #accu  similarity calculation 
        if getacc:
            sameWithOutStero=comparing_smiles(new_row,SMILESpre)#try to ingnore cis chiral, as 2d coords including all the infos
            sameWithOutStero_exp=comparing_smiles(new_row,expandStero_smi)#this ignore chairity and *number be * NOTE

            if (type(SMILESori)!=type('a')) or (type(SMILESpre)!=type('a')):
                if sameWithOutStero or sameWithOutStero_exp:
                    mysum += 1
                else:
                    print(f"smiles problems\n{SMILESori}\n{SMILESpre}\n{image_path}")
                    failed.append([SMILESori,SMILESpre,image_path])
                    mydiff.append([SMILESori,SMILESpre,image_path])
                    continue
            mol1 = Chem.MolFromSmiles(SMILESori)#TODO considering smiles with rdkit not recongized in real data
            if mol1 is None:
                rd_smi_ori, success1_=rdkit_canonicalize_smiles(SMILESori)
                mol1=Chem.MolFromSmiles(rd_smi_ori)
            if (mol_rebuit is None) or (mol1 is None):
                if sameWithOutStero or sameWithOutStero_exp:
                    mysum += 1
                else:
                    print(f'get rdkit mol None\n{SMILESori}\n{SMILESpre}\n{image_path}')
                    failed.append([SMILESori,SMILESpre,image_path])
                    mydiff.append([SMILESori,SMILESpre,image_path])
                    continue
            if mol1:
                rdk_smi1=Chem.MolToSmiles(mol1)
            else:
                rdk_smi1=SMILESori
            if mol_rebuit:
                rdk_smi2=Chem.MolToSmiles(mol_rebuit)
            else:
                rdk_smi2=''
            # if rdk_smi1==rdk_smi2 or rdk_smi1==expandStero_smi or sameWithOutStero:#also considering the abbre in Ori
            if rdk_smi1==rdk_smi2 or rdk_smi1==expandStero_smi:
                mysum += 1
            else:
                if sameWithOutStero or sameWithOutStero_exp:
                    mysum += 1
                else:
                    mydiff.append([SMILESori,SMILESpre,image_path])
                    if visual_check:
                        combined_img.save(f"{ima_checkdir}/{imprefix}Boxed_diff{len(mydiff)}.png")
            try:
                morganfps1 = AllChem.GetMorganFingerprint(mol1, 3,useChirality=True)
                morganfps2 = AllChem.GetMorganFingerprint(mol_rebuit, 3,useChirality=True)
                morgan_tani = DataStructs.DiceSimilarity(morganfps1, morganfps2)
                fp1 = Chem.RDKFingerprint(mol1)
                fp2 = Chem.RDKFingerprint(mol_rebuit)
                tanimoto = DataStructs.FingerprintSimilarity(fp1, fp2)
                if expandStero_smi!= '':
                    fp3 = Chem.RDKFingerprint(molexp)
                    morganfps3 = AllChem.GetMorganFingerprint(molexp, 3,useChirality=True)
                    morgan_tani3 = DataStructs.DiceSimilarity(morganfps1, morganfps3)
                    tanimoto3 = DataStructs.FingerprintSimilarity(fp1, fp3)
                if morgan_tani3> morgan_tani or tanimoto3> tanimoto :
                    sim+=morgan_tani3
                    simRD+=tanimoto3
                else:
                    simRD+=tanimoto
                    sim+=morgan_tani
            except Exception as e:
                print(f"mol to fingerprint erros")
                simRD+=0
                sim+=0
                continue

    if getacc:
        sim_100 = 100*sim/len(smiles_data)
        simrd100 = 100*simRD/len(smiles_data)
        flogout.write(f"rdkit concanlized==smiles:{100*mysum/len(smiles_data)}%\n")
        flogout.write(f"failed:{len(failed)}\n totoal saved in csv : {len(smiles_data)}\n")
        flogout.write(f"avarage similarity morgan tanimoto: RDKFp tanimoto:: {sim_100}%,  {simrd100}%  \n")#morgan_tani considering chiraty
        flogout.write(f'I2M@@:: match--{mysum},unmatch--{len(mydiff)},failed--{len(failed)},correct %{100*mysum/len(smiles_data)} \n')
        #molscribe evalutate
        from src.solver.evaluate import SmilesEvaluator
        evaluator = SmilesEvaluator(smiles_data['SMILESori'], tanimoto=False)
        res_pre=evaluator.evaluate(smiles_data['SMILESpre'])
        res_exp=evaluator.evaluate(smiles_data['SMILESexp'])
        flogout.write(f'MolScribe style evaluation@SMILESpre:: {str(res_pre)} \n')
        flogout.write(f'MolScribe style evaluation@SMILESexp:: {str(res_exp)} \n')
        flogout.close()
    print(f"will save {len(smiles_data)} dataframe into csv") 
    smiles_data.to_csv(outcsv_filename, index=False)
