from scipy import ndimage
import os
import cv2
import time
import random
import re
import string
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2
#################not requred anymore # !pip install albumentations==1.1.0
#pip install SmilesPE

import matplotlib.pyplot as plt
import copy

import math
import rdkit
import indigo
from indigo import Indigo
from indigo.renderer import IndigoRenderer
#conda install fontconfig -y
#conda install -c anaconda freetype -p /home/jovyan/bo2/py38 -y
from augment import SafeRotate, PadWhite, SaltAndPepperNoise
from utils import FORMAT_INFO
from chemistry import get_num_atoms, normalize_nodes
from constants import RGROUP_SYMBOLS, SUBSTITUTIONS, ELEMENTS, COLORS
from scipy.spatial import KDTree
import scipy
import gzip

from copy import deepcopy
from torchvision.transforms import ToPILImage
# from torchvision.transforms import functional as F
import math
from PIL import Image, ImageDraw



to_img = ToPILImage()

# def img_to_tensor(image,keypoints=None, normalize=None):
#     tensor = torch.from_numpy(np.moveaxis(image / (255.0 if image.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
#     if normalize is not None:
#         return F.normalize(tensor, **normalize)
#     return tensor

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}
cv2.setNumThreads(1)

INDIGO_HYGROGEN_PROB = 0.2
INDIGO_FUNCTIONAL_GROUP_PROB = 0.8
INDIGO_CONDENSED_PROB = 0.5
# INDIGO_RGROUP_PROB = 0.5
INDIGO_RGROUP_PROB = 0.5
INDIGO_COMMENT_PROB = 0.4
INDIGO_DEARMOTIZE_PROB = 0.8
INDIGO_COLOR_PROB = 0.2
# 1/2/3/4 : single/double/triple/aromatic https://lifescience.opensource.epam.com/indigo/api/index.html?highlight=bondorder
rdkitbond_type_dict = {
            rdkit.Chem.rdchem.BondType.SINGLE:1,#same as indigo
            rdkit.Chem.rdchem.BondType.DOUBLE:2,
            rdkit.Chem.rdchem.BondType.TRIPLE:3,
            rdkit.Chem.rdchem.BondType.AROMATIC:4,

            rdkit.Chem.rdchem.BondDir.BEGINWEDGE:5,
            rdkit.Chem.rdchem.BondDir.BEGINDASH:6,#keep same oreder as indigo
            rdkit.Chem.rdchem.BondDir.UNKNOWN:7,
            }#not consider the double bond stero Z/E
# https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Bond.GetBondDir
def point_in_polygon(x, y, poly):
    """
    Determine if a point is inside a given polygon or not.

    Args:
    x (int): x-coordinate of the point.
    y (int): y-coordinate of the point.
    poly (list): List of tuples each containing the coordinates of a polygon's corners.

    Returns:
    bool: True if the point is inside the polygon, False otherwise.
    """
    num = len(poly)
    j = num - 1
    c = False
    for i in range(num):
        if ((poly[i][1] > y) != (poly[j][1] > y)) and (
                x < (poly[j][0] - poly[i][0]) * (y - poly[i][1]) / (poly[j][1] - poly[i][1]) + poly[i][0]):
            c = not c
        j = i
    return c


def get_transforms(input_size, image_augment=True, rotate=True, debug=False):
    trans_list = []
    if image_augment:
        print('imgage_aug!!!!!!!!!!!!!!!!!')
        trans_list += [
            # NormalizedGridDistortion(num_steps=10, distort_limit=0.3),
            A.CropAndPad(percent=[-0.01, 0.00], keep_size=False, p=0.5),
            PadWhite(pad_ratio=0.4, p=0.2),
            A.Downscale(scale_min=0.2, scale_max=0.5, interpolation=3),
            A.Blur(),
            A.GaussNoise(),
            SaltAndPepperNoise(num_dots=20, p=0.5)#return image2
        ]
        if rotate:#TODO why  make x y >512?? how to avoid this!!
            trans_list +=[
                        # A.Rotate(limit=45, p=0.9),
                        #   A.HorizontalFlip(p=0.5),
                        #     A.VerticalFlip(p=0.5),
                        SafeRotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            ]

    trans_list.append(A.Resize(input_size-16*2, input_size-16*2))#TODO make in center, considering some atom at edge
    trans_list.append(
        A.PadIfNeeded(min_height=input_size, min_width=input_size, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255], p=1.0)
    )
    # trans_list.append(A.Resize(input_size, input_size))
    if not debug:
        if random.random()<=0.01:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]# commonly known as "ImageNet normalization" 
            trans_list += [
                # A.ToGray(p=1),#why must be gray? TODO trianing easy ??
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),#return image3
            ]
        else:#TODO when not be gray and not normal, result like?
            '''
            Dividing by 255 is a simpler and more generic approach. It normalizes the pixel values to the range of 0-1, regardless of the specific dataset or its characteristics.
            '''
            trans_list += [               
                # img_to_tensor,
                # A.Normalize(),#blue backgound color，same as "ImageNet normalization" 
                ToTensorV2(),

                ]
    # print(len(trans_list),'@trans_list')
    return A.Compose(trans_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def add_functional_group(indigo, mol, debug=False):
    if random.random() > INDIGO_FUNCTIONAL_GROUP_PROB:
        return mol
    # Delete functional group and add a pseudo atom with its abbrv
    substitutions = [sub for sub in SUBSTITUTIONS]
    random.shuffle(substitutions)
    for sub in substitutions:
        query = indigo.loadSmarts(sub.smarts)
        matcher = indigo.substructureMatcher(mol)
        matched_atoms_ids = set()
        for match in matcher.iterateMatches(query):
            if random.random() < sub.probability or debug:
                atoms = []
                atoms_ids = set()
                for item in query.iterateAtoms():
                    atom = match.mapAtom(item)
                    atoms.append(atom)
                    atoms_ids.add(atom.index())
                if len(matched_atoms_ids.intersection(atoms_ids)) > 0:
                    continue
                abbrv = random.choice(sub.abbrvs)
                superatom = mol.addAtom(abbrv)
                for atom in atoms:
                    for nei in atom.iterateNeighbors():
                        if nei.index() not in atoms_ids:
                            if nei.symbol() == 'H':
                                # indigo won't match explicit hydrogen, so remove them explicitly
                                atoms_ids.add(nei.index())
                            else:
                                superatom.addBond(nei, nei.bond().bondOrder())
                for id in atoms_ids:
                    mol.getAtom(id).remove()
                matched_atoms_ids = matched_atoms_ids.union(atoms_ids)
    return mol


def add_explicit_hydrogen(indigo, mol,p=0.2):
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append((atom, hs))
        except:
            continue
    if len(atoms) > 0 and random.random() < p:
        atom, hs = random.choice(atoms)
        for i in range(hs):
            h = mol.addAtom('H')
            h.addBond(atom, 1)
    return mol


def add_rgroup(indigo, mol, smiles,RGROUP_SYMBOLS,p=0.5):
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except:
            continue
    if len(atoms) > 0 and '*' not in smiles:
        # if random.random() < INDIGO_RGROUP_PROB:
        if random.random() < p:
            atom_idx = random.choice(range(len(atoms)))
            atom = atoms[atom_idx]
            atoms.pop(atom_idx)
            symbol = random.choice(RGROUP_SYMBOLS)
            r = mol.addAtom(symbol)
            r.addBond(atom, 1)
    return mol


def get_rand_symb():
    symb = random.choice(ELEMENTS)
    if random.random() < 0.1:
        symb += random.choice(string.ascii_lowercase)
    if random.random() < 0.1:
        symb += random.choice(string.ascii_uppercase)
    if random.random() < 0.1:
        symb = f'({gen_rand_condensed()})'
    return symb


def get_rand_num():
    if random.random() < 0.9:
        if random.random() < 0.8:
            return ''
        else:
            return str(random.randint(2, 9))
    else:
        return '1' + str(random.randint(2, 9))


def gen_rand_condensed():
    tokens = []
    for i in range(5):
        if i >= 1 and random.random() < 0.8:
            break
        tokens.append(get_rand_symb())
        tokens.append(get_rand_num())
    return ''.join(tokens)


def add_rand_condensed(indigo, mol):
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except:
            continue
    if len(atoms) > 0 and random.random() < INDIGO_CONDENSED_PROB:
        atom = random.choice(atoms)
        symbol = gen_rand_condensed()
        r = mol.addAtom(symbol)
        r.addBond(atom, 1)
    return mol


def generate_output_smiles(indigo, mol):
    # TODO: if using mol.canonicalSmiles(), explicit H will be removed
    '''
    special cases
    [*][N-]/C(/[*])=C(/[*])\[N][*]
    [*][N-]/C%91=C%92\[N]%93.[*]\%92.[*]%93.[*]\%91 |^1:4|
    
    '''
    smiles = mol.smiles()
    mol = indigo.loadMolecule(smiles)
    if ' ' in smiles:#smiles='CC(C)N1CC([*])CC1C(=O)C(C)C' original input, then output no space
        if '*' in smiles and '$' in smiles:
            part_a, part_b = smiles.split(' ', maxsplit=1)
            part_b = re.search(r'\$.*\$', part_b).group(0)[1:-1]
            symbols = [t for t in part_b.split(';') if len(t) > 0]
            output = ''
            cnt = 0
            for i, c in enumerate(part_a):
                if c != '*':
                    output += c
                else:
                    if cnt<len(symbols):
                        output += f'[{symbols[cnt]}]'
                    else:
                        output += f'[*]'
                    cnt += 1
            return mol, output
        else:
            # special cases with extension
            # print(f'mol.smiles()::{smiles}')
            smiles = smiles.split(' ')[0]
    # else:
    return mol, smiles


def add_comment(indigo1):
    if random.random() < INDIGO_COMMENT_PROB:
        indigo1.setOption('render-comment', str(random.randint(1, 20)) + random.choice(string.ascii_letters))
        indigo1.setOption('render-comment-font-size', random.randint(40, 60))
        indigo1.setOption('render-comment-alignment', random.choice([0, 0.5, 1]))
        indigo1.setOption('render-comment-position', random.choice(['top', 'bottom']))
        indigo1.setOption('render-comment-offset', random.randint(2, 30))



def add_color(indigo1, mol):
    if random.random() < INDIGO_COLOR_PROB:
        indigo1.setOption('render-coloring', True)
    if random.random() < INDIGO_COLOR_PROB:
        indigo1.setOption('render-base-color', random.choice(list(COLORS.values())))
    if random.random() < INDIGO_COLOR_PROB:
        if random.random() < 0.5:
            indigo1.setOption('render-highlight-color-enabled', True)
            indigo1.setOption('render-highlight-color', random.choice(list(COLORS.values())))
        if random.random() < 0.5:
            indigo1.setOption('render-highlight-thickness-enabled', True)
        for atom in mol.iterateAtoms():
            if random.random() < 0.1:
                atom.highlight()
    return mol

#rdkitbased img
def rdkit_addGroup(mol_sm="NCCC(CCO)CCBr",cyano_sm="C#N"):
    mol = Chem.MolFromSmiles(mol_sm)
    atoms_A = mol.GetNumAtoms()
    # Create a cyano group molecule
    cyano = Chem.MolFromSmiles(cyano_sm)
    atoms_site=[]
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()  # Get the atom index
        num_hydrogens = atom.GetTotalNumHs()  # Count hydrogens (including implicit)
        if num_hydrogens >0:
            atoms_site.append(atom_idx)
    # h_atom_indexA=6
    h_atom_indexA=random.choice(atoms_site)
    combined_mol = Chem.CombineMols(mol, cyano)
    rw_combined = Chem.RWMol(combined_mol)
    merged_atom_id=[]
    for i,atom in enumerate(rw_combined.GetAtoms()):
        if i >=atoms_A:
            num_hydrogens = atom.GetTotalNumHs()  # Count hydrogens (including implicit)
            if num_hydrogens >0:
                merged_atom_id.append(i)
    h_atom_indexB= random.choice(merged_atom_id)
    rw_combined.AddBond(h_atom_indexA,h_atom_indexB, Chem.rdchem.BondType.SINGLE)  # Triple bond to CN
    # reset radical electrons
    bonding_atoms=[h_atom_indexA,h_atom_indexB]
     # Convert all implicit hydrogens to explicit
    mol_with_h = Chem.AddHs(rw_combined,explicitOnly=True)#NOTE this avoid the N valence errors,n AddH node to convert implicit Hs to H atom here
    # Chem.RemoveHs(rw_mol)#only work for H atom
    rw_mol = Chem.RWMol(mol_with_h)
    for atom in reversed(mol_with_h.GetAtoms()):
        #H atom neigbor only one normaly
        if atom.GetSymbol() == 'H' and atom.GetNeighbors()[0].GetIdx() in bonding_atoms:#remove bond after H
            rw_mol.RemoveAtom(atom.GetIdx())
            bonding_atoms.remove(atom.GetNeighbors()[0].GetIdx())

    # # List of hydrogen atoms to remove
    # hydrogen_indices = []
    # # Identify and prepare to remove explicit hydrogen atoms
    # for atom in rw_mol.GetAtoms():
    #     if atom.GetSymbol() == 'H':
    #         heavy_atom = atom.GetNeighbors()[0]  # Hydrogen should have exactly one neighbor
    #         heavy_atom.SetNumExplicitHs(heavy_atom.GetNumExplicitHs() + 1)
    #         hydrogen_indices.append(atom.GetIdx())
    # # Remove hydrogen atoms, starting from the highest index to avoid changing indices of remaining atoms
    # for idx in sorted(hydrogen_indices, reverse=True):
    #     rw_mol.RemoveAtom(idx)
    # Sanitize the molecule to correct any issues and update structure
    Chem.SanitizeMol(rw_mol)
    return rw_mol

#generate_indigo_image
def indigo_img(idx,smiles, mol_augment=True, default_option=False, shuffle_nodes=False,
                          include_condensed=False, debug=False,  addRandom=False,RGROUP_SYMBOLS=RGROUP_SYMBOLS,radius=3,image_size=480,number_querry=100):
    indigo1 = Indigo()
    renderer = IndigoRenderer(indigo1)
    # print('generate_indigo_image seting finished !!!!!!!!!!!!!!!!!!!!!')
    indigo1.setOption('render-output-format', 'png')
    indigo1.setOption('render-background-color', '1,1,1')
    indigo1.setOption('render-stereo-style', 'none')
    indigo1.setOption('render-label-mode', 'hetero')
    # image_size = np.random.randint(360,512-2*16)#NOTE we will rescale and padding as 512 in A.transformer step
    # image_size = np.random.randint(460,512-2*16)#NOTE we will rescale and padding as 512 in A.transformer step
    # indigo1.setOption('render-image-size', f'{image_size},{image_size}')
    # indigo1.setOption('render-font-family', 'Arial')
    color_mol=True# if mol_augment else False
    if not default_option:
        thickness = random.uniform(0.5, 2)  # limit the sum of the following two parameters to be smaller than 4
        indigo1.setOption('render-relative-thickness', thickness)
        indigo1.setOption('render-bond-line-width', random.uniform(1, 4 - thickness))
        # if random.random() < 0.5:
        #     indigo1.setOption('render-font-family', random.choice(['Arial', 'Times', 'Courier', 'Helvetica']))
        indigo1.setOption('render-label-mode', random.choice(['hetero', 'terminal-hetero']))
        indigo1.setOption('render-implicit-hydrogens-visible', random.choice([True, False]))
    if addRandom:
        if random.random() < 0.1:
            indigo1.setOption('render-stereo-style', 'old')
        if random.random() < 0.2:
            indigo1.setOption('render-atom-ids-visible', True)
        # if random.random() < 0.6:#line stye mol will be effectd
    ori_sm=smiles
    try:
        mol = indigo1.loadMolecule(smiles)
        num_atoms=len([a  for i,a in enumerate(mol.iterateAtoms())])
        if num_atoms>number_querry:
            # print(f' model decoder limitation indigo_img filter atoms numbers {num_atoms} > {number_querry}')
            img = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
            graph={}
            success=False
            return img, smiles, graph, success
        #even no mol_aug, make aromati comment randomly
        if random.random() <= 0.5:
            mol.dearomatize()
        else:
            mol.aromatize()#0.2 aro
        smiles = mol.canonicalSmiles()
        if random.random() < 0.6: add_comment(indigo1)
        
        #TODO make molecule charged
        #inset triple bond
        #easy ocr return  the atom label xy  
        ##add the molgrapher drawing skills done

        if mol_augment:
            # print('mol_aug!!!!!!!!!!!!!!!!!')
            mol = add_explicit_hydrogen(indigo1, mol,p=0.2)
            mol = add_rgroup(indigo1, mol, smiles,RGROUP_SYMBOLS=RGROUP_SYMBOLS)#such as * R1 CF3, will be new atom symbols
            # if include_condensed:
            #     mol = add_rand_condensed(indigo1, mol)
            try:#NOTE this aug the new symbols from substuition
                mol = add_functional_group(indigo1, mol, debug)
            except Exception as e:
                if 'valence' in str(e):
                    # print(e,'passed skip @add_functional_group failed')
                    pass
                else:
                    print(e,'check this  @add_functional_group failed!!!',f'ori_sm/smiles::\n{ori_sm}\n{smiles}')

            mol, smiles = generate_output_smiles(indigo1, mol)

        # buf = renderer.renderToBuffer(mol)#NOTE this bond length not changed
        # img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # decode buffer to image with 1 as RGB
        # img = np.repeat(np.expand_dims(img, 2), 3, axis=2)  # expand to RGB
        #ABCnet
        mol.layout()
        x = []
        y = []
        for i,a in enumerate(mol.iterateAtoms()):
            position = a.xyz()
            x.append(position[0])
            y.append(position[1])
        x = np.array(x)
        y = np.array(y)
        deltas = []
        for bond in mol.iterateBonds():
            begin_atom = bond.source()
            end_atom = bond.destination()
            delta = np.sqrt(
                (x[begin_atom.index()] - x[end_atom.index()]) ** 2 + (y[begin_atom.index()] - y[end_atom.index()]) ** 2)
            deltas.append(delta)
        delta = np.array(deltas).mean()
        bond_length = int(delta*(image_size)/max(x.max()-x.min(),y.max()-y.min()))
        bond_length = min(max(bond_length*np.random.uniform(2,3),40),200)#40~200
        bond_length = int(delta * (image_size-bond_length) / max(x.max() - x.min(), y.max() - y.min()))
        scale = bond_length / delta
        #Desired average bond length in pixels, default 100, NOTE if bond_length=10 too small above not sutable
        indigo1.setOption("render-bond-length", bond_length)#NOTE have to apply when sacled

        # renderer.renderToFile(mol, path)
        # mol = add_color(indigo1, mol)#TODO testing color here
        buf = renderer.renderToBuffer(mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # decode buffer to image
        h,w,c=img.shape#NOTE h,w is based on the 'render-bond-length'
        x = x - (x.max() + x.min()) / 2
        y = y - (y.max() + y.min()) / 2
        # x = image_size // 2 + (scale * x).astype('int32')
        # y = image_size // 2 - (scale * y).astype('int32')#why -?
        x = w // 2 + (scale * x).astype('int32')
        y = h // 2 - (scale * y).astype('int32')#results from indigo black box drawing fn 

        bond_x = []
        bond_y = []
        bond_img_x = []
        bond_img_y = []
        b_eid=[]
        try:#align x y on image, add_color have to be not before this step!!
            for bond in mol.iterateBonds():
                bond_type = bond.bondOrder()
                begin_atom = bond.source()
                end_atom = bond.destination()
                if ((bond_type == 1)) and (begin_atom.symbol() == end_atom.symbol()):
                    bond_x.append((x[begin_atom.index()] + x[end_atom.index()]) / 2)
                    bond_y.append((y[begin_atom.index()] + y[end_atom.index()]) / 2)
                    bond.highlight()
                    # renderer.renderToFile(mol, '_temp2.png')
                    # img_ref = cv2.imread('_temp2.png', flags=1)
                    buf_tmp = renderer.renderToBuffer(mol)
                    img_ref = cv2.imdecode(np.asarray(bytearray(buf_tmp), dtype=np.uint8), 1)  # decode buffer to image
                    img_ref = (img_ref[:, :, 2] > 230) * (img_ref[:, :, 0] < 230) * (img_ref[:, :, 1] < 230)
                    # plt.imshow(img_ref)
                    # plt.show()
                    red_indices = np.where(img_ref)
                    col_indices, row_indices = red_indices
                    red_min_x, red_max_x = row_indices.min(), row_indices.max()
                    red_min_y, red_max_y = col_indices.min(), col_indices.max()
                    red_mean_x = (red_min_x + red_max_x) / 2
                    red_mean_y = (red_min_y + red_max_y) / 2
                    bond_img_x.append(red_mean_x)
                    bond_img_y.append(red_mean_y)
                    bond.unhighlight()
            if len(bond_x)<3:
                bond_x = []
                bond_y = []
                bond_img_x = []
                bond_img_y = []
                for bond in mol.iterateBonds():
                    bond_type = bond.bondOrder()
                    begin_atom = bond.source()
                    end_atom = bond.destination()
                    if (bond_type == 1 or bond_type==4):
                        bond_x.append((x[begin_atom.index()] + x[end_atom.index()]) / 2)
                        bond_y.append((y[begin_atom.index()] + y[end_atom.index()]) / 2)
                        bond.highlight()
                        # renderer.renderToFile(mol, '_temp2.png')
                        # img_ref = cv2.imread('_temp2.png', flags=1)
                        buf_tmp = renderer.renderToBuffer(mol)
                        img_ref = cv2.imdecode(np.asarray(bytearray(buf_tmp), dtype=np.uint8), 1)  # decode buffer to image
                        img_ref = (img_ref[:, :, 2] > 230) * (img_ref[:, :, 0] < 230) * (img_ref[:, :, 1] < 230)
                        # plt.imshow(img_ref)
                        # plt.show()
                        red_indices = np.where(img_ref)
                        col_indices, row_indices = red_indices
                        red_min_x, red_max_x = row_indices.min(), row_indices.max()
                        red_min_y, red_max_y = col_indices.min(), col_indices.max()
                        red_mean_x = (red_min_x + red_max_x) / 2
                        red_mean_y = (red_min_y + red_max_y) / 2
                        bond_img_x.append(red_mean_x)
                        bond_img_y.append(red_mean_y)
                        bond.unhighlight()
            if len(bond_x) < 2:
                print(f'id@{idx}::len(bond_x) < 2')
                graph={}
                success=False
                return img, smiles, graph, success
        except Exception as e:
                print(e,e.args,f'@generate_image with mol_augment {mol_augment}@@#align x y on image!! smiles::{smiles}\n orism::{ori_sm}')
                graph={}
                success=False
                return img, smiles, graph, success
        if color_mol:
            if random.random()>0.75:
                try:#add color drawing
                    indigo1.setOption('render-coloring', True)
                    mol = add_color(indigo1, mol)#TODO testing color here
                    buf = renderer.renderToBuffer(mol)
                except Exception as e:
                    if 'color' in str(e):#unknow color set it be black-white
                        indigo1.setOption('render-coloring', False)
                        buf = renderer.renderToBuffer(mol)
                    else:
                        print(e,e.args,f'@generate_image with mol_augment {mol_augment}@add_color!! smiles::{smiles}\n orism::{ori_sm}')
                        graph={}
                        success=False
                        return img, smiles, graph, success
            else:#even color mol, black-white still need largely as most commen in real world
                indigo1.setOption('render-coloring', False)
                buf = renderer.renderToBuffer(mol)
        else:
            indigo1.setOption('render-coloring', False)
            buf = renderer.renderToBuffer(mol)
        # buf = renderer.renderToBuffer(mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # decode buffer to image

        if len(bond_x) == 0:
            print(f'id@{idx}::len(bond_x) == 0:')
            graph={}
            success=False
            return img, smiles, graph, success

        bond_x = np.array(bond_x)
        bond_y = np.array(bond_y)
        bond_img_x = np.array(bond_img_x)
        bond_img_y = np.array(bond_img_y)
        #for align pupose
        x = (x + bond_img_x.mean() - bond_x.mean()).astype('int32')
        y = (y + bond_img_y.mean() - bond_y.mean()).astype('int32')
        # img = cv2.imread(path, flags=0)
        # img = cv2.resize(img,(512,512))
        # cv2.imwrite(path,img)
        # x = x*(512/image_size)
        # y = y*(512/image_size)
        max_size=max(h,w)
        min_size=min(h,w)
        if min_size/max_size>0.5:
            just_scale=True
        else:
            just_scale=False
        if image_size >=max_size and not just_scale:
            if image_size< h:
                image_size=h
            if image_size< w:
                image_size=w    
            pad_height1 = (image_size-h) // 2
            pad_width1 =  (image_size-w) // 2
            pad_height2 = image_size-h-pad_height1
            pad_width2 =  image_size-w-pad_width1
            x=x+pad_width1
            y=y+pad_height1#update x y if need 
            # Pad the image
            img = np.pad(img, ((pad_height1, pad_height2), (pad_width1, pad_width2),(0, 0) ), mode='constant',constant_values=255)
        # else:
        #     padded_image=img
        #TODO check overlap
        # radius=int(h//100) if int(h//100)<5 else 5#NOTE important if large value 15 causing jj+1 != num atoms
        #NOTE important if large value 15 causing jj+1 != num atoms
        h,w,c=img.shape   
        fig_centers=[(xx,yy) for xx,yy in zip(x,y) ]
        kdtree = KDTree(fig_centers)
        i_xy=dict()
        adj_ij=set()
        for i,center in enumerate(fig_centers):    
            xx,yy=center        
            distance, index = kdtree.query(center, k=2)
            if distance[1] <= radius*2+3 :#reduce the batch_size errors
                # print((index, distance,radius*2),f'{i} center:{center} need update xy:')
                if i>index[1]:
                    ij_=(i,index[1])
                else:
                    ij_=(index[1],i)
                adj_ij.add(ij_)
            i_xy[i]=[xx,yy]

        if len(adj_ij) >0:
            # print(f'id@{idx}::atoms overlap {adj_ij}::{smiles}')
            # graph={}
            # success=False
            # return img, smiles, graph, success
            # not reupdate image, as the indigo stupid blackbox, xy not align with new img
            diffcult_=False
            for jx,ix in adj_ij:
                x_,y_=i_xy[ix]
                x_u,y_u=i_xy[jx]
                exclu_=[(v[1],v[0]) for k,v in i_xy.items() if k!=jx]
                yy,xx=KDupdate_coordinates(y_u,x_u,exclu_,radius,h,w,idx)
                i_xy[jx]=[xx,yy]#update coords
                if (yy,xx)==(y_u,x_u):
                    diffcult_=True
                    break
            if diffcult_:
                print(f'id@{idx}::diffcult_ to update coordinates with overlapping')
                img = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
                graph={}
                success=False
                return img, smiles, graph, success
            else:
                #also update img
                indigo1 = Indigo()
                mol_reload = indigo1.loadMolecule(smiles)
                for i, atom in enumerate(mol_reload.iterateAtoms()):
                    newxyz=i_xy[i][0],i_xy[i][1],0
                    atom.setXYZ(*newxyz)
                renderer = IndigoRenderer(indigo1)
                indigo1.setOption('render-output-format', 'png')
                indigo1.setOption('render-background-color', '1,1,1')
                indigo1.setOption('render-stereo-style', 'none')
                indigo1.setOption('render-label-mode', 'hetero')
                # indigo1.setOption("render-bond-length", bond_length)#NOTE have to apply when sacled
                indigo1.setOption('render-image-size', f'{w},{h}')#AS final size
                # mol_reload.saveMolfile("mol_reload.mol")
                buf = indigo1.writeBuffer()
                buf.sdfAppend(mol_reload)#NOTE stero bond may missing !
                s = buf.toString()
                rdm = Chem.MolFromMolBlock(s,sanitize=False)
                img, graph =rdkit_img(rdm,w,h)
                if graph:
                    success = True
                else:
                    success = False
                return img, smiles, graph, success
                # print(f'udpate img with rdkit')
                # buf = renderer.renderToBuffer(mol_reload)
                # img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # decode buffer to image
        else:
            check_points=False
            if check_points:
                path=f'tmp_{idx}.png'
                plt.imshow(img)
                plt.plot(x, y, 'r.')
                plt.savefig(f'{path}_xy.png')
                plt.close()
            
            # if ((x<=0)|(x>=512)|(y<=0)|(y>=512)).any():
            #     print(f'id@{idx}::too large or too')
                # return None,None
            coords, symbols = [], []
            charges=[]
            index_map = {}
            atoms = [atom for atom in mol.iterateAtoms()]#TODO get atom net charge for using
            # charges=[atom.charge() for atom in mol.iterateAtoms()]
            if shuffle_nodes:
                random.shuffle(atoms)
            for i, atom in enumerate(atoms):
                # x, y, z = atom.xyz()
                index_map[atom.index()] = i
                coords.append([x[atom.index()], y[atom.index()] ])
                symbols.append(atom.symbol())
                charges.append(atom.charge())
                # print([i,x,y,atom.symbol()])
            # bonds_comp = []
            # aromatic_atoms = []
            # for i,bond in enumerate(mol.iterateBonds()):
            #     bond_type = bond.bondOrder()
            #     begin_atom_idx = bond.source().index()
            #     end_atom_idx = bond.destination().index()
            #     if bond_type == 4:
            #         aromatic_atoms.append(begin_atom_idx)
            #         aromatic_atoms.append(end_atom_idx)
            #     y1, x1 = y[begin_atom_idx],x[begin_atom_idx]
            #     y2, x2 = y[end_atom_idx],x[end_atom_idx]#NOTE x--y replaced
            #     xx, yy = (x1 + x2) / 2, (y1 + y2) / 2
            #     assert yy<=512 and xx<=512,f'{[xx,yy,w,h]}why not ????!!'
            #     bonds_comp.append((round(xx),round(yy),bond_type,
            #     begin_atom_idx,end_atom_idx
            #     )) 
            graph = {
            # 'bonds':bonds_comp,#ori
            'id2sid':index_map,
            'coords': coords,#shuffed
            'symbols': symbols,##shuffed
            'charges':charges,##shuffed
            'mol':mol,
                    }
            success= True

            return img, smiles, graph, success
        
    except Exception as e:
        try:
            rdm=Chem.MolFromSmiles(smiles)
            Chem.rdDepictor.Compute2DCoords(rdm)#before rdkit_img need confor 
            img, graph=rdkit_img(rdm,image_size,image_size)
            # print(f'chirality errors fix with rdkit@except generate_image  ')
            if graph:
                success = True
            else:
                success = False
            if 'chirality not possible' in str(e):
                print('@@!!',str(e),'passed with rdkit_img')
        except Exception as ee:
            print(smiles,f'!!! mol_augment {mol_augment}@!!',ee)
            # pass 
            img = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
            graph = {}
            success = False
            # else:
            print(e.__class__.__name__,f'erros:{e}@@!! mol_augment {mol_augment}@!!')
            print(f'ori/smiles:\n{ori_sm}\n{smiles}','!!!')
            # if debug:#check the dataset
            # raise Exception
            # img = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
            # graph = {}
            # success = False
    
    return img, smiles, graph, success#but atom aoverlaping
    # return img,  graph #but atom aoverlaping
    
def generate_image(idx,smiles, mol_augment=True, default_option=False, shuffle_nodes=False,
                          include_condensed=False, debug=False,  addRandom=False,RGROUP_SYMBOLS=RGROUP_SYMBOLS):
    rdkitDrawP=0.5
    insert3=0.5
    image_size=480
    radius= 3
    RDKIT_addedgroup=['[N-]=C=N',"C#N","C#C",'O=[NH+][O-]','[N-]=[N+]=N','C#CC#C',
                      'O=C([C@@H]1[C@H](CC(N)=O)[C@H]2C[C@@H]1C=C2)O',#overlap bonds
                      ]
    
    # i3m=rdkit_addGroup(mol_sm=smiles,cyano_sm='O=C([C@@H]1[C@H](CC(N)=O)[C@H]2C[C@@H]1C=C2)O')#TO avoid the bond connection issue
    # smiles=Chem.MolToSmiles(i3m)
    
    if random.random()>=insert3:
        i3m=rdkit_addGroup(mol_sm=smiles,cyano_sm=random.choice(RDKIT_addedgroup))
        smiles=Chem.MolToSmiles(i3m)

    if random.random()<=rdkitDrawP:
        rdm=Chem.MolFromSmiles(smiles)
        Chem.AddHs(rdm, explicitOnly=True)
        Chem.rdDepictor.Compute2DCoords(rdm)#Note have to in this way get wedge bond
        # print(mol.GetNumConformers())
        img, graph=rdkit_img(rdm,image_size,image_size)
        if graph:
            success = True
        else:
            img, smiles, graph, success=indigo_img(idx,smiles, mol_augment=mol_augment, default_option=default_option, shuffle_nodes=shuffle_nodes,
                          include_condensed=include_condensed, debug=debug,  addRandom=addRandom,RGROUP_SYMBOLS=RGROUP_SYMBOLS,radius=radius,image_size=image_size)

    else:
        img, smiles, graph, success=indigo_img(idx,smiles, mol_augment=mol_augment, default_option=default_option, shuffle_nodes=shuffle_nodes,
                          include_condensed=include_condensed, debug=debug,  addRandom=addRandom,RGROUP_SYMBOLS=RGROUP_SYMBOLS,radius=radius,image_size=image_size)

    return img, smiles, graph, success#but atom aoverlaping



def get_bond_size(keypoints):
    min_distances = []
    for index_keypoint_query, keypoint_query in enumerate(keypoints):
        distances = []
        for index_keypoint_key, keypoint_key in enumerate(keypoints):
            if index_keypoint_key < index_keypoint_query:
                distance = math.dist(keypoint_query, keypoint_key)
                distances.append(distance)

        if len(distances) > 0:
            min_distances.append(min(distances))

    if len(min_distances) > 0:
        # Upper bound estimate of the bond length, as the 75th percentile of minimal distances
        bond_length = np.percentile(min_distances, 75) 
    else:
        print("Get bond size error")
        bond_length = 100
    return bond_length

def get_circle_coordinates(center,h,w, radius, num_points=1):
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=True)
    x = center[1] + radius * np.cos(angles)
    y = center[0] + radius * np.sin(angles)
    coordinates =[(math.ceil(y),math.ceil(x))  for  y, x in list(zip(y, x)) if y>0 and x>0 and y<h and x<w]
    random.shuffle(coordinates)
    return coordinates

def updateOverlap_coordinates(y,x,exclusion_set,radius,h,w):
    # if isinstance(set,exclusion_set):
    if type(exclusion_set) is set:
        exclusion_lis=copy.deepcopy(exclusion_set)
    elif type(exclusion_set) is list:
        exclusion_lis=set(exclusion_set)
    center_yxs = get_circle_coordinates((y,x),h,w, radius=(radius)*2+3, num_points=72)
    for i, yx_new in enumerate(center_yxs):
        circle_points=get_circle_coordinates(yx_new,h,w, radius=radius, num_points=72)
        c_s=set(circle_points)
        if not exclusion_lis.intersection(c_s):#no insert sets
            distance = np.linalg.norm(np.array(yx_new) - np.array((y,x)))
            if yx_new not in exclusion_lis and distance> radius*2:
                return yx_new
    print(f'time costing stopped, try {i+1}times @ updateOverlap_coordinates')
    return y,x

def KDupdate_coordinates(y,x,exclusion_set,radius,h,w,idx,num_points=72):
    # if isinstance(set,exclusion_set):
    if type(exclusion_set) is set:
        exclusion_lis=list(exclusion_set)
    elif type(exclusion_set) is list:
        exclusion_lis=copy.deepcopy(exclusion_set)
    center_yxs = get_circle_coordinates((y,x),h,w, radius=(radius)*2+3, num_points=72)
    for i, yx_new in enumerate(center_yxs):
        # rid=random.choice(range(len(center_yxs)))
        # yx_new=center_yxs[rid]
        exclusion_lis.append(yx_new)
        update_kdt=KDTree(exclusion_lis)
        udate_dist,update_id=update_kdt.query(yx_new,k=2)
        if udate_dist[1]>=(radius)*2+2:
            return yx_new
        else:
            exclusion_lis.pop()
    exp_rad=2
    j=2
    while True:
        if exp_rad>w/4:
            break
        center_yxs = get_circle_coordinates((y,x),h,w, radius=(radius)*2*exp_rad+3, num_points=72)
        exp_rad=exp_rad*2
        for i, yx_new in enumerate(center_yxs):
            # rid=random.choice(range(len(center_yxs)))
            # yx_new=center_yxs[rid]
            exclusion_lis.append(yx_new)
            update_kdt=KDTree(exclusion_lis)
            udate_dist,update_id=update_kdt.query(yx_new,k=2)
            if udate_dist[1]>=(radius)*2+2:
                return yx_new
            else:
                exclusion_lis.pop()
        # print(f'time costing, try {num_points}*{j}times with radius:{radius}@distance{(radius)*2*exp_rad+3}')
        j+=1
    print(f'should skip this strnge molecule idx@@{idx} !!')
    return y,x
######################################################################################################################


class TrainDataset_gz(Dataset):
    def __init__(self, args, df, a_i, split='train', dynamic_indigo=False,
                image_augment=True,
                mol_augment=True,
                default_option=False,
                 ):
        super().__init__()
        self.data_elements={'*'}#NOTE if empty, get errors
        self.df = df
        self.args = args
        self.mol_augment=mol_augment
        self.default_option=default_option
        # if 'file_path' in df.columns:
        #     self.file_paths = df['file_path'].values
        #     if not self.file_paths[0].startswith(args.data_path):
        #         self.file_paths = [os.path.join(args.data_path, path) for path in df['file_path']]

        self.smiles = df['SMILES'].values if 'SMILES' in df.columns else None
        self.transform = get_transforms(args.image_size,  image_augment= args.img_augment)
        # self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])
        self.dynamic_indigo = (dynamic_indigo and split == 'train')
        self.coords_df = None

        self.precomputed_gaussians={}
        self.a_i = a_i
        #charge and edge
        c_=['empty']#or pad
        c_.extend(list(range(-6,6+1,1)))
        self.c_i={ c:i for i, c in enumerate(c_) }#
        self.e_i={ e:i for i,e in enumerate(range(8))}   #  0/1/2/3/4/5/6/7 : empty /single/double/triple/aromatic/WEDGE bond/dash bond/wavy bond/PAD
        self.e_i['PAD']=len(self.e_i)#0/1/2/3/4/-10 as max_padd used in bms_colloate 'em4d'
        self.padding_size=[16,16,16,16]
        self.num_class=len(self.a_i)+len(self.c_i)+len(self.e_i)
        print([len(self.a_i),len(self.c_i),len(self.e_i),self.num_class,self.c_i,self.e_i,self.a_i])

    def __len__(self):
        return len(self.df)

    def image_transform(self, image, coords=[], renormalize=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
        augmented = self.transform(image=image, keypoints=coords)#image augment, such roted blur,GaussNoise ...#Note albumentations Version matter new one may removed the old roate fun
        image = augmented['image']
        if len(coords) > 0:
            coords = np.array(augmented['keypoints'])
            # if renormalize:
            #     coords = normalize_nodes(coords, flip_y=False)
            # else:#NOTE  0~1
            #     _, height, width = image.shape
            #     coords[:, 0] = coords[:, 0] / width
            #     coords[:, 1] = coords[:, 1] / height
            # coords = np.array(coords).clip(0, 1)
            return image, coords
        return image

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            with open(os.path.join(self.args.save_path, f'error_dataset_{int(time.time())}.log'), 'w') as f:
                f.write(str(e))
            raise e
        
    def getitem(self, idx):
        graph_new=dict()
        batch_img=f'data/staker_img/img_{idx}.pt.gz'
        batch_refa=f'data/staker_img/amap_{idx}.pt.gz'
        batch_refb=f'data/staker_img/bmap_{idx}.pt.gz'
        batch_refc=f'data/staker_img/cmap_{idx}.pt.gz'
        with gzip.open(batch_img, 'rb') as fi:
            image=torch.load(fi) 
        with gzip.open(batch_refa, 'rb') as fia:
            graph_new['a_map']=torch.load(fia) 
        with gzip.open(batch_refb, 'rb') as fib:
            graph_new['b_map']=torch.load(fib) 
        with gzip.open(batch_refc, 'rb') as fic:
            graph_new['c_map']=torch.load(fic) 
        _=[]

        return idx, image, graph_new, _
        
    
####################################################################################################


class TrainDataset(Dataset):
    def __init__(self, args, df,  a_i, split='train', dynamic_indigo=False,
                image_augment=True,
                mol_augment=True,
                default_option=False,
                 ):
        super().__init__()
        self.data_elements={'*'}#NOTE if empty, get errors
        self.df = df
        self.args = args
        self.mol_augment=mol_augment
        self.default_option=default_option
        # if 'file_path' in df.columns:
        #     self.file_paths = df['file_path'].values
        #     if not self.file_paths[0].startswith(args.data_path):
        #         self.file_paths = [os.path.join(args.data_path, path) for path in df['file_path']]

        self.smiles = df['SMILES'].values if 'SMILES' in df.columns else None
        self.transform = get_transforms(args.image_size,  image_augment= args.img_augment)
        # self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])
        self.dynamic_indigo = (dynamic_indigo and split == 'train')
        self.coords_df = None

        self.precomputed_gaussians={}
        self.a_i = a_i
        #charge and edge
        c_=['empty']#or pad
        c_.extend(list(range(-6,6+1,1)))
        self.c_i={ c:i for i, c in enumerate(c_) }#
        self.e_i={ e:i for i,e in enumerate(range(8))}   #  0/1/2/3/4/5/6/7 : empty /single/double/triple/aromatic/WEDGE bond/dash bond/wavy bond/PAD
        self.e_i['PAD']=len(self.e_i)#0/1/2/3/4/-10 as max_padd used in bms_colloate 'em4d'
        
        self.padding_size=[16,16,16,16]
        self.num_class=len(self.a_i)+len(self.c_i)+len(self.e_i)
        print([len(self.a_i),len(self.c_i),len(self.e_i),self.num_class,self.c_i,self.e_i,self.a_i])

    def __len__(self):
        return len(self.df)

    def image_transform(self, image, coords=[], renormalize=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
        augmented = self.transform(image=image, keypoints=coords)#image augment, such roted blur,GaussNoise ...#Note albumentations Version matter new one may removed the old roate fun
        image = augmented['image']
        if len(coords) > 0:
            coords = np.array(augmented['keypoints'])
            # if renormalize:
            #     coords = normalize_nodes(coords, flip_y=False)
            # else:#NOTE  0~1
            #     _, height, width = image.shape
            #     coords[:, 0] = coords[:, 0] / width
            #     coords[:, 1] = coords[:, 1] / height
            # coords = np.array(coords).clip(0, 1)
            return image, coords
        return image

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            with open(os.path.join(self.args.save_path, f'error_dataset_{int(time.time())}.log'), 'w') as f:
                f.write(str(e))
            raise e
    ##################################################################################################
    def ace_(self, image,graph,idx,pad=16):#512X512
        c,h,w=image.size()
        sm=graph['smiles']
        # print(c,h,w)
        radius=int(h//100) if int(h//100)<3 else 3#NOTE important if large value 15 causing jj+1 != num atoms
        w_=w-2*pad
        h_=h-2*pad
        keypoints=graph['coords']#from indigo may not compaitable with AumblieA roteing, leading some atom coords miss in the top or bottom
        """
        charge map labeled with by charge_index empty:0,-6~6/1~13 based on staker.csv datset
        """
        labels = torch.zeros(h,w).to(torch.int64).numpy()# 
        map_yx = torch.zeros(h,w).to(torch.int64).numpy()# 
        max_x=max([k[0] for k in keypoints])
        max_y=max([k[1] for k in keypoints])
        if int (max_y)>1 or int (max_y)>1:
            normalized=False
        else:
            normalized=True
        # print(normalized,'@normalized')
        coords_yx=[]
        sphere_yx=set()
        olp_ids=[]
        fig_centers=[]
        for i,center in enumerate(keypoints):            
            if normalized:                                                                                                                                                                                                                      
                x=int(center[0]*(w_))+pad
                y=int(center[1]*(h_))+pad
            else:
                x=int(center[0])#+pad#TODO
                y=int(center[1])#+pad
            fig_centers.append((x,y))
        kdtree = KDTree(fig_centers)
        i_xy=dict()
        adj_ij=set()
        for i,center in enumerate(fig_centers):    
            xx,yy=center        
            distance, index = kdtree.query(center, k=2)
            if distance[1] <= radius*2+3:#NOTE not change xy after graph generated
                # print((index, distance,radius*2),f'{i} center:{center} need update xy:')
                if i>index[1]:
                    ij_=(i,index[1])
                else:
                    ij_=(index[1],i)
                adj_ij.add(ij_)
            i_xy[i]=[xx,yy]

        if len(adj_ij) >0:#avoid >=512
            for jx,ix in adj_ij:
                x_,y_=i_xy[ix]
                x_u,y_u=i_xy[jx]
                exclu_=[(v[1],v[0]) for k,v in i_xy.items() if k!=jx]
                yy,xx=KDupdate_coordinates(y_u,x_u,exclu_,radius,h,w,idx)
                # y,x =KDupdate_coordinates(y,x,[(yy,xx) for ii, xx, yy in excludes_],radius,h,w)
                i_xy[jx]=[xx,yy]#update coords
        update_failed=False
        
        atom_box_class=[]#x1,y1,x2,y2 top left and bottom right  + class
        for i,v in i_xy.items():       
            # top_left_x=x-radius
            # top_left_y=y-radius
            # bottom_right_x=x+radius
            # bottom_right_y=y+radius
            x=v[0]          
            y=v[1]          
            if labels[y-1,x-1]==0:#
                labels[y-1,x-1]=i+1#0~511
                map_yx[y-1,x-1]=i+1  #0~511
                coords_yx.append((y-1,x-1))
                sphere_yx.add((y-1,x-1))
                center_x=x-1
                center_y=y-1
                for ii in range(center_x - radius, center_x + radius + 1):
                    for ji in range(center_y - radius, center_y + radius + 1):
                        if (ii - center_x)**2 + (ji - center_y)**2 <= radius**2:
                            if ii >=w-1:
                                if labels[ ji,w-1]==0:
                                    labels[ ji,w-1] = i+1
                                    sphere_yx.add((ji,w-1))
                            elif ji >=h-1:
                                if labels[h-1,ii] ==0:
                                    labels[h-1,ii] = i+1
                                    sphere_yx.add((h-1,ii))
                            else:
                                if labels[ji,ii] ==0:
                                    labels[ji,ii] = i+1
                                    sphere_yx.add((ji,ii))
            else:#overlap atoms
                olp_ids.append((labels[y-1,x-1],i+1))
                y_new,x_new=updateOverlap_coordinates(y-1,x-1,sphere_yx,radius,h,w)#if not, x=x-1
                if (y_new,x_new)==(y-1,x-1):
                    update_failed=True
                    break
                else:
                    y,x=y_new,x_new
                labels[y,x]=i+1#0~511
                map_yx[y,x]=i+1  #0~511
                coords_yx.append((y,x))
                sphere_yx.add((y,x))
                center_x=x
                center_y=y
                for ii in range(center_x - radius, center_x + radius + 1):
                    for ji in range(center_y - radius, center_y + radius + 1):
                        if (ii - center_x)**2 + (ji - center_y)**2 <= radius**2:
                            if ii >=w-1:
                                if labels[ ji,w-1]==0:
                                    labels[ ji,w-1] = i+1
                                    sphere_yx.add((ji,w-1))
                            elif ji >=h-1:
                                if labels[h-1,ii] ==0:
                                    labels[h-1,ii] = i+1
                                    sphere_yx.add((h-1,ii))
                            else:
                                if labels[ji,ii] ==0:
                                    labels[ji,ii] = i+1
                                    sphere_yx.add((ji,ii))
        if update_failed:  return (False for ii in range(5))


        # atom_box_class=[]#x1,y1,x2,y2 top left and bottom right  + class
        # for i,v in i_xy.items():       
        #     top_left_x=x-radius
        #     top_left_y=y-radius
        #     bottom_right_x=x+radius
        #     bottom_right_y=y+radius

        coords_yxnp=np.array(coords_yx)
        index_new=np.lexsort([coords_yxnp[:,1],coords_yxnp[:,0]])#y first then x asending
        sid2new={ si:i for i,si in enumerate(index_new)}
        yx_incre=coords_yxnp[index_new]#new order y x increasing
        graph['coords']=yx_incre
        sid2ori={v:k for k,v in graph['id2sid'].items()}
        ori2new={ sid2ori[k]:v   for k,v in sid2new.items()}
        #atom and charge update order
        charges=[graph['charges'][i] for i in  index_new]
        atoms=[ graph['symbols'][i] for i in  index_new]
        
        atom_box_class=[]#x1,y1,x2,y2 top left and bottom right  + class
        charge_box_class=[]#x1,y1,x2,y2 top left and bottom right  + class
        for i,v in enumerate(yx_incre):      
            y,x=v[0],v[1] 
            # top_left_x=x-radius
            # top_left_y=y-radius
            # bottom_right_x=x+radius
            # bottom_right_y=y+radius
            # yx= graph['coords'][label-1]
            c_=charges[i]
            a_=atoms[i]
            if a_ not in  self.a_i:
                atoms[i]="*"
                a_='*'
            if c_ not in  self.c_i:#unkown charge
                charges[i]="*"
                c_='*'
            # atom_box_class.append([top_left_x,top_left_y,bottom_right_x,bottom_right_y,self.a_i[a_]])
            # charge_box_class.append([top_left_x,top_left_y,bottom_right_x,bottom_right_y,self.c_i[c_]])
            
            #NOTE use the detr_version loss need box as x1,y1,x2,y2 without rot @@ assert (boxes1[:, 2:] >= boxes1[:, :2]).all()    
            if a_=='C':
                ww,hh=10,10
            elif len(a_)>=2:
                ww,hh=radius*10,radius*10
            else:
                ww,hh=radius*7,radius*7
            
            #  if len(bond_box_class)<=100:
            atom_box_class.append([x,y,ww,hh,self.a_i[a_]])#TODO need lst1 considiering H label expliciy
            charge_box_class.append([x,y,ww,hh,self.c_i[c_]])
        #new order
        # yx_incre=np.argwhere(map_yx > 0)
        # for jj,yx in enumerate(yx_incre):
        #     center_x = yx[1]
        #     center_y = yx[0]
        #     # Step 4: Update values within the sphere's radius
        #     for i in range(center_x - radius, center_x + radius + 1):
        #         for j in range(center_y - radius, center_y + radius + 1):
        #             if (i - center_x)**2 + (j - center_y)**2 <= radius**2:
        #                 if i >=w-1:
        #                     if labels[ j,w-1]==0:
        #                         labels[ j,w-1] = jj+1
        #                 elif j >=h-1:
        #                     if labels[h-1,i] ==0:
        #                         labels[h-1,i] = jj+1
        #                 else:
        #                     if labels[j,i] ==0:
        #                         labels[j,i] = jj+1
        # # map_ = scipy.ndimage.maximum_filter(map_,size=radius)#expand
        # labels, num_labels = scipy.ndimage.label(map_, structure= np.ones((3,3)))#may need dynamicly changed 

        # um_labels not include 0, but labels 0 is bg
        #ndimage.label the order keep same with np.argwhere, [y,x] increasing order
        
        if len(charges)!=len(yx_incre) or len(atoms)!=len(yx_incre):
            pil_image=to_img(image)
            sm=graph['sm']
            sf=f'{sm}_{len(charges)}@{len(yx_incre)}@{len(atoms)}_@{idx}'
            pil_image.save(sf+'.png')
            plt.imshow(labels)#so the labels order from 0 to 20 same as 
            plt.savefig(fname=sf+'map.png')

            plt.imshow(labels,cmap='rainbow')#so the labels order from 0 to 20 same as 
            plt.colorbar()
            plt.savefig(fname=sf+'labels.png')

            print(f"why not equal ? should same num as atoms {len(charges)}@{len(yx_incre)}@{len(atoms)}")
            # assert len(atoms)==jj+1, f"why not equal ? should same num as atoms {len(atoms)}@{jj+1}"
        # assert len(charges) ==len(atoms), f"why not equal ? should same num as atoms {len(charges)}@{jj+1}@{len(atoms)}"
        #edges
        charge_coordsmap=deepcopy(labels)
        atom_coordsmap=deepcopy(labels)
        labeled_array, num_features = scipy.ndimage.label(labels, structure= np.ones((3,3)))#may need dynamicly changed 
        # carbon=np.where(labels > 0, 1, 0)
        # center_list = scipy.ndimage.measurements.center_of_mass(carbon, labeled_array,list(range(1, num_features+1)))
        
        for label in range(1, len(atoms) +1):#NOTE  0 is empty or padding
            c_=charges[label-1]
            a_=atoms[label-1]
            # yx= graph['coords'][label-1]
            if a_ not in  self.a_i:
                atoms[label-1]="*"
                a_='*'
            if c_ not in  self.c_i:#unkown charge
                charges[label-1]="*"
                c_='*'
            # charge_coordsmap[contour_pixel[0]]=c_i[c_]#wrong way
            #labels use the sid index
            contour_mask = labeled_array == label       
            contour_pixel = np.argwhere(contour_mask)
            # if center_list[label-1] in contour_pixel:#NOTEO labels use the shufftle orders
            for y,x in contour_pixel:
                charge_coordsmap[y,x]=self.c_i[c_]
                atom_coordsmap[y,x]=self.a_i[a_]
            # atom_coordsmap[contour_pixel]=self.a_i[a_]
        c_max=max(np.unique(charge_coordsmap))
        if c_max>13:
            print(np.unique(atom_coordsmap),'atom_charge',[c_,label-1,self.c_i[c_],self.c_i])

        #udate grapher
        graph['symb']=graph['symbols']  
        #symbols 2 idex  
        graph['charges']=[self.c_i[c_] for c_ in charges]
        graph['symbols']=[self.a_i[a_] for a_ in  atoms]

        # sid_Order={yx:i for i,yx in enumerate(coords_yx)}
        #NOTE watch out the order changed np.argwhere keep same with ndimage.label [y,x] increasing 
        # yx_incre=np.argwhere(map_yx > 0)
        # increOrder={yx:i for i,yx in enumerate(yx_incre)}
        # sid2new={sid_Order[(yx[0],yx[1])]:i for i,yx in enumerate(yx_incre)}
        n=len(sid2new)
        sid2ori={v:k for k,v in graph['id2sid'].items()}
        ori2new={ sid2ori[k]:v   for k,v in sid2new.items()}

        mol=graph['mol']#TODO update mols coords setxyz() and get new image

        # atoms = [atom for atom in mol.iterateAtoms()]#TODO get atom net charge for using
        # for i, atom in enumerate(atoms):
        #     new_xy=yx_incre[ori2new[i]]
        #     newxyz=new_xy[1],new_xy[0],0
        #     atom.setXYZ(*newxyz)
        edges = np.zeros((n, n), dtype=int)
        bonds_comp=[]
        if isinstance(mol,indigo.indigo.indigo_object.IndigoObject):
            # elements={atom.symbol() for i, atom in enumerate(mol.iterateAtoms())}
            # self.data_elements.update(elements)
            for bond in mol.iterateBonds():#TODO now we not supporting  the unk bonding currently, adding it next version
                s = ori2new[bond.source().index()]
                t = ori2new[bond.destination().index()]
                y1,x1=yx_incre[s]
                y2,x2=yx_incre[t]
                yy,xx=(y1+y2)/2,(x1+x2)/2
                bond_type=bond.bondOrder()
                edges[s, t] = bond.bondOrder()
                edges[t, s] = bond.bondOrder()#TODO assign the stero Center auto with Fragmentstein or molscribe postprocess
                if bond.bondStereo() in [5, 6]:#5 wedit 6 dash
                    edges[s, t] = bond.bondStereo()
                    edges[t, s] = 11 - bond.bondStereo()
                    bond_type=bond.bondStereo()
                bonds_comp.append((round(xx),round(yy),bond_type,[x1,y1,x2,y2]))
                # [
                # Indigo.UP ,
                # Indigo.DOWN ,
                # Indigo.EITHER ,
                # Indigo.CIS ,
                # Indigo.TRANS,
                # ]
                #[5, 6, 4, 7, 8]
                # 1/2/3/4 : single/double/triple/aromatic https://lifescience.opensource.epam.com/indigo/api/index.html?highlight=bondorder
                # if bond.bondStereo() in [5, 6]:#up,down
                #     #edges[s, t] = bond.bondStereo()
                #     edges[t, s] = 11 - bond.bondStereo()
        elif isinstance(mol,rdkit.Chem.rdchem.Mol):
            # elements={atom.GetSymbol() for atom in mol.GetAtoms()}
            # self.data_elements.update(elements)

            #NOTE mol is from molblock that has the 1 conformer,
            # if mol from smiles we need Chem.rdDepictor.Compute2DCoords(mol) get the 2d conformer then WedgeMolBonds
            #rdDepictor.Compute2DCoords(mol)
            Chem.rdmolops.WedgeMolBonds(mol,mol.GetConformer(0))
            for jj,bond in enumerate(mol.GetBonds()):
                s = ori2new[bond.GetBeginAtomIdx()]#rdkit not random atom order now
                t = ori2new[bond.GetEndAtomIdx()]
                edges[s, t] = rdkitbond_type_dict[bond.GetBondType()]#bond.GetBondType() aromatic bond will return 12
                edges[t, s] = rdkitbond_type_dict[bond.GetBondType()]
                if  bond.GetBondDir() in [rdkit.Chem.rdchem.BondDir.BEGINWEDGE,
                                          rdkit.Chem.rdchem.BondDir.BEGINDASH,
                                          rdkit.Chem.rdchem.BondDir.UNKNOWN]:#NOTE indigo now no wavy bond only rdkit
                    bond_type=rdkitbond_type_dict[bond.GetBondDir()]
                else:
                    bond_type=rdkitbond_type_dict[bond.GetBondType()]
                y1,x1=yx_incre[s]
                y2,x2=yx_incre[t]
                yy,xx=(y1+y2)/2,(x1+x2)/2
                # bonds_comp.append((round(xx),round(yy),bond_type))
                bonds_comp.append((round(xx),round(yy),bond_type,[x1,y1,x2,y2]))

                # rdkitbond_type_dict[bond.GetBondType()]
        else:
            print('unkown methods???')
            # raise
        #update image if need
        if len(olp_ids)>0:
            '''overlaping atoms exists, need update image'''
            trans_list = []
            trans_list +=[
                        # A.Downscale(scale_min=0.2, scale_max=0.5, interpolation=3),
                        # A.Blur(),
                        # A.GaussNoise(),
                        # SaltAndPepperNoise(num_dots=20, p=0.5),#return image2
                        # A.ToGray(p=1),#why have to ? easy training??
                        # A.Normalize(mean=mean, std=std),
                        ToTensorV2(),#return image3
                        ]
            ovelap_transImage=A.Compose(trans_list)
            indigo1 = Indigo()
            mol_reload = indigo1.loadMolecule(sm)
            for i, atom in enumerate(mol_reload.iterateAtoms()):
                new_xy=yx_incre[ori2new[i]]
                newxyz=new_xy[1],new_xy[0],0
                atom.setXYZ(*newxyz)
            # mol.layout()#NOTE this assign imgage coords to atoms
            renderer = IndigoRenderer(indigo1)
            indigo1.setOption('render-output-format', 'png')
            indigo1.setOption('render-background-color', '1,1,1')
            indigo1.setOption('render-stereo-style', 'none')
            indigo1.setOption('render-label-mode', 'hetero')
            indigo1.setOption('render-image-size', f'{h},{w}')#AS final size
            indigo1.setOption('render-relative-thickness', 10)
            indigo1.setOption('render-bond-line-width', 2)
            p=random.uniform(0.0, 1)
            if p>0.45:
                indigo1.setOption('render-coloring', True)
            # mol_reload.saveMolfile("mol_reload.mol")
            buf = renderer.renderToBuffer(mol_reload)
            # renderer.renderToFile(mol_reload, 'mol_reload.png')
            img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # decode buffer to image
            trans_ovl=ovelap_transImage(image=img)
            # to_img(trans_ovl['image'])#4 checking 
            image=trans_ovl['image'] #NOTE may exist crossing bonds for simple way of avoiding the overlap
            print(len(olp_ids),f'overlaping atoms exists!! also update image, has {i}@atoms, {sm},newImage{image.size()}')
        
        #TODO add bond map, as previous may too complex
        graph['bonds']=bonds_comp
        bond_coordsmap=np.zeros((h,w))
        #NOTE do not know why pixel cluset size not equal, now let bond at same level with atom
        # labeled_image, num_labels  = scipy.ndimage.label(atom_coordsmap )
        # component_sizes = scipy.ndimage.sum(atom_coordsmap, labeled_image, range(1, num_labels+1))
        # unique_values, value_counts = np.unique(component_sizes, return_counts=True)
        # most_frequent_index = np.argmax(value_counts)
        # bondr=round(math.sqrt(unique_values[most_frequent_index] / math.pi))-2
        # bondr=4
        # Generate a blank white image for calculation purposes
        img = Image.new("RGB", (h,w), "white")
        # draw = ImageDraw.Draw(img)

        bond_length=100#NOTE hard adjust not use the depednet bond change
        bond_box_class=[]
        for bondcomp in graph['bonds']:#TODO bond overlap
            midpoint=bondcomp[:2]
            x1,y1,x2,y2=bondcomp[-1]
            dy,dx=y2-y1, x2-x1
            bond_box_class.append(list(midpoint)+[abs(dx*0.7) if abs(dx)>5 else 5,
                                                  abs(dy*0.7) if abs(dy)>5 else 5,bondcomp[2]])# Convert boxes to the format [x_center, y_center, width, height]

            # angle = np.degrees(np.arctan2(dy,dx))
            # rectangle_length = np.linalg.norm([dx-2*radius,dy-2*radius])#TODO may need - radius*2
            # rectangle_width=rectangle_length
            # rectangle_length = bond_length * 0.3
            # rectangle_width = bond_length * 0.06
            # Calculate corners of the rectangle
            # corners = [
            # [midpoint[0] - rectangle_length / 2, midpoint[1] - rectangle_width / 2],#up left
            # [midpoint[0] + rectangle_length / 2, midpoint[1] - rectangle_width / 2],
            # [midpoint[0] + rectangle_length / 2, midpoint[1] + rectangle_width / 2],#bottom right
            # [midpoint[0] - rectangle_length / 2, midpoint[1] + rectangle_width / 2]
            # ]
            # bond_box_class.append(corners[0] + corners[-2] +[bondcomp[2]])# loss need this box version
            
            #draw rectangel on bond
            # rot_corners = [
            # ((corner[0] - midpoint[0]) * np.cos(np.radians(angle)) - (corner[1] - midpoint[1]) * np.sin(np.radians(angle)) + midpoint[0],
            #  (corner[0] - midpoint[0]) * np.sin(np.radians(angle)) + (corner[1] - midpoint[1]) * np.cos(np.radians(angle)) + midpoint[1])
            # for corner in corners
            # ] 

            # Draw the rectangle on the image
            # draw.polygon(rot_corners, outline='red')
            # Collect pixels within this rectangle
            # bond_pixels = []
            # for x in range(int(min(rot_corners, key=lambda t: t[0])[0]), int(max(rot_corners, key=lambda t: t[0])[0]) + 1):
            #     for y in range(int(min(rot_corners, key=lambda t: t[1])[1]), int(max(rot_corners, key=lambda t: t[1])[1]) + 1):
            #         if point_in_polygon(x, y, rot_corners):
            #             # bond_pixels.append((x, y))
            #             if y>511 or x>511:continue
            #             else: bond_coordsmap[y,x]=int(bondcomp[2])
            
            # rectangles_pixels.append(bond_pixels)
            # x,y=bondcomp[:2]
            # # bond_coordsmap[y-bondr:y+bondr+1,x-bondr:x+bondr+1]=int(bondcomp[2])#square, we need sphere shape
            # for ii in range(x - bondr, x + bondr + 1):
            #     for ji in range(y - bondr, y + bondr + 1):
            #         if (ii - x)**2 + (ji - y)**2 <= bondr**2:
            #             bond_coordsmap[ji,ii]=int(bondcomp[2])

        # abc_dict={'a':atom_coordsmap,'b':bond_coordsmap,'c':charge_coordsmap}
        # for kk,vv in abc_dict.items():
        #     labeled_image, num_labels  = scipy.ndimage.label(vv )
        #     component_sizes = scipy.ndimage.sum(vv, labeled_image, range(1, num_labels+1))
        #     print(component_sizes,f'{kk} size')
        # if len(bond_box_class)<=100:
        graph['bond_box_class']=np.array(bond_box_class)
        graph['atom_box_class']=np.array(atom_box_class)
        graph['charge_box_class']=np.array(charge_box_class)
        return  image, atom_coordsmap, charge_coordsmap, bond_coordsmap, edges,graph#now edge atom index keep same [y,x] increasing order
        # return  image, edges,graph#now edge atom index keep same [y,x] increasing order

    def ace_2(self, image,graph,idx,pad=16):#512X512
            c,h,w=image.size()
            sm=graph['smiles']
            radius=int(h//100) if int(h//100)<3 else 3#NOTE important if large value 15 causing jj+1 != num atoms
            w_=w-2*pad
            h_=h-2*pad
            keypoints=graph['coords']#from indigo may not compaitable with AumblieA roteing, leading some atom coords miss in the top or bottom
            """
            charge map labeled with by charge_index empty:0,-6~6/1~13 based on staker.csv datset
            """
            labels = torch.zeros(h,w).to(torch.int64).numpy()# 
            map_yx = torch.zeros(h,w).to(torch.int64).numpy()# 
            max_x=max([k[0] for k in keypoints])
            max_y=max([k[1] for k in keypoints])
            if int (max_y)>1 or int (max_y)>1:
                normalized=False
            else:
                normalized=True
            # print(normalized,'@normalized')
            coords_yx=[]
            fig_centers=[]
            for i,center in enumerate(keypoints):            
                if normalized:                                                                                                                                                                                                                      
                    x=int(center[0]*(w_))+pad
                    y=int(center[1]*(h_))+pad
                else:
                    x=int(center[0])#+pad#TODO
                    y=int(center[1])#+pad
                fig_centers.append((x,y))
                coords_yx.append((y,x))
            coords_yxnp=np.array(coords_yx)
            index_new=np.lexsort([coords_yxnp[:,1],coords_yxnp[:,0]])#y first then x asending
            sid2new={ si:i for i,si in enumerate(index_new)}
            yx_incre=coords_yxnp[index_new]#new order y x increasing
            graph['coords']=yx_incre
            sid2ori={v:k for k,v in graph['id2sid'].items()}
            ori2new={ sid2ori[k]:v   for k,v in sid2new.items()}
            #atom and charge update order
            charges=[graph['charges'][i] for i in  index_new]
            atoms=[ graph['symbols'][i] for i in  index_new]
            
            atom_box_class=[]#x1,y1,x2,y2 top left and bottom right  + class
            charge_box_class=[]#x1,y1,x2,y2 top left and bottom right  + class
            for i,v in enumerate(yx_incre):      
                y,x=v[0],v[1] 
                c_=charges[i]
                a_=atoms[i]
                if a_ not in  self.a_i:
                    atoms[i]="*"
                    a_='*'
                if c_ not in  self.c_i:#unkown charge
                    charges[i]="*"
                    c_='*'
                #NOTE use the detr_version loss need box as x1,y1,x2,y2 without rot @@ assert (boxes1[:, 2:] >= boxes1[:, :2]).all()    
                if a_=='C':
                    ww,hh=10,10
                elif len(a_)>=2:
                    ww,hh=radius*10,radius*10
                else:
                    ww,hh=radius*7,radius*7
                
                #  if len(bond_box_class)<=100:
                atom_box_class.append([x,y,ww,hh,self.a_i[a_]])#TODO need lst1 considiering H label expliciy
                charge_box_class.append([x,y,ww,hh,self.c_i[c_]])
            
            if len(charges)!=len(yx_incre) or len(atoms)!=len(yx_incre):
                pil_image=to_img(image)
                sm=graph['sm']
                sf=f'{sm}_{len(charges)}@{len(yx_incre)}@{len(atoms)}_@{idx}'
                pil_image.save(sf+'.png')
                plt.imshow(labels)#so the labels order from 0 to 20 same as 
                plt.savefig(fname=sf+'map.png')

                plt.imshow(labels,cmap='rainbow')#so the labels order from 0 to 20 same as 
                plt.colorbar()
                plt.savefig(fname=sf+'labels.png')

                print(f"why not equal ? should same num as atoms {len(charges)}@{len(yx_incre)}@{len(atoms)}")
                # assert len(atoms)==jj+1, f"why not equal ? should same num as atoms {len(atoms)}@{jj+1}"
            # assert len(charges) ==len(atoms), f"why not equal ? should same num as atoms {len(charges)}@{jj+1}@{len(atoms)}"
            graph['symb']=graph['symbols']  
            #symbols 2 idex  
            graph['charges']=[self.c_i[c_] for c_ in charges]
            graph['symbols']=[self.a_i[a_] for a_ in  atoms]
            # sid2new={sid_Order[(yx[0],yx[1])]:i for i,yx in enumerate(yx_incre)}
            n=len(sid2new)
            sid2ori={v:k for k,v in graph['id2sid'].items()}
            ori2new={ sid2ori[k]:v   for k,v in sid2new.items()}

            mol=graph['mol']#TODO update mols coords setxyz() and get new image

            edges = np.zeros((n, n), dtype=int)
            bonds_comp=[]
            if isinstance(mol,indigo.indigo.indigo_object.IndigoObject):
                # elements={atom.symbol() for i, atom in enumerate(mol.iterateAtoms())}
                # self.data_elements.update(elements)
                for bond in mol.iterateBonds():#TODO now we not supporting  the unk bonding currently, adding it next version
                    s = ori2new[bond.source().index()]
                    t = ori2new[bond.destination().index()]
                    y1,x1=yx_incre[s]
                    y2,x2=yx_incre[t]
                    yy,xx=(y1+y2)/2,(x1+x2)/2
                    bond_type=bond.bondOrder()
                    edges[s, t] = bond.bondOrder()
                    edges[t, s] = bond.bondOrder()#TODO assign the stero Center auto with Fragmentstein or molscribe postprocess
                    if bond.bondStereo() in [5, 6]:#5 wedit 6 dash
                        edges[s, t] = bond.bondStereo()
                        edges[t, s] = 11 - bond.bondStereo()
                        bond_type=bond.bondStereo()
                    bonds_comp.append((round(xx),round(yy),bond_type,[x1,y1,x2,y2]))

            elif isinstance(mol,rdkit.Chem.rdchem.Mol):
                Chem.rdmolops.WedgeMolBonds(mol,mol.GetConformer(0))
                for jj,bond in enumerate(mol.GetBonds()):
                    s = ori2new[bond.GetBeginAtomIdx()]#rdkit not random atom order now
                    t = ori2new[bond.GetEndAtomIdx()]
                    edges[s, t] = rdkitbond_type_dict[bond.GetBondType()]#bond.GetBondType() aromatic bond will return 12
                    edges[t, s] = rdkitbond_type_dict[bond.GetBondType()]
                    if  bond.GetBondDir() in [rdkit.Chem.rdchem.BondDir.BEGINWEDGE,
                                            rdkit.Chem.rdchem.BondDir.BEGINDASH,
                                            rdkit.Chem.rdchem.BondDir.UNKNOWN]:#NOTE indigo now no wavy bond only rdkit
                        bond_type=rdkitbond_type_dict[bond.GetBondDir()]
                    else:
                        bond_type=rdkitbond_type_dict[bond.GetBondType()]
                    y1,x1=yx_incre[s]
                    y2,x2=yx_incre[t]
                    yy,xx=(y1+y2)/2,(x1+x2)/2
                    # bonds_comp.append((round(xx),round(yy),bond_type))
                    bonds_comp.append((round(xx),round(yy),bond_type,[x1,y1,x2,y2]))
                    # rdkitbond_type_dict[bond.GetBondType()]
            else:
                print('unkown methods???')
            #TODO add bond map, as previous may too complex
            graph['bonds']=bonds_comp
            img = Image.new("RGB", (h,w), "white")
            # draw = ImageDraw.Draw(img)
            bond_length=100#NOTE hard adjust not use the depednet bond change
            bond_box_class=[]
            for bondcomp in graph['bonds']:#TODO bond overlap
                midpoint=bondcomp[:2]
                x1,y1,x2,y2=bondcomp[-1]
                dy,dx=y2-y1, x2-x1
                bond_box_class.append(list(midpoint)+[abs(dx*0.7) if abs(dx)>5 else 5,
                                                    abs(dy*0.7) if abs(dy)>5 else 5,bondcomp[2]])# Convert boxes to the format [x_center, y_center, width, height]
            graph['bond_box_class']=np.array(bond_box_class)
            graph['atom_box_class']=np.array(atom_box_class)
            graph['charge_box_class']=np.array(charge_box_class)
            return  image, edges,graph#now edge atom index keep same [y,x] increasing order    
    ##################################################################################################
    def getitem(self, idx):
        # ref = {}
        f=0
        if self.dynamic_indigo:
            # begin = time.time()
            # print([self.smiles[idx], self.args.mol_augment, self.args.default_option,\
            ##indigo graph atom 顺序与输入的SMILES 相同，RDKIT 的graph atom index 则不同与输入的SMILES 而是RDKIT 标准化的SMILES
            image_np, smiles, graph, success = generate_image(idx,
                self.smiles[idx], mol_augment=self.mol_augment, default_option=self.default_option,
                shuffle_nodes=self.args.shuffle_nodes, debug=True,
                include_condensed=self.args.include_condensed,RGROUP_SYMBOLS=RGROUP_SYMBOLS)
                # include_condensed=self.args.include_condensed,image_size=self.args.image_size)
            
            if not success:
                f+=1
                if f>20:
                    print(f,f'failed {f} times')
                return idx, None, {}, None
            
            graph['smiles']=smiles

            # raw_image = image
            # if idx < 30 and 
            if self.args.save_image:
                path = os.path.join(self.args.save_path, 'images')
                os.makedirs(path, exist_ok=True)
                cv2.imwrite(os.path.join(path, f'{idx}.png'), image_np)
            # image_np==
            
            #iamge is tensor now with image aug update from above indigo mol
            image, coords = self.image_transform(image_np, graph['coords'], renormalize=False)#resize,keep x,y pixel not 0
            # image = F.pad(image, self.padding_size)#TO make the frontier atom y x less
            graph['coords'] = coords#update rotated,rescaled image coords keep same with img
            # ref['charges']=graph['charges']
            #molgrpah
            # heatmap=self.generate_heatmap(image,coords)
            # map_a,map_c,edges,graph_new =self.ace_(image,graph,idx)#TODO np.argwhere
            # image, map_a, map_c,map_b, edges,graph_new=self.ace_(image,graph,idx)#update image
            image, edges,graph_new=self.ace_2(image,graph,idx)#update image
            
            # if isinstance(map_a,bool):
            #     f+=1
            #     if f>20:
            #         print(f,f'failed {f} times')
            #     return idx, None, {}, None
            # print('image',image)
            # if not image:
            #     return idx,image, graph_new
            #TODO try def data set 4 training
            # graph_new['a_map']=torch.from_numpy(map_a)#tensor merge the heatmap with charge, atom  class
            # graph_new['c_map']=torch.from_numpy(map_c)#tensor merge the heatmap with charge, atom  class
            # #  0/1/2/3/4/5/6 : empty/single/double/triple/aromatic/R/STERO
            # graph_new['b_map'] = torch.from_numpy(map_b)# can not present unk bonding with wave
           
            graph_new['edges'] = torch.from_numpy(edges)# can not present unk bonding with wave
            del(graph)
            return idx, image, graph_new, image_np
        
# class AuxTrainDataset(Dataset):

#     def __init__(self, args, train_df, aux_df, tokenizer):
#         super().__init__()
#         self.train_dataset = TrainDataset(args, train_df, tokenizer, dynamic_indigo=args.dynamic_indigo)
#         self.aux_dataset = TrainDataset(args, aux_df, tokenizer, dynamic_indigo=False)

#     def __len__(self):
#         return len(self.train_dataset) + len(self.aux_dataset)

#     def __getitem__(self, idx):
#         if idx < len(self.train_dataset):
#             return self.train_dataset[idx]
#         else:
#             return self.aux_dataset[idx - len(self.train_dataset)]


def pad_images(imgs):
    # B, C, H, W
    max_shape = [0, 0]
    for img in imgs:
        for i in range(len(max_shape)):
            max_shape[i] = max(max_shape[i], img.shape[-1 - i])
    stack = []
    for img in imgs:
        pad = []
        for i in range(len(max_shape)):
            pad = pad + [0, max_shape[i] - img.shape[-1 - i]]
        stack.append(F.pad(img, pad, value=0))
    return torch.stack(stack)


def bms_collate(batch):#Note DataLoad need
    PAD_IDac=0
    PAD_IDyx=-10
    
    ids = []
    imgs = []
    batch = [ex for ex in batch if ex[1] is not None]
    formats = list(batch[0][2].keys())
    # if len(batch)>0:
    # else:
    #     print(batch,'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    #     return None   

    seq_formats = [k for k in formats if
                   k in ['coords', 
                        #  'symb',#use as accumalate  traininng type elemets accounts
                         'symbols', 'charges','smiles']]
    
    # # refs = {key: [[], []] for key in seq_formats}
    refs = {key: [[], []] for key in seq_formats}
    for ex in batch:
        ids.append(ex[0])
        imgs.append(ex[1])
        ref = ex[2]
        for key in seq_formats:
            if key=='smiles':
                refs[key][0].append(ref[key])
                refs[key][1].append([len(ref[key])])
            else:
                refs[key][0].append(torch.LongTensor(ref[key]))
                refs[key][1].append(torch.LongTensor([len(ref[key])]))
    # Sequence
    for key in seq_formats:
        if key =='coords':
            refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=PAD_IDyx)#B, T, N` is the length of the longest sequence
            refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)
        # this padding should work, each of which has shape (length, 4)
        elif key=='smiles':
            pass
        else:
            refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=PAD_IDac)
            refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)
    

    # # print(refs['edges'].size(),'@bms')
    # if 'a_map' in formats:
    #     heatmap_list = [ex[2]['a_map'] for ex in batch]#same as rescaled image size 512X512
    #     refs['a_map'] = torch.stack(heatmap_list,dim=0).to(torch.int64)
    
    # if 'b_map' in formats:
    #     heatmap_list = [ex[2]['b_map'] for ex in batch]#same as rescaled image size 512X512
    #     refs['b_map'] = torch.stack(heatmap_list,dim=0).to(torch.int64)

    # if 'c_map' in formats:
    #     heatmap_list = [ex[2]['c_map'] for ex in batch]#same as rescaled image size 512X512
    #     refs['c_map'] = torch.stack(heatmap_list,dim=0).to(torch.int64)

    #account used
    # symbols={'C'}
    # if 'symb' in formats:
    #     for ex in batch:
    #        symbols.update(set(ex[2]['symb'])) 
    #     refs['symb']=symbols
    # images=pad_images(imgs)

    #accout the heavy atom each mol
    symbols_num=[]
    if 'symb' in formats:
        for ex in batch:
            symbols_num.append(len(ex[2]['symb']))
            # symbols.update(set(ex[2]['symb'])) 
    refs['symb']=symbols_num

    images=pad_images(imgs)
    # if "atom_box_class" in formats:#centx,centy,w,h,class
    #     abox=[ex[2]['atom_box_class'] for ex in batch]
    #     cbox=[ex[2]['charge_box_class'] for ex in batch]
    #     bbox=[ex[2]['bond_box_class'] for ex in batch]
    #     print(abox)
    boxx=True
    if boxx:
        refs['targets_a'] = [
        {"labels": torch.from_numpy(ex[2]['atom_box_class'][:, 4]).type(torch.LongTensor), 
            "boxes" : torch.from_numpy(ex[2]['atom_box_class'][:, :4]).type(torch.FloatTensor), } 
        for ex in batch
    ]
        refs['targets_c'] = [
        {"labels": torch.from_numpy(ex[2]['charge_box_class'][:, 4]).type(torch.LongTensor), 
            "boxes" : torch.from_numpy(ex[2]['charge_box_class'][:, :4]).type(torch.FloatTensor), } 
        for ex in batch
    ]
        refs['targets_b'] = [
        {"labels": torch.from_numpy(ex[2]['bond_box_class'][:, 4]).type(torch.LongTensor), 
            "boxes" : torch.from_numpy(ex[2]['bond_box_class'][:, :4]).type(torch.FloatTensor), } 
        for ex in batch
    ]
    # refs['targets_a']=[]
    # for ex in batch:
    #     np_abox=(ex[2]['atom_box_class'])
    #     labels=  torch.from_numpy(np_abox[:, 4]).type(torch.LongTensor)
    #     boxes=torch.from_numpy(np_abox[:, :4]).type(torch.FloatTensor)
    #     # print(labels)
    #     refs['targets_a'].append({'labels':labels,'boxes':boxes})
    # concats=torch.cat(( images,refs['a_map'].unsqueeze(1),refs['c_map'].unsqueeze(1)),1)
    # b_cat,c_cat,h_cat,w_cat=concats.size()
    # n_at=[]
    # for b,cmap in enumerate(refs['c_map']):
    #     atoms_=[]
    #     cmap=cmap.cpu().numpy()
    #     labels_c, num_labelcs = scipy.ndimage.label(cmap)#, structure= np.ones((3,3)))#num_labels not include 0, but labels 0 is bg
    #     for lab in range(1, num_labelcs+1):
    #         contour_mask=labels_c==lab#0 for pading, atom label start from1
    #         atoms_.append(torch.from_numpy(contour_mask).expand(c_cat,h_cat,w_cat))
    #     n_at.append((num_labelcs,atoms_))
    # mols_batch=[]
    # failed_b=[]
    # for b,la in enumerate(n_at):
    #     l,atoms_list=la
    #     # print(l,len(atoms_list))
    #     try:
    #         atoms_stack=torch.stack(atoms_list,dim=0)
    #         catb=concats[b].expand(l, c_cat, h_cat, w_cat)
    #         tmp_selected=torch.masked_select(catb,atoms_stack)#TODO some atoms may get more masks 
    #         mol_m=tmp_selected.reshape(l,c_cat,-1)
    #         mols_batch.append(mol_m)
    #     except Exception as e:
    #         # print(e,f'@@@@failed {failed}')
    #         print(atoms_stack.size(),concats[b].size(),[tmp_selected.size(),l,c_cat,tmp_selected.size()[0]/(l*c_cat)])
    #         print(catb.size(),atoms_stack.size(),[b,l,tmp_selected.size()],refs['smiles'],f'@{b} item or image')#may not 317??
    #         failed_b.append(b)
    #         pass
    # emb4d=pad_sequence(mols_batch,  batch_first=True, padding_value=5)#bond pad with 5
    # # Edges
    # if 'edges' in formats:
    #     edges_list=[]
    #     for b,ex in enumerate(batch):
    #         if b not in failed_b:
    #             edges_list.append(ex[2]['edges'])

    #     max_len = max([len(edges) for edges in edges_list])
    #     refs['edges'] = torch.stack(
    #         [F.pad(edges, (0, max_len - len(edges), 0, max_len - len(edges)), value=5) for edges in edges_list],
    #         dim=0).to(torch.int64)

    # # target_dict={'a':refs['a_map'],'b':refs['edges'],'c':refs['c_map'],'emb4d':emb4d}
    # refs['emb4d']=emb4d
    ori_imgs=[ex[3] for ex in batch]

    return ids, images, refs,ori_imgs


import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import MolDrawOptions
import lxml.etree as et
from io import BytesIO,StringIO
import cairosvg
from PIL import Image
import cssutils

#rdkig image
def rdkit_img(rdm,w,h,number_querry=100):
    min_y,min_x,scale=0,0,1
    mol = Draw.rdMolDraw2D.PrepareMolForDrawing(rdm)
    rdkit.Chem.Draw.rdMolDraw2D.MolToACS1996SVG(mol)
    sio = StringIO()
    writer = Chem.SDWriter(sio)
    writer.write(mol)
    writer.close()
    mt_data = sio.getvalue()
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(w, h)
    options = MolDrawOptions()
    # options.bgColor=(255, 255, 255)
    if np.random.rand()<=0.45:
        options.useBWAtomPalette()#No colorful
        options.comicMode=True
        # options.bondLineWidth = np.random.randint(4,6)
    options.bondLineWidth = random.choice([0.5,1,2,3])
    options.padding = np.random.uniform(0.05,0.4)
    options.additionalAtomLabelPadding = np.random.uniform(0.05,0.25)
    options.minFontSize = 22
    #simulate hand-drawn lines for bonds
    # options.atomLabelFontFace=
    wavy_bondP=0.6
    if np.random.rand()<=wavy_bondP:
        #rdkit draw wavy bonds: Set wavy bonds around STEREOANY stereo
        for bond in rdm.GetBonds():
            # Select a single bond to transform to double
            if (bond.GetBondType() == Chem.rdchem.BondType.SINGLE) and (bond.GetStereo() == 0) and (bond.GetIsAromatic() == False):
                match = True
                # Verify that neighboring atoms are simple carbons
                for atom_index in [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]:
                    atom = rdm.GetAtomWithIdx(atom_index)
                    if (atom.GetNumImplicitHs() <= 1) or (atom.GetSymbol() != "C") or \
                            (atom.GetFormalCharge() != 0) or atom.IsInRing() or atom.HasProp("_displayLabel"):
                        match = False
                        break
                if not match:
                    continue      
                # s = bond.GetBeginAtomIdx()#rdkit not random atom order now
                # t = bond.GetEndAtomIdx() 
                # print((s,t))
                bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
                bond.SetStereo(Chem.rdchem.BondStereo.STEREOANY) 
                # Set wavy bonds around double bonds with STEREOANY stereo
                Chem.rdmolops.AddWavyBondsForStereoAny(rdm, addWhenImpossible=0, clearDoubleBondFlags=False) 
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE) 
                # display_wavy_bond = True
                break
    
    drawer.SetDrawOptions(options)

    drawer.DrawMolecule(mol)#,forceCoords=True)#as indigo-->rdkit has indigo drawing coordiantes,do not get new
    #check coordinates is overlap or not
    x = []
    y = []
    coords, symbols = [], []
    charges=[]
    index_map = {}
    graph={}
    for i in range(mol.GetNumAtoms()):
        xx,yy = (drawer.GetDrawCoords(i)[0]-min_y)*scale, (drawer.GetDrawCoords(i)[1]-min_x)*scale
        #note replace x,y position in rdkit
        charge = mol.GetAtomWithIdx(i).GetFormalCharge()
        symbol= mol.GetAtomWithIdx(i).GetSymbol()
        x.append(xx)
        y.append(yy)
        coords.append((xx,yy))
        charges.append(charge)
        symbols.append(symbol)
        index_map[mol.GetAtomWithIdx(i).GetIdx()]=i
    # number_querry=100
    if len(symbols)>number_querry:#number querry in DETR
        # print(f' model decoder limitation rdkit_image filter atoms numbers {len(symbols)} > {number_querry}')
        img = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
        graph={}
        return img, graph

    coords=[(round(xx),round(yy)) for xx,yy in coords]
    kdtree = KDTree(coords)
    i_xy=dict()
    adj_ij=set()
    radius=3
    for i,center in enumerate(coords):    
        xx,yy=center        
        distance, index = kdtree.query(center, k=2)
        i_xy[i]=[xx,yy]
        if distance[1] <= radius*2+2 :#reduce the batch_size errors
            # print(f'rdkit_image failed as coordinates with overlapping {distance[1]}from {symbols[index[0]]}--{symbols[index[1]]}')
            img = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
            graph={}
            return img, graph#TODO below code can be used to update coordinates in the feature, but RDkit drawing coords as blackbox now
    #         if i>index[1]:
    #             ij_=(i,index[1])
    #         else:
    #             ij_=(index[1],i)
    #         adj_ij.add(ij_)

    # if len(adj_ij) >0:
    #     diffcult_=False
    #     for jx,ix in adj_ij:
    #         x_,y_=i_xy[ix]
    #         x_u,y_u=i_xy[jx]
    #         exclu_=[(v[1],v[0]) for k,v in i_xy.items() if k!=jx]
    #         yy,xx=KDupdate_coordinates(y_u,x_u,exclu_,radius,h,w)
    #         i_xy[jx]=[xx,yy]#update coords
    #         if (yy,xx)==(y_u,x_u):
    #             diffcult_=True
    #             break
    #     if diffcult_:#TODO may chagne SVG value in the NEXT version
    #         print(f'rdkit_image::diffcult_ to update coordinates with overlapping')
    #         img = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
    #         graph={}
    #         return img,graph

    drawer.FinishDrawing()
    svg_str = drawer.GetDrawingText()
    # svg_str=rdkit.Chem.Draw.rdMolDraw2D.MolToACS1996SVG(rdm)
    svg = et.fromstring(svg_str.encode('iso-8859-1'))
    if np.random.rand()<0.25:
        atom_elems = svg.xpath(r'//svg:text', namespaces={'svg': 'http://www.w3.org/2000/svg'})
        for elem in atom_elems:
            style = elem.attrib['style']
            css = cssutils.parseStyle(style)
            css.setProperty('font-weight', 'bold')
            css_str = css.cssText.replace('\n', ' ')
            elem.attrib['style'] = css_str
    svg_str = et.tostring(svg)

    png = cairosvg.svg2png(bytestring=svg_str)
    # img = np.array(Image.open(BytesIO(png)), dtype=np.float32)#rdkit with black background ??now
    img = np.array(Image.open(BytesIO(png)), dtype=np.uint8)#NOTE if use float, TotensorV2 in pyotrch will bg to be black,np.uint8 also used in indigo image to graph
    # Invert the background color from black to white
    # img = 255 - image_array
    # Convert the inverted image array back to PIL image
    # inverted_image = Image.fromarray(img)
    # print(img.shape)
    # bonds_comp = []
    # for i in range(mol.GetNumBonds()):
    #     bond = mol.GetBondWithIdx(i)
    #     bond_type = rdkitbond_type_dict[bond.GetBondType()]
    #     begin_atom_idx = bond.GetBeginAtomIdx()
    #     end_atom_idx = bond.GetEndAtomIdx()
        # if (end_atom_idx,begin_atom_idx) in bond_stereo_dict.keys():
        #     begin_atom_idx, end_atom_idx  = end_atom_idx,  begin_atom_idx
        # bond_stereo_type = bond_stereo_dict[(begin_atom_idx, end_atom_idx)]
        # y1, x1 = (drawer.GetDrawCoords(begin_atom_idx)[0]-min_y)*scale, (drawer.GetDrawCoords(begin_atom_idx)[1]-min_x)*scale
        # y2, x2 = (drawer.GetDrawCoords(end_atom_idx)[0]-min_y)*scale, (drawer.GetDrawCoords(end_atom_idx)[1]-min_x)*scale
        # yy, xx = (x1 + x2) / 2, (y1 + y2) / 2# rdkit need cross x,y 
        # assert yy<=512 and xx<=512,f'why not ????!!'
        # bonds_comp.append((round(xx),round(yy),bond_type,
        # begin_atom_idx,end_atom_idx
        # )) 
    #TODO add bonds
    graph = {
        # 'bonds':bonds_comp,
        'coords': coords,
        'id2sid':index_map,#NOTE the atom index fixed as read from mol file, same from indigo index
        'symbols': symbols,
        'charges':charges,
        'mol':mol,
                }
    #plot used
    # ax_=[x for x,y in coords ]
    # ay_=[y for x,y in coords ]
    # bx_=[b[0] for b in bonds_comp ]
    # by_=[b[1] for b in bonds_comp ]
    # plt.imshow(img)
    # plt.plot(ax_,ay_,'r.')
    # plt.plot(bx_,by_,'g.')
    return img,  graph  