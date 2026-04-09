#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys,copy
import torchvision
import argparse
import torch
import tqdm


# conda install -n ocr   conda-forge::libgl   for import cv2 succus


# In[2]:


sys.path.append("/home/jovyan/rt-detr/rt-detr")
from src.solver.utils import output_to_smiles

os.chdir('/home/jovyan/rt-detr/rt-detr')
os.getcwd()


# In[3]:


import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
from src.data import get_coco_api_from_dataset



# In[4]:


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


# In[5]:


from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F



def remove_backslash_and_slash(input_string):
    if "\\" in input_string:
        input_string = input_string.replace("\\", "")
    if "/" in input_string:
        input_string = input_string.replace("/", "")

    return input_string



def image_to_tensor(image_path):
    # Open the image using PIL
    image = Image.open(image_path)
    w, h = image.size
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
idx_to_labels = {0:'other',1:'C',2:'O',3:'N',4:'Cl',5:'Br',6:'S',7:'F',8:'B',
            9:'I',10:'P',11:'*',12:'Si',13:'NONE',14:'BEGINWEDGE',15:'BEGINDASH',
            16:'=',17:'#',18:'-4',19:'-2',20:'-1',21:'1',22:'+2',} #NONE is single ?


# In[6]:


from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True,use_gpu =False,
    rec_algorithm='SVTR_LCNet', rec_model_dir='/home/jovyan/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer',
    lang="en")  # need to run only once to download and load model into memory

#* recong used


# In[7]:


home="/home/jovyan/rt-detr"
pp="tools/output/rtdetr_r50vd_6x_coco_real_resample_charge_large/best_checkpoint.pth"
cc="tools/output/rtdetr_r50vd_6x_coco_real_resample_adapter_both/checkpoint0068.pth"
tt="./output/rtdetr_r50vd_6x_coco_real_resample_charge_large_adpter2/best_checkpoint.pth"
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default=f'{home}/rt-detr/configs/rtdetr/rtdetr_r50vd_6x_coco.yml')
# parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/tools/output/rtdetr_r50vd_6x_coco_real_resample_charge_large/checkpoint0032.pth')
# parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/tools/output/rtdetr_r50vd_6x_coco_real_resample/checkpoint0052.pth')
# parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{pp}')
# parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{cc}')
parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{tt}')

parser.add_argument('--tuning', '-t', type=str,)# default='/home/jovyan/model_checkpoint/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth')
parser.add_argument('--test-only',default=True,)
parser.add_argument('--amp', default=False,)

args, unknown = parser.parse_known_args()#in jupyter


# In[8]:


cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )
args.gpu_device=0
cfg.device=torch.device('cuda', args.gpu_device) if torch.cuda.is_available() else torch.device('cpu') 


# In[9]:


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


# In[10]:


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


# In[11]:


print("when training use this as valdation dataset::",
    cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'],"\n",
cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'])

# print(type(cfg.val_dataloader))#note this val_dataloader is the training config used, not used for testing 
# print(cfg.val_dataloader.dataset)


# In[12]:


opts = Draw.MolDrawOptions()
opts.addAtomIndices = False
opts.addStereoAnnotation = False

chemical_elements = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc",
    "Lv", "Ts", "Og"
]
# In[13]:


"""
python /home/jovyan/rt-detr/LG_SMILES_1st-main/preprocess/preprocess_data_charge.py 
python /home/jovyan/rt-detr/LG_SMILES_1st-main/preprocess/coco2coco.py # as above step have been coco styles
--->val_dataloader: from  configs/dataset/coco_detection.yml as the above prepared:
img_folder: /home/jovyan/rt-detr/data/real_processed/staker_with_charge/images/val
ann_file: /home/jovyan/rt-detr/data/real_processed/staker_with_charge/annotations/val.json

python ~/rt-detr/rt-detr/tools/test.py 
python /home/jovyan/rt-detr/LG_SMILES_1st-main/validation.py
"""

#CELF match:854,unmatch:3
#CELF+OCRstring match:856,unmatch:1  overlapping atoms

#staker match:44467,unmatch:1010, erros:16  (不考虑/\异构)
#staker+OCR

#acs 
#acs+OCR

#staker 
#staker+OCR

#model 预测miss  rdkti  画图atom overalaping,   构建有坐标2D 图 可以easy checking


# In[14]:


# gg=['US20070249620A1_p0006_x1375_y2591_c00009.png', 'US20050113580A1_p0038_x1307_y1020_c00053.png', 'US20030130506A1_p0008_x1381_y1349_c00031.png']
# imgdir="/home/jovyan/rt-detr/data/real_processed/CLEF_with_charge/images/test"
# image_path = f'{imgdir}/{gg[2]}'#

# dataname='USPTO'
dataname='JPO'
staker_cdf=pd.read_csv(f'{dataname}_with_charge.csv')
staker_cdf_imgs=staker_cdf['img'].to_list()
image_path=staker_cdf_imgs[1]



#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
expan=0#NOTE   this control how much the part of bond in crop_Img


correcte=[]
uncorrects=[]

# add for loop for csv file
# Example usage: #change thie image
debug=True
for j_, osi_row in staker_cdf.iterrows():
    # if j_>=1:break
    # image_path='/home/jovyan/rt-detr/data/real_processed/acs_with_charge/images/test/jo001614p-Scheme-c2-2.png'
    # input_s='[H]C1=CC=C2C=CC=CC2=C1C3=C(P(=O)([Ar])[Ar])C=CC4=CC=CC=C34'

    image_path=osi_row['img']
    input_s=osi_row['SMILES_ori']

    tensor,w,h = image_to_tensor(image_path)
    tensor=tensor.unsqueeze(0)
    # print(tensor.size())  # Output tensor shape (C x H x W)
    # _model.training=False
    _model.eval()#have to uset this
    with torch.no_grad():
        # print(_model.training)
        outputs = _model(tensor)

    ori_size=torch.Tensor([w,h]).long().unsqueeze(0)
    result_ = postprocessor(outputs, ori_size)
    # result_ = postprocessor(out_, torch.Tensor([w,h]))
    score_=result_[0]['scores']
    boxe_=result_[0]['boxes']
    label_=result_[0]['labels']
    selected_indices =score_ > 0.5
    output={
        'labels': label_[selected_indices],
        'boxes': boxe_[selected_indices],
        'scores': score_[selected_indices]
    }

    filtered_output_dict={image_path: output
    }


    x_center = (output["boxes"][:, 0] + output["boxes"][:, 2]) / 2
    y_center = (output["boxes"][:, 1] + output["boxes"][:, 3]) / 2
    center_coords = torch.stack((x_center, y_center), dim=1)
    output = {'bbox':         output["boxes"].to("cpu").numpy(),
                'bbox_centers': center_coords.to("cpu").numpy(),
                'scores':       output["scores"].to("cpu").numpy(),
                'pred_classes': output["labels"].to("cpu").numpy()}


    if debug:
        import importlib
        importlib.reload(draw_box_utils)
        importlib.reload(src.solver.utils)
        from src.solver.utils import bbox_to_graph_with_charge,mol_from_graph_with_chiral,assemble_atoms_with_charges
        from draw_box_utils import draw_objs,STANDARD_COLORS,draw_text


    #visual checking
    predict_boxes = output['bbox']
    predict_classes = output['pred_classes']
    predict_scores = output['scores']

    img_ori = Image.open(image_path).convert('RGB')
    img_ori_1k = img_ori.resize((1000,1000))
    img = Image.open(image_path).convert('RGB')
    img = img.resize((1000,1000))

    boxed_img = draw_objs(img,
                            predict_boxes*10/3,
                            predict_classes,
                            predict_scores,
                            category_index=idx_to_labels,
                            box_thresh=0.5,
                            line_thickness=3,
                            font='arial.ttf',
                            font_size=10)
    try:
        atoms_df, bonds_list,charge_list =bbox_to_graph_with_charge(output, idx_to_labels=idx_to_labels,
                                                            bond_labels=bond_labels,  result=[])
        smiles,mol_rebuit=mol_from_graph_with_chiral(atoms_df, bonds_list,charge_list )#NOTE, get SDF from mol_rebuit


        need_cut=[]
        ppstr=[]
        ppstr_score=[]
        crops=[]
        index_token=dict()
        mol = rdkit.Chem.RWMol(mol_rebuit)
        other2pps=False

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
                    print(s_, "paddleOCR not recongized",score_)
                    s_='*'
                if s_=='+' or s_=='-':
                    s_="*"
                if len(s_)>1:
                    s_=re.sub(r'[^a-zA-Z0-9\*\-\+]', '', s_)#remove special chars
                    if re.match(r'^\d+$', s_):
                        # print(f'why only numbers ?  {s_}')
                        s_=f'{s_}*'#number+ *
                if s_=='L':s_='Li'
                match = re.match(r'^(\d+)?(.*)', s_)
                if match:
                    numeric_part, remaining_part = match.groups()
                    elemt_=re.sub(r'[^a-zA-Z]', '', remaining_part)#remove special chars
                    if elemt_ in chemical_elements:
                        if numeric_part:
                            s_=f'{numeric_part}{elemt_}'
                        else:
                            s_=f'{elemt_}'#only period  element 
                    else:#not recongized by rdkit strs combins
                        if numeric_part:
                            s_=f'{numeric_part}*'
                        else:
                            if s_[:-1] in chemical_elements :s_=s_[:-1]
                            elif s_[:-2] in chemical_elements :s_=s_[:-2]
                            elif s_[:-3] in chemical_elements :s_=s_[:-3]
                            else:
                                print(f'{s_} will be --> *')
                                s_='*'
                index_token[i_]=f'{s_}:{i_}'
                # print(f"idx:{i_}, atm:{row.atom}-->[{s_}:{i_}] with score:{score_}")
                mol.GetAtomWithIdx(i_).SetProp("atomLabel", f"{s_}")
                ppstr.append(s_)
                ppstr_score.append(score_)
        # print(f"nums: {len(index_token)} '*' need paddleOCR converting",index_token)
        final_mol = mol.GetMol()
    except Exception as e:
        # print(f"idx:{i_}, atm:{row.atom}-->[{s_}:{i_}] with score:{score_}")
        # print(atoms_df)
        print(f"not recongnized::{image_path}\n{input_s}")

        print(e,'\n')
        uncorrects.append(image_path)
        continue
    try:
        rdkit_input_s=Chem.MolToSmiles(Chem.MolFromSmiles(input_s),kekuleSmiles=True)
        if len(ppstr)==1:
            cur_smi=Chem.MolToSmiles(final_mol,kekuleSmiles=True)#as csv use kekuleSmiles style
            test_smiles=re.sub(r'\*', f'[{s_}]', cur_smi)
            # new_mol=Chem.MolFromSmiles(test_smiles)
            rdkit_test_smiles=Chem.MolToSmiles(Chem.MolFromSmiles(test_smiles),kekuleSmiles=True)
            rdkit_test_smiles=Chem.MolToSmiles(Chem.MolFromSmiles(rdkit_test_smiles))#,kekuleSmiles=True)
            if '.[' in rdkit_test_smiles:#O.[C]-->O.C
                sl=rdkit_test_smiles.split('.')
                new_sl=[]
                for sss in sl:
                    if sss[0]=='[' and sss[-1]==']' and len(sss) in range(3,6):
                        new_sl.append(sss[1:-1])
                    else:
                        new_sl.append(sss)
                rdkit_test_smiles='.'.join(new_sl) 
            rdkit_input_s=Chem.MolToSmiles(Chem.MolFromSmiles(rdkit_input_s))#,kekuleSmiles=True)NOTE kekuleSmiles may change atom order
            other2pps=True  
            
            test_mol=Chem.MolFromSmiles(rdkit_test_smiles)
            rdkit_test_smiles=Chem.MolToSmiles(test_mol)#,kekuleSmiles=True)

            original_mol=Chem.MolFromSmiles(rdkit_input_s)
            rdkit_input_s=Chem.MolToSmiles(original_mol)#,kekuleSmiles=True)NOTE kekuleSmiles may change atom order
            
            keku_smi_ori=Chem.MolToSmiles(original_mol,kekuleSmiles=True)
            keku_smi=Chem.MolToSmiles(test_mol,kekuleSmiles=True)
            if '*' not in keku_smi:
                keku_inch_ori=  Chem.MolToInchi(Chem.MolFromSmiles(keku_smi_ori))
                keku_inch_test=  Chem.MolToInchi(Chem.MolFromSmiles(keku_smi))
            else:
                keku_inch_ori=  1
                keku_inch_test=  2
            if rdkit_input_s == rdkit_test_smiles or keku_smi_ori == keku_smi or keku_inch_ori==keku_inch_test:
                correcte.append(image_path)
            else:
                print(f"final        -->{rdkit_test_smiles}")
                print(f"rdkit_input_s-->{rdkit_input_s}")
                uncorrects.append(image_path)

        elif len(ppstr)>1:
            index_mol=copy.deepcopy(final_mol)
            show_atom_number(index_mol, 'molAtomMapNumber')
            index_smi=Chem.MolToSmiles(index_mol,kekuleSmiles=True)
            ai_tokens=atomwise_tokenizer(index_smi)
            new_toks=[]
            for k,v in index_token.items():
                for i, atoken in enumerate(ai_tokens):
                    if f':{k}' in atoken:
                        # index_token[k]
                        ai_tokens[i]=re.sub(r'\[.*?\]', f'[{index_token[k]}]', ai_tokens[i])
                        print(f':{k},  {atoken} --> [{index_token[k]}] || {ai_tokens[i]}' )
                    # new_toks.append(atoken)
            new_smi=''.join(ai_tokens)
            new_mol=Chem.MolFromSmiles(new_smi)
            m_noid=remove_atom_number(new_mol)#TODO this may lead problem
            #add H style
            # m_noid.AddHs()
            s_noid=Chem.MolToSmiles(m_noid)
            if '.[' in s_noid:
                sl=s_noid.splite('.')
                new_sl=[]
                for sss in sl:
                    if sss[0]=='[' and sss[-1]==']' and len(sss) in range(3,6):
                        new_sl.append(sss[1:-1])
                    else:
                        new_sl.append(sss)
                s_noid='.'.join(new_sl)

            test_smiles = re.sub(r'\[(\d+)\*', '[*',s_noid)#remove_number_before_star
            test_smiles = remove_SP(test_smiles)
            other2pps=True
            rdkit_test_smiles=Chem.MolToSmiles(Chem.MolFromSmiles(test_smiles),kekuleSmiles=True)
            rdkit_test_smiles = remove_backslash_and_slash(rdkit_test_smiles)
            
            test_mol=Chem.MolFromSmiles(rdkit_test_smiles)
            rdkit_test_smiles=Chem.MolToSmiles(test_mol)#,kekuleSmiles=True)

            original_mol=Chem.MolFromSmiles(rdkit_input_s)
            rdkit_input_s=Chem.MolToSmiles(original_mol)#,kekuleSmiles=True)NOTE kekuleSmiles may change atom order
            
            keku_smi_ori=Chem.MolToSmiles(original_mol,kekuleSmiles=True)
            keku_smi=Chem.MolToSmiles(test_mol,kekuleSmiles=True)
            if '*' not in keku_smi:
                keku_inch_ori=  Chem.MolToInchi(Chem.MolFromSmiles(keku_smi_ori))
                keku_inch_test=  Chem.MolToInchi(Chem.MolFromSmiles(keku_smi))
            else:
                keku_inch_ori=  1
                keku_inch_test=  2
            rd_smi=Chem.MolToSmiles(test_mol)
            rd_smi_ori=Chem.MolToSmiles(original_mol)
            if rdkit_input_s == rdkit_test_smiles or keku_smi_ori == keku_smi or keku_inch_ori==keku_inch_test:
                correcte.append(image_path)
            else:
                print(f"before replace-->{index_smi}")#TODO NOTE rdkit None* try re way
                print(f"after  replace-->{new_smi}")
                print(f"final        -->{rdkit_test_smiles}")
                print(f"rdkit_input_s-->{rdkit_input_s}")
                # print(f"input_s     -->{input_s}")
                print(rdkit_test_smiles==rdkit_input_s)
                uncorrects.append(image_path)
                print(f"not right::{image_path}\n{input_s}")
        
        else:
            uncorrects.append(image_path)
            other2pps=False
            print(f"not right:@{len(ppstr)}@:{image_path}\n{input_s}\n{smiles}")

    except Exception as e:
        other2pps=False
        uncorrects.append(image_path)
        print(f"not recongnized::{image_path}\n{input_s}")
        print(e,'\n')



print(f'corrected ::{len(correcte)},uncorrected::{len(uncorrects)}  total::{len(staker_cdf_imgs)} with *box_expand::{expan}')

#TODO get the * with coords, check String OCR, check rebuild process charge and bond missing!!


# In[29]:

visual_check=False
# other2pps=True
if visual_check:
    img_ori = Image.open(image_path).convert('RGB')
    img_ori_1k = img_ori.resize((1000,1000))
    if other2pps:
        img_rebuit = Draw.MolToImage(final_mol, options=opts,size=(1000, 1000))
    else:
        img_rebuit = Draw.MolToImage(mol_rebuit, options=opts,size=(1000, 1000))

    combined_img = Image.new('RGB', (img_ori_1k.width + boxed_img.width + img_rebuit.width, img_ori_1k.height))
    combined_img.paste(img_ori_1k, (0, 0))
    combined_img.paste(boxed_img, (img_ori_1k.width, 0))
    combined_img.paste(img_rebuit, (img_ori_1k.width + boxed_img.width, 0))
