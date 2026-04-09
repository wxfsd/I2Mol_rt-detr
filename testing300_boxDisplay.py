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
os.environ["CUDA_VISIBLE_DEVICES"]='1'
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
#drawing box
from PIL import Image, ImageDraw, ImageFont

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
# idx_to_labels = {0:'other',1:'C',2:'O',3:'N',4:'Cl',5:'Br',6:'S',7:'F',8:'B',
#             9:'I',10:'P',11:'*',12:'Si',13:'NONE',14:'BEGINWEDGE',15:'BEGINDASH',
#             16:'=',17:'#',18:'-4',19:'-2',20:'-1',21:'1',22:'+2',} #NONE is single ?
idx_to_labels = {0:'other',1:'C',2:'O',3:'N',4:'Cl',5:'Br',6:'S',7:'F',8:'B',
            9:'I',10:'P',11:'*',12:'Si',13:'NONE',14:'BEGINWEDGE',15:'BEGINDASH',
            16:'=',17:'#',18:'-4',19:'-2',20:'-1',21:'1',22:'2',} #NONE is single ?

home="/home/jovyan/rt-detr"
pt_outhome='/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output'
pp="tools/output/rtdetr_r50vd_6x_coco_real_resample_charge_large/best_checkpoint.pth"
cc="tools/output/rtdetr_r50vd_6x_coco_real_resample_adapter_both/checkpoint0068.pth"
tt="./output/rtdetr_r50vd_6x_coco_real_resample_charge_large_adpter2/best_checkpoint.pth"
diffS='./output/rtdetr_r50vd_6x_coco_real_resample_charge_large_adpterWithoutJPO_diffSize/checkpoint0071.pth'
# tr1='blured_merged_diff300start11'
# tr1='blured_merged_diff300start12'
# tr1='blured_merged_diff300start12_hand'
tr1='blured_merged_diff300start12_hand_addedObstac'
tr1='merged9'
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default=f'{home}/rt-detr/configs/rtdetr/rtdetr_r50vd_6x_coco.yml')
# parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/tools/output/rtdetr_r50vd_6x_coco_real_resample_charge_large/checkpoint0032.pth')
# parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/tools/output/rtdetr_r50vd_6x_coco_real_resample/checkpoint0052.pth')
# parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{pp}')
# parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{cc}')
parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{tt}')
# parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{diffS}')

# parser.add_argument('--resume', '-r', type=str, default=f'{pt_outhome}/{tr1}/best_checkpoint.pth')


parser.add_argument('--tuning', '-t', type=str,)# default='/home/jovyan/model_checkpoint/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth')
parser.add_argument('--test-only',default=True,)
parser.add_argument('--amp', default=False,)

args, unknown = parser.parse_known_args()#in jupyter
cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )
args.gpu_device=0
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
postprocessor2=RTDETRPostProcessor(num_classes=23, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False)
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

def I2M_boxed(image_path,debug=False):
    # Example usage: #change thie image
    tensor,w,h = image_to_tensor(image_path)
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



debug=False
# das=[
# # "CLEF",
# # "acs",
# # "JPO",
# # "UOB",
# # "USPTO",
# # "staker",
# "hand",
# ]
# ffou=open(f'DecimerV2.log' , 'w')
opts = Draw.MolDrawOptions()
opts.addAtomIndices = False
opts.addStereoAnnotation = False
###################################################################################
dir_name = "handPredictResultBoxed"
os.makedirs(dir_name, exist_ok=True)
csv_path="/home/jovyan/rt-detr/rt-detr/hand_output.csv"
###########
dir_name = "handPredictResultBoxed_diff"
os.makedirs(dir_name, exist_ok=True)
csv_path="/home/jovyan/rt-detr/rt-detr/hand_diff_output.csv"



df = pd.read_csv(csv_path)
print(df.head())
for i, row in df.iterrows():
    # if i>2:break#testing
    png_filename = df.img_path[i]
    smiles = df.smiles[i]
    # img_path = os.path.join(acs_dir, png_filename)
    img_path = png_filename
    png_ = os.path.basename(img_path)
    # if debug:
    print(i,img_path)
    try:
        # image_path = "../Sample_Images/P_16381197.png"
        # smiles_pred = predict_SMILES(img_path)
        smiles_pred, mol_pred,output=I2M_boxed(img_path)
        #visual checking
        predict_boxes = output['bbox']
        predict_classes = output['pred_classes']
        predict_scores = output['scores']

        img_ori = Image.open(img_path).convert('RGB')
        w_ori, h_ori = img_ori.size  # 获取原始图像的尺寸
        # print(w_ori, h_ori, "orignianl vs 1000,1000")
        # 计算缩放比例
        scale_x = 1000 / w_ori
        scale_y = 1000 / h_ori
        img_ori_1k = img_ori.resize((1000,1000))
        img = Image.open(img_path).convert('RGB')
        img = img.resize((1000,1000))
        newbox = predict_boxes * [scale_x, scale_y, scale_x, scale_y]
        boxed_img = draw_objs(img,
                                newbox,
                                predict_classes,
                                predict_scores,
                                category_index=idx_to_labels,
                                box_thresh=0.5,
                                line_thickness=3,
                                font='arial.ttf',
                                font_size=10)
        img_ori = Image.open(img_path).convert('RGB')
        img_ori_1k = img_ori.resize((1000,1000))
        # if other2ppsocr:
        #     img_rebuit = Draw.MolToImage(final_mol, options=opts,size=(1000, 1000))
        # else:
        img_rebuit = Draw.MolToImage(mol_pred, options=opts,size=(1000, 1000))
        combined_img = Image.new('RGB', (img_ori_1k.width + boxed_img.width + img_rebuit.width, img_ori_1k.height))
        combined_img.paste(img_ori_1k, (0, 0))
        combined_img.paste(boxed_img, (img_ori_1k.width, 0))
        combined_img.paste(img_rebuit, (img_ori_1k.width + boxed_img.width, 0))
        # 2. 获取绘图对象
        draw = ImageDraw.Draw(combined_img)
        # 3. 准备要写入的文本，这里示例写入 combined_img 的实际宽高
        text = f"W::{w_ori}, H::{h_ori} resized to 1000, 1000"  # 或者直接写成 "W,H"
        try:
            font = ImageFont.truetype("arial.ttf", 10)  # 字体文件，字号24
        except IOError:
            # 如果没有对应字体文件，会抛出 IOError，改用默认字体
            font = ImageFont.load_default()
        # 5. 写文字到图像上
        # 位置 (10, 10) 只是示例，你可以根据需要自行调整
        text_position = (450, 10)
        text_color = (255, 0, 255)  #
        draw.text(text_position, text, fill=text_color, font=font)
        combined_img.save(f"{dir_name}/{png_}")
        # combined_img

        # result = evaluate(img_path)#put in for loop
        # smiles_pred=converter.decode(''.join(result).replace("<start>","").replace("<end>",""))
    except Exception as e:
        print(e)
        smiles_pred=None

"""
@gpu2
source /cadd_data/samba_share/from_docker/cu12/etc/profile.d/conda.sh
conda activate 
cd /cadd_data/samba_share/bowen/workspace/dv2_test

nohup python /cadd_data/samba_share/bowen/workspace/decimerV2_.py > allv2.out 2>&1 & 
python /home/jovyan/rt-detr/LG_SMILES_1st-main/validation.py

"""
