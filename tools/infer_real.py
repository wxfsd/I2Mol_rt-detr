"""by bowen
"""
import os 
import sys

import torch 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import pandas as pd

class CustomImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, debug=False):
        if 'real' in csv_path:
            self.data = pd.read_csv(csv_path, header=0)#change here
            self.real_ds=True 
        else:
            self.data = pd.read_csv(csv_path, header=None)#change here 
            self.real_ds=False 

        self.image_dir = image_dir
        self.transform = transform if transform else self.default_transforms()
        self.debug = debug
    
    def default_transforms(self):
        return T.Compose([
            T.Resize((640, 640)),  # 调整大小
            # T.ToImageTensor(),  # 转换为 PyTorch Tensor
            T.ToTensor(),
            lambda x: x.to(torch.float32),  # 手动转换数据类型# T.ConvertDtype(dtype=torch.float32),  # 转换数据类型
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # print('idx',idx)
        if self.real_ds:
            png_filename = self.data.file_path[idx].split('/')[-1]
            smiles = self.data.SMILES[idx]
            # if da=='hand'and png_filename not in png_files:
            #     continue
            img_path = os.path.join(self.image_dir, png_filename)
        else:
            row = self.data.iloc[idx]
            png_filename = os.path.basename(row[0])  # 获取图像文件名
            smiles = row[1]  # SMILES 结构信息
            img_path = os.path.join(self.image_dir, png_filename)
        
        # 加载图像
        image = Image.open(img_path)
        w, h = image.size
        
        # 处理灰度或其他模式
        if image.mode == "L":
            if self.debug: print("检测到灰度图像 (1 通道)，转换为 RGB...")
            image = image.convert("RGB")
        elif image.mode != "RGB":
            if self.debug: print(f"检测到 {image.mode} 模式，转换为 RGB...")
            image = image.convert("RGB")
        
        # 应用转换
        image_tensor = self.transform(image)
        
        target = {
            "orig_size": torch.tensor([w, h], dtype=torch.int32),
            "SMILES": smiles,
            "img_path": img_path,
            "image_id": torch.tensor(idx, dtype=torch.int32)
        }
        
        return image_tensor, target



def main(args, ) -> None:
    '''main
    '''

    # # 迭代 DataLoader
    # for i_, batch in enumerate(dataloader):
    #     images, targets=batch
    #     print("Batch images shape:", images.shape)  # (batch_size, C, H, W)
    #     print(f"{i_} Batch targets:", targets)
    #     break
    os.environ["CUDA_VISIBLE_DEVICES"]=f'{args.gpuid}'
    gpu_id=int(args.gpuid)
    dist.init_distributed()
    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )
    print(cfg.device,'before')
    cfg.device=torch.device('cuda', gpu_id)
    print(cfg.device,'after!!!')

        # 示例使用
    if args.infer:
        cfg.infer=args.infer
        cfg.csv_path =args.csv_path
        cfg.image_dir =args.image_dir
        dataset = CustomImageDataset(cfg.csv_path, cfg.image_dir, debug=False)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4,drop_last=False)
        cfg.infer_dataloader=dataloader #TODO here
        cfg.visual_check=args.visual_check
        cfg.other2ppsocr=args.other2ppsocr
        cfg.getacc=args.getacc

    cfg.outcsv_filename=args.outcsv_filename#NOTE passing the outcsv name here
    # print(cfg,cfg.__dir__(),"zzzzzzzzzzzzzzzzzz")
    # cfg.device=torch.device('cuda', args.gpu_device) if torch.cuda.is_available() else torch.device('cpu') 
    solver = TASKS[cfg.yaml_cfg['task']](cfg)#will get the postprocessor here from TASKS to solver
    
    if args.test_only:
        if args.infer:
            solver.infer()
        else:   
            solver.val()
    else:
        solver.fit()


if __name__ == '__main__':
    # datanme=f'USPTO'
    # datanme=f'JPO'
    home="/home/jovyan/rt-detr"
    pp="tools/output/rtdetr_r50vd_6x_coco_real_resample_charge_large/best_checkpoint.pth"
    cc="tools/output/rtdetr_r50vd_6x_coco_real_resample_adapter_both/checkpoint0068.pth"
    # tt="./output/rtdetr_r50vd_6x_coco_real_resample_charge_large_adpter2/best_checkpoint.pth"
    diffS='./output/rtdetr_r50vd_6x_coco_real_resample_charge_large_adpterWithoutJPO_diffSize/checkpoint0071.pth'
    obst='/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output/blured_merged_diff300start12_hand_addedObstac/best_checkpoint.pth'#23?
    tt="/home/jovyan/rt-detr/output/rtdetr_r50vd_6x_coco_real_resample_charge_large_adpter2/best_checkpoint.pth"
    bmd= '/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output/merged9/best_checkpoint.pth'#30
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default=f'{home}/rt-detr/configs/rtdetr/rtdetr_r50vd_6x_coco.yml')
    # parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/tools/output/rtdetr_r50vd_6x_coco_real_resample_charge_large/checkpoint0032.pth')
    # parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/tools/output/rtdetr_r50vd_6x_coco_real_resample/checkpoint0052.pth')
    # parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{pp}')
    # parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{cc}')
    # parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{diffS}')
    # parser.add_argument('--resume', '-r', type=str, default=f'{tt}')
    parser.add_argument('--resume', '-r', type=str, default=f'{bmd}')
    # parser.add_argument('--resume', '-r', type=str, default=f'{obst}')
    parser.add_argument('--tuning', '-t', type=str,)# default='/home/jovyan/model_checkpoint/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth'
    parser.add_argument('--test-only',default=True,)
    parser.add_argument('--infer', default=True,)
    
    parser.add_argument('--amp', default=False,)
    parser.add_argument('--dataname', '-dn', type=str, default=None)
    parser.add_argument('--gpuid',  type=str, default='1')

    args = parser.parse_args()
    if args.dataname:
        datanme= args.dataname
    else:
        # datanme='JPO'
        datanme='acs'
    # args.annot_file=f'/home/jovyan/rt-detr/data/real_processed/CLEF_with_charge/annotations/val.json'#NOTE same as the rtdetr_r50vd_6x_coco.yml dataset.yml
    # args.annot_file=f'/home/jovyan/volume/samba_share/from_docker/data/{datanme}_with_charge/annotations/test.json'#NOTE same as the rtdetr_r50vd_6x_coco.yml dataset.yml
    # args.outcsv_filename=f'/home/jovyan/rt-detr/rt-detr/output_2025Real/output_charge_{datanme}_tt.csv'
    if args.resume==bmd:
        args.outcsv_filename=f'/home/jovyan/rt-detr/rt-detr/output_2025Real/output_charge_{datanme}_bmd.csv'
    elif  args.resume==tt:
        args.outcsv_filename=f'/home/jovyan/rt-detr/rt-detr/output_2025/output_charge_{datanme}_tt.csv'

    if args.infer:
        args.csv_path =f"/home/jovyan/volume/samba_share/from_docker/data/work_space/ori/real/{datanme}.csv" #279
        args.image_dir  = f"/home/jovyan/volume/samba_share/from_docker/data/work_space/ori/real/{datanme}" #*smi 279、        args.csv_path =f"/home/jovyan/volume/samba_share/from_docker/data/csv_deal/{datanme}.csv" #279
        args.visual_check=True
        args.other2ppsocr=False
        args.getacc=True

    # datanme='Decmier_handraw'
    # args.outcsv_filename=f'/home/jovyan/rt-detr/rt-detr/output/Decimer_charge_{datanme}.csv'
    main(args)
"""TODO fix below case molscrib process ori 
get rdkit mol None
[EDG]C1=CC2=C(C=C1)C3(CCC3)NCC2
*.*c1ccc2c(c1)CC*C21CCC1.Br
"""