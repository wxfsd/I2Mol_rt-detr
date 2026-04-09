"""by lyuwenyu
"""

import os 
import sys

import torch 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS

os.environ["CUDA_VISIBLE_DEVICES"]='1'


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )
    cfg.outcsv_filename=args.outcsv_filename#NOTE passing the outcsv name here
    # print(cfg,cfg.__dir__(),"zzzzzzzzzzzzzzzzzz")
    # cfg.device=torch.device('cuda', args.gpu_device) if torch.cuda.is_available() else torch.device('cpu') 
    solver = TASKS[cfg.yaml_cfg['task']](cfg)#will get the postprocessor here from TASKS to solver
    print(solver.cfg)
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':
    # datanme=f'USPTO'
    # datanme=f'JPO'
    home="/home/jovyan/rt-detr"
    pp="tools/output/rtdetr_r50vd_6x_coco_real_resample_charge_large/best_checkpoint.pth"
    cc="tools/output/rtdetr_r50vd_6x_coco_real_resample_adapter_both/checkpoint0068.pth"
    tt="./output/rtdetr_r50vd_6x_coco_real_resample_charge_large_adpter2/best_checkpoint.pth"
    diffS='./output/rtdetr_r50vd_6x_coco_real_resample_charge_large_adpterWithoutJPO_diffSize/checkpoint0071.pth'
    obst='/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output/blured_merged_diff300start12_hand_addedObstac/best_checkpoint.pth'#23?
    bmd= '/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output/merged9/best_checkpoint.pth'#30
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default=f'{home}/rt-detr/configs/rtdetr/rtdetr_r50vd_6x_coco.yml')
    # parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/tools/output/rtdetr_r50vd_6x_coco_real_resample_charge_large/checkpoint0032.pth')
    # parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/tools/output/rtdetr_r50vd_6x_coco_real_resample/checkpoint0052.pth')
    # parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{pp}')
    # parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{cc}')
    # parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{tt}')
    # parser.add_argument('--resume', '-r', type=str, default=f'{home}/rt-detr/{diffS}')
    parser.add_argument('--resume', '-r', type=str, default=f'{bmd}')


    # parser.add_argument('--resume', '-r', type=str, default=f'{obst}')

    parser.add_argument('--tuning', '-t', type=str,)# default='/home/jovyan/model_checkpoint/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth'
    parser.add_argument('--test-only',default=True,)
    parser.add_argument('--amp', default=False,)

    datanme='acs'
    datanme='JPO'
    args = parser.parse_args()
    # args.annot_file=f'/home/jovyan/rt-detr/data/real_processed/CLEF_with_charge/annotations/val.json'#NOTE same as the rtdetr_r50vd_6x_coco.yml dataset.yml
    # args.annot_file=f'/home/jovyan/volume/samba_share/from_docker/data/{datanme}_with_charge/annotations/test.json'#NOTE same as the rtdetr_r50vd_6x_coco.yml dataset.yml
    args.outcsv_filename=f'/home/jovyan/rt-detr/rt-detr/output_2025/output_charge_{datanme}_tt.csv'
    # datanme='Decmier_handraw'
    # args.outcsv_filename=f'/home/jovyan/rt-detr/rt-detr/output/Decimer_charge_{datanme}.csv'

    main(args)
#from rdkit mol.debug()
# Atoms:
#         0 6 C chg: 0  deg: 1 exp: 2 imp: 2 hyb: SP2
#         1 6 C chg: 0  deg: 2 exp: 4 imp: 0 hyb: SP
#         2 6 C chg: 0  deg: 2 exp: 4 imp: 0 hyb: SP
#         3 6 C chg: 0  deg: 1 exp: 2 imp: 2 hyb: SP2
#         4 6 C chg: 0  deg: 1 exp: 2 imp: 2 hyb: SP2
#         5 6 C chg: 0  deg: 1 exp: 2 imp: 2 hyb: SP2
#         6 7 N chg: 0  deg: 1 exp: 2 imp: 1 hyb: SP2
#         7 53 I chg: 0  deg: 2 exp: 4 imp: 1 hyb: SP2
#         8 7 N chg: 0  deg: 1 exp: 2 imp: 1 hyb: SP2
#         9 7 N chg: 0  deg: 1 exp: 2 imp: 1 hyb: SP2
#         10 16 S chg: 0  deg: 1 exp: 2 imp: 0 hyb: SP2
#         11 6 C chg: 0  deg: 2 exp: 4 imp: 0 hyb: SP
# Bonds:
#         0 1->4 order: 2 conj?: 1
#         1 0->6 order: 2
#         2 2->7 order: 2 conj?: 1
#         3 5->3 order: 2
#         4 10->2 order: 2 conj?: 1
#         5 7->11 order: 2 conj?: 1
#         6 8->9 order: 2
#         7 11->1 order: 2 conj?: 1
#ls /home/jovyan/rt-detr/data/real/CLEF/US20050009817A1_p0023_x0563_y1914_c00076.png
#after get .csv run:: /home/jovyan/rt-detr/LG_SMILES_1st-main/validation.py 
#match:832,unmatch:25  <--/home/jovyan/rt-detr/rt-detr/output/output_charge_CLEF.csv
#  