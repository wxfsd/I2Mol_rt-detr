"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS

os.environ["CUDA_VISIBLE_DEVICES"]='2'

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
    cfg.output_dir=args.output_dir


    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='/home/jovyan/rt-detr/rt-detr/configs/rtdetr/rtdetr_r50vd_6x_coco.yml')
    # parser.add_argument('--resume', '-r', type=str, default='/home/jovyan/rt-detr/rt-detr/output/rtdetr_r50vd_6x_coco_real_resample_charge_large_adpterWithoutJPO/checkpoint0006.pth')
    parser.add_argument('--resume', '-r', type=str)#, default='/home/jovyan/rt-detr/rt-detr/output/rtdetr_r50vd_6x_coco_real_resample_charge_large_adpterWithoutJPO_diffSize/checkpoint0035.pth')
    # parser.add_argument('--tuning', '-t', type=str, default='/home/jovyan/rt-detr/rt-detr/output/rtdetr_r50vd_6x_coco_real_resample_charge_large_adpter2/best_checkpoint.pth')#300X300 trained
    # parser.add_argument('--tuning', '-t', type=str, default='/home/jovyan/rt-detr/rt-detr/output/rtdetr_r50vd_6x_coco_real_resample_charge_large_adpterWithoutJPO_diffSize/checkpoint0071.pth')#diffSeized image
    parser.add_argument('--tuning', '-t', type=str, default='/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output/blured_merged_diff300start12_hand_addedObstac/best_checkpoint.pth')
    # parser.add_argument('--tuning', '-t', type=str)#, default='/home/jovyan/rt-detr/model_checkpoint/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth')
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)

    args = parser.parse_args()
    datanme='chembl'
    # args.outcsv_filename=f'/home/jovyan/rt-detr/rt-detr/output/mergedCharged_withoutJPO_diffsize.csv'#trick when training with target dataset as val
    # args.outcsv_filename=f'/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output/blured_merged_diff300start12.csv'#trick when training with target dataset as val
    # args.output_dir="/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output/blured_merged_diff300start12"
    # args.outcsv_filename=f'/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output/blured_merged_diff300start12_hand.csv'#trick when training with target dataset as val
    # args.output_dir="/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output/blured_merged_diff300start12_hand"
    # args.outcsv_filename=f'/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output/blured_merged_diff300start12_hand_addedObstac.csv'#trick when training with target dataset as val
    # args.output_dir="/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output/blured_merged_diff300start12_hand_addedObstac"
    args.outcsv_filename=f'/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output/merged9.csv'#trick when training with target dataset as val
    args.output_dir="/home/jovyan/volume/samba_share/from_docker/ocr_data/rtdetr_output/merged9"
    
    
    
    main(args)

    """
    with charge adapter etc.  number of params: 52890557

    nohup python tools/train.py > blured_merged_diff300start11.log  2>&1 &
    nohup python tools/train.py > blured_merged_diff300start12.log  2>&1 &
    nohup python tools/train.py > blured_merged_diff300start12_hand.log  2>&1 &
    nohup python tools/train.py > blured_merged_diff300start12_hand_addedObstac.log  2>&1 &
    nohup python tools/train.py > merged9.log  2>&1 &
    
    """