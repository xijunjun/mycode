# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline
from pathlib import Path
import sys,os
import argparse

sys.path.append(str(Path(__file__).parent.parent))


import realesrgan.archs
import realesrgan.data
import realesrgan.models



if __name__ == '__main__':
    # print(sys.argv)

    parser = argparse.ArgumentParser()
    # parser.add_argument('-exp_base_root', type=str, required=True, help='Path to project.')
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()

    train_result_root='/disks/disk1/Workspace/mycode/exproot'
    expname=args.opt.split('/')[-1].split('.')[0]
    exp_base_root=os.path.join(train_result_root,expname)


    # exp_base_root=args.exp_base_root
    # print(exp_base_root)
    # exit(0)
    root_path=exp_base_root
    # root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
