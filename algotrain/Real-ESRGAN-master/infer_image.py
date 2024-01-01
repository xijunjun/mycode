#coding:utf-8
import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2,random
import os
import numpy as np
import shutil
import platform

import argparse
import cv2
import torch



def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

def limit_img_auto(imgin):
    img = np.array(imgin)
    sw = 1920 * 1.0
    sh = 1080 * 1.0
    h, w, c = img.shape
    swhratio = 1.0 * sw / sh
    whratio = 1.0 * w / h
    # 横向的长图
    if whratio > swhratio:
        tw = int(sw)
        if tw < w:
            th = int(h * (tw / w))
            img = cv2.resize(img, (tw, th))
    else:
        th = int(sh)
        if th < h:
            tw = int(w * (th / h))
            img = cv2.resize(img, (tw, th))
    return img

def get_imkey_ext(impath):
    imname=os.path.basename(impath)
    ext='.'+imname.split('.')[-1]
    imkey=imname[0:len(imname)-len(ext)]
    return imkey,ext


if __name__ == '__main__':

    # device='cuda'
    device='cpu'

    srcroot=r'/home/tao/mynas/Dataset/hairforsr/femalehd_crop2048_eyescrop_out'

    ims=get_ims(srcroot)


    checkpath=[
        # r'/disks/disk1/Workspace/mycode/algotrain/Real-ESRGAN-master/experiments/pretrained_models/RealESRGAN_x2plus.pth'
        r'/disks/disk1/Workspace/mycode/exproot/eye_sr_debug/experiments/train_RealESRGANx2plus_400k_B12G4/models/net_g_6000.pth'
    ][0]
    loadnet = torch.load(checkpath,map_location=torch.device(device))

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

    # prefer to use params_ema
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)
    model.eval()
    model = model.to(device)


    numim=len(ims)
    for i,im in enumerate(ims):

        imkey,ext=get_imkey_ext(im)


        imgori=cv2.imread(im)

        imgori=cv2.resize(imgori,(256,256))

        img=np.array(imgori)


        img = img.astype(np.float32)/255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        output_img=model(img)

        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))


        output_img_np=(output_img*255).astype(np.uint8)


        imgori=cv2.resize(imgori,(256*2,256*2))
        cv2.imshow('imgori',limit_img_auto(imgori))
        cv2.imshow('output_img_np',limit_img_auto(output_img_np))



        key=cv2.waitKey(0)
        if key==27:
            exit(0)

