#coding:utf-8

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

    srcroot=r'/home/tao/mynas/Dataset/hairforsr/femalehd_crop2048'

    faceland_root=srcroot+'_faceland'

    ims=get_ims(srcroot)



    numim=len(ims)
    for i,im in enumerate(ims):

        imkey,ext=get_imkey_ext(im)

        npypath_face=os.path.join(faceland_root,imkey+'.npy')

        landmarks=np.load(npypath_face)


        imgori=cv2.imread(im,cv2.IMREAD_UNCHANGED)
        h,w,c=imgori.shape
        img=np.zeros((h,w,3),imgori.dtype)
        img[:,:,:]=imgori[:,:,0:3]



        for pt in landmarks:
            cv2.circle(img, (pt[0], pt[1]), 10, (255, 0, 0), -1, -1)

        cv2.imshow('img',limit_img_auto(img))



        key=cv2.waitKey(0)
        if key==27:
            exit(0)

