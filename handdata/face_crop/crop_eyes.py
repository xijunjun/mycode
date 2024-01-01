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


def pts2rct(pts):
    pts = np.array(pts)
    tlx = min(pts[:, 0])
    tly = min(pts[:, 1])
    brx = max(pts[:, 0])
    bry = max(pts[:, 1])
    return [tlx, tly, brx, bry]

def get_eyeims(img,faceland,INSIZE):
    # landpts=[]
    leye=[]
    reye=[]
    numpts_eye=8
    for i in range(0,numpts_eye):
        leye.append(faceland[60+i])
        reye.append(faceland[68 + i])

    leyerct=pts2rct(leye)
    reyerct = pts2rct(reye)
    leye=np.array(leye)
    reye = np.array(reye)
    leyex=int(1.0*sum(leye[:,0])/numpts_eye)
    leyey=int(1.0*sum(leye[:,1])/numpts_eye)
    reyex=int(1.0*sum(reye[:,0])/numpts_eye)
    reyey=int(1.0*sum(reye[:,1])/numpts_eye)
    eyewidth=max(max(leyerct[2]-leyerct[0],leyerct[3]-leyerct[1]),max(reyerct[2]-reyerct[0],reyerct[3]-reyerct[1]))

    # tarw=random.randint(90,120)
    tarw=160

    cropw=int(1.0*eyewidth/tarw*INSIZE)//2
    lcroprct=[leyex-cropw,leyey-cropw,leyex+cropw,leyey+cropw]
    rcroprct = [reyex - cropw, reyey - cropw, reyex + cropw, reyey + cropw]
    # img=cv2.imread(items[0])
    # img = np.array(Image.open(items[0]).convert('RGB'))

    leyeim = img[lcroprct[1]:lcroprct[3], lcroprct[0]:lcroprct[2]]
    reyeim=img[rcroprct[1]:rcroprct[3],rcroprct[0]:rcroprct[2]]
    # print (line)
    leyeim=cv2.resize(leyeim,(INSIZE,INSIZE))
    reyeim = cv2.resize(reyeim, (INSIZE, INSIZE))
    if random.random()>0.5:
        return leyeim
    return reyeim


def get_eyeims_fix(img,faceland,INSIZE):
    # landpts=[]
    h,w,c=img.shape
    leye=[]
    reye=[]
    numpts_eye=8
    for i in range(0,numpts_eye):
        leye.append(faceland[60+i])
        reye.append(faceland[68 + i])

    leyerct=pts2rct(leye)
    reyerct = pts2rct(reye)
    leye=np.array(leye)
    reye = np.array(reye)
    leyex=int(1.0*sum(leye[:,0])/numpts_eye)
    leyey=int(1.0*sum(leye[:,1])/numpts_eye)
    reyex=int(1.0*sum(reye[:,0])/numpts_eye)
    reyey=int(1.0*sum(reye[:,1])/numpts_eye)
    eyewidth=max(max(leyerct[2]-leyerct[0],leyerct[3]-leyerct[1]),max(reyerct[2]-reyerct[0],reyerct[3]-reyerct[1]))

    # tarw=random.randint(90,120)
    # tarw=160
    # cropw=int(1.0*eyewidth/tarw*INSIZE)//2

    sizewin1024=384
    cropw=int(w/1024.0*sizewin1024*0.5)

    lcroprct=[leyex-cropw,leyey-cropw,leyex+cropw,leyey+cropw]
    rcroprct = [reyex - cropw, reyey - cropw, reyex + cropw, reyey + cropw]
    # img=cv2.imread(items[0])
    # img = np.array(Image.open(items[0]).convert('RGB'))

    leyeim = img[lcroprct[1]:lcroprct[3], lcroprct[0]:lcroprct[2]]
    reyeim=img[rcroprct[1]:rcroprct[3],rcroprct[0]:rcroprct[2]]
    # print (line)
    leyeim=cv2.resize(leyeim,(INSIZE,INSIZE))
    reyeim = cv2.resize(reyeim, (INSIZE, INSIZE))
    # if random.random()>0.5:
    #     return leyeim
    return leyeim,reyeim



if __name__ == '__main__':

    srcroot=r'/home/tao/mynas/Dataset/hairforsr/femalehd_crop2048'

    eye_dstroot=srcroot+'_eyescrop'

    os.makedirs(eye_dstroot,exist_ok=True)

    faceland_root=srcroot+'_faceland'
    ims=get_ims(srcroot)




    numim=len(ims)
    for i,im in enumerate(ims):

        print('{}of{}'.format(i,numim))


        imkey,ext=get_imkey_ext(im)

        npypath_face=os.path.join(faceland_root,imkey+'.npy')

        landmarks=np.load(npypath_face)
        # print(landmarks.shape)


        imgori=cv2.imread(im,cv2.IMREAD_UNCHANGED)
        h,w,c=imgori.shape
        img=np.zeros((h,w,3),imgori.dtype)
        img[:,:,:]=imgori[:,:,0:3]




        # for pt in landmarks:
        #     cv2.circle(img, (pt[0], pt[1]), 10, (255, 0, 0), -1, -1)

        # eyeimg=get_eyeims(img,landmarks,512)
        eyeimgl,eyeimgr=get_eyeims_fix(img,landmarks,384*2)


        impath_eyel=os.path.join(eye_dstroot,imkey+'_eyel.jpg')
        impath_eyer=os.path.join(eye_dstroot,imkey+'_eyer.jpg')

        cv2.imwrite(impath_eyel,eyeimgl)
        cv2.imwrite(impath_eyer,eyeimgr)


        # cv2.imshow('img',limit_img_auto(img))
        # cv2.imshow('eyeimgl',eyeimgl)
        # cv2.imshow('eyeimgr',eyeimgr)



        # key=cv2.waitKey(0)
        # if key==27:
        #     exit(0)

