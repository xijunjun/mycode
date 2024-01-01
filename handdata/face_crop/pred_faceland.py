# coding:utf-8

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import os
import numpy as np
import shutil
import platform, math

import argparse
import cv2
import torch

from facexlib.alignment import init_alignment_model, landmark_98_to_68
from facexlib.detection import init_detection_model
from facexlib.visualization import visualize_alignment


def get_ims(imgpath):
    imgpathlst = []
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg', '.jpeg', '.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst


face_template_512 = [[200, 245], [315, 245], [256, 370]]


def crop_face_bypt(src_pts, srcimg):
    srctp = np.array(face_template_512, np.float32)
    dsttp = np.array(src_pts, np.float32)
    A = cv2.getAffineTransform(dsttp, srctp)
    res = cv2.warpAffine(srcimg, A, (512, 512))
    return res


def get_rotate_pt(pts):
    theta = 90 / 180.0 * math.pi
    ptsnew = np.array(pts)
    ptsnew -= pts[0]
    pt = ptsnew[1]
    x, y = pt[0], pt[1]
    x2 = (x * math.cos(theta) - y * math.sin(theta))
    y2 = (y * math.cos(theta) + x * math.sin(theta))
    ptsnew[1] = [x2, y2]
    ptsnew += pts[0]

    return ptsnew[1]


def crop_face_by2pt(src_pts, srcimg):
    face_template_512_new = np.array(face_template_512, np.float32) / 4.0
    rt = get_rotate_pt(face_template_512_new[0:2])
    face_template_512_new[2] = np.array(rt)

    src_pts_new = src_pts
    rt = get_rotate_pt(src_pts_new[0:2])
    src_pts_new[2] = np.array(rt)

    dsttp = np.array(src_pts_new, np.float32)
    srctp = np.array(face_template_512_new, np.float32)
    A = cv2.getAffineTransform(dsttp, srctp)
    res = cv2.warpAffine(srcimg, A, (128, 128))
    # cv2.resize(res,(128,128))
    return res


def limit_img_longsize_scale(img, target_size):
    img_ = np.array(img)
    h, w, c = img.shape
    if w > h:
        scale = target_size / w
    else:
        scale = target_size / h
    scale = min(1.0, scale)
    tw, th = int(scale * w), int(scale * h)
    if scale < 1:
        img_ = cv2.resize(img_, (tw, th))
    return img_, scale


def rct2rctwh(rct):
    return np.array([rct[0], rct[1], rct[2] - rct[0], rct[3] - rct[1]])


def rctwh2rct(rct):
    return np.array([rct[0], rct[1], rct[2] + rct[0], rct[3] + rct[1]])


def bigger_rct(wratio, hratio, rctin):
    rct = rct2rctwh(rctin)
    wratio = (wratio - 1.0) * 0.5
    hratio = (hratio - 1.0) * 0.5
    delta_w = (rct[2]) * wratio + 0.5
    delta_h = (rct[3]) * hratio + 0.5
    # return limit_rct([int(rct[0]-delta_w),int(rct[1]-delta_h),int(rct[2]+delta_w*2),int(rct[3]+delta_h*2)],imgshape)
    rct = [rct[0] - delta_w, rct[1] - delta_h, rct[2] + delta_w * 2, rct[3] + delta_h * 2]
    rct = rctwh2rct(rct)
    return rct


def crop_pad(img, bdrctin):
    bdrct=np.array(bdrctin)

    h, w, c = img.shape

    extend_tlx = min(0, bdrct[0][0])
    extend_tly = min(0, bdrct[0][1])
    extend_brx = max(w, bdrct[1][0])
    extend_bry = max(h, bdrct[1][1])
    extendw = extend_brx - extend_tlx
    extendh = extend_bry - extend_tly
    bkimg = np.zeros((extendh, extendw, 3), img.dtype)
    # print('cropimg',extend_tlx,extend_tly)

    xshift = 0 - extend_tlx
    yshift = 0 - extend_tly
    bkimg[yshift:yshift + h, xshift:xshift + w] = img

    bdrct[:, 0] += xshift
    bdrct[:, 1] += yshift

    cropimg = bkimg[bdrct[0][1]:bdrct[1][1], bdrct[0][0]:bdrct[1][0]]
    return cropimg, xshift, yshift


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


def pt_trans(pts,param):

    dst=[]
    for pt in pts:
        x = pt[0]*param[0][0]+pt[1]*param[0][1]+param[0][2]
        y = pt[0] * param[1][0] + pt[1] * param[1][1] + param[1][2]
        dst.append([x,y])
    return  np.array(dst)

def pt_trans_one(pt,param):
    x = pt[0]*param[0][0]+pt[1]*param[0][1]+param[0][2]
    y = pt[0] * param[1][0] + pt[1] * param[1][1] + param[1][2]
    return  np.array([x,y])

def land98to5(land98):
    land98_=np.array(land98)

    pts5=[]
    indlist=list(range(60,69))
    indlist[-1]=96
    x,y=land98_[indlist][:,0].mean(),land98_[indlist][:, 1].mean()
    pts5.append([x,y])

    indlist=list(range(68,76))
    indlist[-1]=97
    x,y=land98_[indlist][:,0].mean(),land98_[indlist][:, 1].mean()
    pts5.append([x,y])

    pts5.append(land98_[54])
    pts5.append(land98_[76])
    pts5.append(land98_[82])

    return  np.array(pts5,np.int32)

def get_imkey_ext(impath):
    imname=os.path.basename(impath)
    ext='.'+imname.split('.')[-1]
    imkey=imname[0:len(imname)-len(ext)]
    return imkey,ext



def get_face_land_rct(det_net,align_net,framein):
    with torch.no_grad():
        frame=np.array(framein)
        img_scale, scale = limit_img_longsize_scale(frame, 512)
        bboxes = det_net.detect_faces(img_scale, 0.97) / scale

        bboxes=list(bboxes)
        bboxes.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),reverse=True)
        landmark_list=[]
        bbox_list=[]
        for box in bboxes:
            rct = box[0:4].astype(np.int32)
            land5 = box[5:5 + 10].reshape((5, 2)).astype(np.int32)
            exd_rct = np.array(bigger_rct(1.1, 1.1, rct), np.int32)
            # print('exd_rct[0], exd_rct[1]:',exd_rct[0], exd_rct[1])
            faceimg, xshift, yshift = crop_pad(frame, np.array(exd_rct.reshape(2, 2)))
            # cv2.rectangle(frame, (exd_rct[0], exd_rct[1]), (exd_rct[2], exd_rct[3]), (0, 255, 0), 2)
            # cv2.rectangle(frame, (rct[0], rct[1]), (rct[2], rct[3]), (0, 0, 255), 2)

            # cv2.imshow('frame',limit_img_auto(frame))
            # cv2.imshow('faceimg',limit_img_auto(faceimg))
            # print('exd_rct[0], exd_rct[1]:',exd_rct[0], exd_rct[1])

            landmarks = align_net.get_landmarks(faceimg) + np.array([exd_rct[0], exd_rct[1]])
            landmarks = np.array(landmarks, np.int32)
            landmark_list.append(landmarks)
            bbox_list.append(rct)

    return bbox_list,landmark_list



if __name__=='__main__':
    # cap = cv2.VideoCapture(0)
    align_net = init_alignment_model('awing_fan')
    det_net = init_detection_model('retinaface_resnet50', half=False)

    srcroot = '/home/tao/mynas/Dataset/hairforsr/femalehd'
    # dstroot =srcroot+'_crop2048'
    # # dstroot = '/home/tao/mynas/Dataset/hairforsr/femalehd_crop2048/'

    # faceland_root=dstroot+'_faceland'

    # os.makedirs( dstroot,exist_ok=True)
    # os.makedirs( faceland_root,exist_ok=True)



    ims = get_ims(srcroot)

    face_size = 2048
    face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                              [201.26117, 371.41043], [313.08905, 371.15118]])
    face_template = face_template * (face_size / 512.0)

    resizeratio = 0.25

    with torch.no_grad():
        numim=len(ims)
        for i, im in enumerate(ims):
            print('{}of{}'.format(i,numim),im)
            if  i!=10:
                continue

            imkey,ext=get_imkey_ext(im)



            all_face_rcts = []
            all_face_lands = []
            frame = cv2.imread(im)
            img_const=np.array(frame)
            # img_scale, scale = limit_img_longsize_scale(frame, 512)
            # bboxes = det_net.detect_faces(img_scale, 0.97) / scale

            bboxes, landmark_list = get_face_land_rct(det_net, align_net, img_const)

            for n,box in enumerate(bboxes):



                rct = box[0:4].astype(np.int)
                landmarks =landmark_list[n]
                landmarks = np.array(landmarks, np.int)

                for pt in landmarks:
                    # print(pt)
                    cv2.circle(frame, (pt[0], pt[1]), 10, (255, 0, 0), -1, -1)

                land5=land98to5(landmarks)
                # affine_matrix = cv2.estimateAffinePartial2D(land, face_template, method=cv2.LMEDS)[0]
                affine_matrix = cv2.estimateAffinePartial2D(land5, face_template, method=cv2.LMEDS)[0]
                facealign = cv2.warpAffine(img_const, affine_matrix, (face_size, face_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 135, 135))


                land_in_crop=pt_trans(landmarks,affine_matrix)
                land_in_crop=np.array(land_in_crop,dtype=np.int32)


                # for pt in land_in_crop:
                #     cv2.circle(facealign, (pt[0], pt[1]), 10, (255, 255, 0), -1, -1)
                # cv2.imshow('facealign',limit_img_auto(facealign))

                # faceland_root

                # imgpath_face=os.path.join(dstroot,imkey+'_'+str(n)+'.jpg')
                # npypath_face=os.path.join(faceland_root,imkey+'_'+str(n))
                # cv2.imwrite(imgpath_face,facealign)
                # np.save(npypath_face,land_in_crop)



            cv2.imshow("capture", limit_img_auto(frame))
            key = cv2.waitKey(0)
            if key==27:
                break
