#coding:utf-8

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import os
import numpy as np
import shutil
import platform,math

import argparse
import cv2
import torch

from facexlib.alignment import init_alignment_model, landmark_98_to_68
from facexlib.detection import init_detection_model
from facexlib.visualization import visualize_alignment


def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

face_template_512=[[200,245],[315,245],[256,370]]

def crop_face_bypt(src_pts,srcimg):
    srctp=np.array(face_template_512,np.float32)
    dsttp=np.array(src_pts,np.float32)
    A = cv2.getAffineTransform( dsttp,srctp)
    res=cv2.warpAffine(srcimg,A,(512,512))
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


def crop_face_by2pt(src_pts,srcimg):
    face_template_512_new=np.array(face_template_512,np.float32)/4.0
    rt=get_rotate_pt(face_template_512_new[0:2])
    face_template_512_new[2]=np.array(rt)


    src_pts_new=src_pts
    rt=get_rotate_pt(src_pts_new[0:2])
    src_pts_new[2]=np.array(rt)


    dsttp=np.array(src_pts_new,np.float32)
    srctp=np.array(face_template_512_new,np.float32)
    A = cv2.getAffineTransform( dsttp,srctp)
    res=cv2.warpAffine(srcimg,A,(128,128))
    # cv2.resize(res,(128,128))
    return res


if __name__=='__main__':
    # cap = cv2.VideoCapture(0)
    # align_net = init_alignment_model('awing_fan')
    det_net = init_detection_model('retinaface_resnet50', half=False)

    srcroot='/home/tao/mynas/Dataset/hairforsr/femalehd/'
    # dstroot='/disks/disk1/Dataset/Project/SuperResolution/taobao_stand_face/'
    dstroot='/home/tao/mynas/Dataset/hairforsr/femalehd_crop2048'

    os.makedirs(dstroot,exist_ok=True)

    ims=get_ims(srcroot)

    face_size=1024
    face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                   [201.26117, 371.41043], [313.08905, 371.15118]])
    face_template = face_template * (face_size / 512.0)

    resizeratio=0.25

    with torch.no_grad():

        for i,im in enumerate(ims):
            frame=cv2.imread(im)

            frameresize=cv2.resize(frame,(0,0),fx=resizeratio,fy=resizeratio)
            bboxes = det_net.detect_faces(frameresize, 0.97)

            for box in bboxes:
                # print (box)
                box/=resizeratio

                rct = box[0:4].astype(np.int)
                land = box[5:5 + 10].reshape((5, 2)).astype(np.int)

                # cv2.rectangle(frame, (rct[0], rct[1]), (rct[2], rct[3]), (0, 0, 255), 2)
                # for pt in land:
                #     print (pt)
                #     cv2.circle(frame, (pt[0], pt[1]), 3, (255, 0, 0), -1, -1)

                # face_ctl_pts=np.array([land[0],land[1],0.5*(land[3]+land[4])],np.float32)
                # facealign=crop_face_bypt(face_ctl_pts, frame)
                # facealign=crop_face_by2pt(face_ctl_pts, frame)

                affine_matrix = cv2.estimateAffinePartial2D(land, face_template, method=cv2.LMEDS)[0]
                facealign= cv2.warpAffine(frame, affine_matrix, (face_size,face_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))

            # landmarks = align_net.get_landmarks(frame)
            # landmarks = landmarks.astype(np.int)

            cv2.imwrite(os.path.join(dstroot,os.path.basename(im)),facealign)

            print(i)

            continue

            # print (landmarks)
            #
            # for pt in landmarks:
            #     cv2.circle(frame,(pt[0],pt[1]),3,(255,0,0),-1,-1)

            cv2.imshow("capture", frame)
            cv2.imshow("facealign", facealign)

            # if cv2.waitKey(10) & 0xff == ord('q'):
            #     break
            key =cv2.waitKey(0)
            if key==27:
                break


