import cv2
from config import *
import numpy as np
import os

image = cv2.imread(FILE_PATH)/ 255.0
ori_parsing = cv2.imread(PARSING_PATH)
ori_parsing = cv2.cvtColor(ori_parsing, cv2.COLOR_BGR2GRAY)

inpainting_mask = -1
ix = -1
iy = -1
temp_img = -1
temp_parsing = -1
dst = image
dst_img = image
parsing = ori_parsing / 255.0
dst_parsing = parsing
parsing = np.expand_dims(parsing,2)
ori_parsing = np.expand_dims(ori_parsing,2)
ori_255 = (dst_parsing * 255).astype("uint8")
flag = False
target = -1


def showContours(image,parsing):
    global target
    temp_img = image * 255
    temp_parsing = np.concatenate((parsing,parsing,parsing),2) * 255

    temp_parsing = temp_parsing.astype('uint8')
    temp_img = temp_img.astype('uint8')

    temp_parsing = cv2.cvtColor(temp_parsing, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(temp_parsing, 127, 255, 0)

    images, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(temp_img, contours, -1, (0, 0, 255), 1)
    target = img


def move(event,x,y, flags, param):
    global dst,dst_parsing,parsing,image,dst_img,ix,iy,ori_255,temp_img,temp_parsing,target,flag,inpainting_mask
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        temp_parsing,temp_img = dst_parsing,dst_img
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON and ori_255[iy,ix] != 0:
        flag = True
        rows, cols, _ = parsing.shape
        dx = x - ix
        dy = y - iy
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        dst_parsing = cv2.warpAffine(temp_parsing, M, (cols, rows))
        dst_img = cv2.warpAffine(temp_img, M, (cols, rows))

        dst_parsing = np.expand_dims(dst_parsing, 2)
        other_img = image * (1 - parsing)

        inpainting_mask = parsing - dst_parsing
        dst = other_img * (1 - dst_parsing) + dst_img * dst_parsing
        showContours(dst, dst_parsing)
        cv2.imshow('Operation', target)

    if event == cv2.EVENT_LBUTTONUP and flag:
        flag = False
        ori_255 = (dst_parsing * 255).astype("uint8")
        cv2.imshow("pic", dst)
        cv2.imshow("mask", inpainting_mask)

cv2.namedWindow('Operation')
cv2.setMouseCallback('Operation', move)
showContours(image, parsing)
cv2.imshow('Operation', target)

while(True):
    k = cv2.waitKey(1)
cv2.destroyAllWindows()
