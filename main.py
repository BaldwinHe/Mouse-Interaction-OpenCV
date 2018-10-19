import cv2
from config import *
import numpy as np
import os

image = cv2.imread(FILE_PATH)/ 255.0
ori_parsing = cv2.imread(PARSING_PATH)
ori_parsing = cv2.cvtColor(ori_parsing, cv2.COLOR_BGR2GRAY)

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


def move(event,x,y, flags, param):
    global dst,dst_parsing,parsing,image,dst_img,ix,iy,ori_255,temp_img,temp_parsing
    if event == cv2.EVENT_LBUTTONDOWN:
        print(event,end=" ")
        print(flags,end="\n")
        ix, iy = x, y
        temp_parsing,temp_img = dst_parsing,dst_img
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON and ori_255[iy,ix] != 0:
        print(event, end=" ")
        print(ori_255[iy,ix],end =" ")
        print(flags, end="\n")
        flag = True
        rows, cols, _ = parsing.shape
        dx = x - ix
        dy = y - iy
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        dst_parsing = cv2.warpAffine(temp_parsing, M, (cols, rows))
        dst_img = cv2.warpAffine(temp_img, M, (cols, rows))

        dst_parsing = np.expand_dims(dst_parsing, 2)
        cv2.imshow('img', dst_img)
        cv2.imshow('parsing', dst_parsing)
        temp_img = image * (1 - parsing)

        inpainting_mask = parsing - dst_parsing
        dst = temp_img * (1 - dst_parsing) + dst_img * dst_parsing
    else:
        print(event, end=" ")
        print(ori_255[iy,ix], end=" ")
        print(flags, end="\n")
        ori_255 = (dst_parsing * 255).astype("uint8")

cv2.namedWindow('Operation')
cv2.setMouseCallback('Operation', move)
while(True):
    cv2.imshow('Operation',dst)
    k = cv2.waitKey(1)
cv2.destroyAllWindows()
