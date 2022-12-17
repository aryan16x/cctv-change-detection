import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

def f_saveSeq(C,counter,th,outputPath):
    if len(C)<th:
        pass
    else:
        k = 1
        for frame in C:
            imName = 'output'+str(counter)+'_'+str(k)+'.jpg'
            finalPath = os.path.join(outputPath,imName)
            bbox,labels,conf = cv.detect_common_objects(frame)
            frame = draw_bbox(frame,bbox,labels,conf)
            cv2.imwrite(finalPath,frame)
            k += 1
            
def f_keep_large_components(I,th):
    R = np.zeros(I.shape)<0
    unique_labels = np.unique(I.flatten())
    for label in unique_labels:
        if label==0:
            pass
        else:
            I2 = I==label
            if np.sum(I2)>th:
                R = R|I2
    return np.float32(255*R)

def f_displaySeq(im_path):
    for im_name in os.listdir(im_path):
        frame = cv2.imread(os.path.join(im_path,im_name))
        frame = cv2.resize(frame, dsize=(600,400))
        cv2.imshow('Image', frame)
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break

    cv2.destroyAllWindows()

im_path = r'D:\ComputerVision\RawImages'

fore_ground_model = cv2.createBackgroundSubtractorMOG2()
least_num_of_frames = 5
idx = []
C = []
counter = 0
outputPath = r'D:\ComputerVision\outputImg'

for im_name in os.listdir(im_path):
    counter += 1
    frame = cv2.imread(os.path.join(im_path,im_name))
    frame = cv2.resize(frame, dsize=(600,400))
    fgmask = fore_ground_model.apply(frame)
    K_r = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    fgmask = cv2.morphologyEx(np.float32(fgmask), cv2.MORPH_OPEN, K_r)
    num_labels,labels_im = cv2.connectedComponents(np.array(fgmask>0,np.uint8))
    fgmask = f_keep_large_components(labels_im,1000)
    
    if np.sum(fgmask)>0:
        idx.append(counter)
        C.append(frame)
    if len(idx)>=2 and idx[-1]>idx[-2]+1:
        f_saveSeq(C,counter,least_num_of_frames,outputPath)
        idx = []
        C = []
        
        
    F = np.zeros(frame.shape,np.uint8)
    F[:,:,0],F[:,:,1],F[:,:,2] = fgmask,fgmask,fgmask
    F2 = np.hstack((frame,F))
    cv2.imshow('Image', F2)
    k = cv2.waitKey(20) & 0xff
    if k==27:
        break

f_saveSeq(C,counter,least_num_of_frames,outputPath)
cv2.destroyAllWindows()

path = r'D:\ComputerVision\outputImg'
f_displaySeq(path)

