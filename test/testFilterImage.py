import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from edgelib import Utilities

def getFilteredImage(img: np.ndarray = None, cutOffLowerFactor: float = None, cutOffUpperFactor: float = None):
    '''
    Filter image by upper and lower boundaries defined by the cut off factor from all
    pixel values > 0.

    img Input image that should be filtered.

    cutOffLowerFactor Boundary for lower cut off, between [0 ... 1].

    cutOffLowerFactor Boundary for upper cut off, between [0 ... 1].

    Returns filtered image.
    '''
    if img is None:
        raise ValueError('Invalid image.')

    if cutOffUpperFactor is None:
        cutOffUpperFactor = cutOffLowerFactor

    if cutOffLowerFactor < 0:
        cutOffLowerFactor = abs(cutOffLowerFactor)

    if cutOffUpperFactor < 0:
        cutOffUpperFactor = abs(cutOffUpperFactor)

    img = img / img.max()

    values = np.sort(img, axis=None)
    values = values[np.where(values>0)]

    lowerIndex = int(cutOffLowerFactor * len(values))
    upperIndex = int((1-cutOffUpperFactor) * len(values))

    lowerBorder = values[lowerIndex]
    upperBorder = values[upperIndex]


    #img[np.where(np.logical_or(img<lowerBorder, img>upperBorder))] = 0
    img[np.where(img<lowerBorder)] = 0
    img[np.where(img>upperBorder)] = 0
    img /= img.max()

    return img

baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/test_dataset/hdr_fusion/flicker_synthetic/flicker_1/bdcn_40k'
baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/test_dataset/hdr_fusion/smooth_synthetic/flicker_2/bdcn_40k'
fileNames = Utilities.getFileNames(baseDir)

for fileName in fileNames:
    img = cv.imread(os.path.join(baseDir, fileName), cv.IMREAD_UNCHANGED)
    fig = plt.figure(1, figsize=(20,40))
    colorMap = None
    plt.subplot(221)
    plt.title('0%')
    plt.imshow(img, cmap=colorMap)
    plt.subplot(222)
    plt.title('5%')
    plt.imshow(getFilteredImage(img, 0.05), cmap=colorMap)
    plt.subplot(223)
    plt.title('10%')
    plt.imshow(getFilteredImage(img, 0.10), cmap=colorMap)
    plt.subplot(224)
    plt.title('50% - 5%')
    plt.imshow(getFilteredImage(img, 0.5, 0.05), cmap=colorMap)
    plt.show()


