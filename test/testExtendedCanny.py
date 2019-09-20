import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from edgelib import Utilities
from edgelib import ImageProcessing

def main():
    '''
    Main function.
    '''
    baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/bsr_bsds500/BSR/BSDS500/data/images/test_png'
    fileNames = Utilities.getFileNames(baseDir)

    for fileName in fileNames:
        img = cv.imread(os.path.join(baseDir, fileName), cv.IMREAD_UNCHANGED)
        c = img.ndim

        edgePreservedBlurred = cv.edgePreservingFilter(img, None, flags=2, sigma_r=0.6)

        if c == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            edgePreservedBlurred = cv.cvtColor(edgePreservedBlurred, cv.COLOR_BGR2GRAY)
        elif c == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY) 
            edgePreservedBlurred = cv.cvtColor(edgePreservedBlurred, cv.COLOR_BGRA2GRAY)

        bilateralBlurred = cv.bilateralFilter(img, 7, 50, 50)
        # guidedBlurred = cv.ximgproc.guidedFilter(img,13,70)  
        blurKernelSize = 5
        blurredImg = cv.blur(img, (blurKernelSize, blurKernelSize))

        # see https://www.learnopencv.com/non-photorealistic-rendering-using-opencv-python-c/

        # medianBilFiltered = ImageProcessing.medianCanny(bilateralBlurred)
        # otsuBilFiltered = ImageProcessing.otsuCanny(bilateralBlurred)
        # medianEpFiltered = ImageProcessing.medianCanny(edgePreservedBlurred)
        # otsuEpFiltered = ImageProcessing.otsuCanny(edgePreservedBlurred)
        normalizedBoxFiltered = cv.Canny(blurredImg, 50, 100, None, 3, True)

        # fig = plt.figure(1)
        # plt.subplot(321)
        # plt.title('Bilateral Blurred')
        # plt.imshow(bilateralBlurred)
        
        # plt.subplot(322)
        # plt.title('Guided Blurred')
        # plt.imshow(guidedBlurred)
        
        # plt.subplot(323)
        # plt.title('Median Bilateral')
        # plt.imshow(medianBilFiltered)
        
        # plt.subplot(324)
        # plt.title('Median Edge Preserved')
        # plt.imshow(medianEpFiltered)

        # plt.subplot(325)
        # plt.title('Otsu Bilateral')
        # plt.imshow(otsuBilFiltered)

        # plt.subplot(326)
        # plt.title('Otsu Edge Preserved')
        # plt.imshow(otsuEpFiltered)

        # plt.show()
        # cv.imwrite(checkPath(os.path.join(baseDir, 'cannyMedianBilateral', fileName)), medianBilFiltered)
        # cv.imwrite(checkPath(os.path.join(baseDir, 'cannyMedianEdgePreserved', fileName)), medianEpFiltered)
        # cv.imwrite(checkPath(os.path.join(baseDir, 'cannyOtsuBilateral', fileName)), otsuBilFiltered)
        # cv.imwrite(checkPath(os.path.join(baseDir, 'cannyOtsuEdgePreserved', fileName)), otsuEpFiltered)
        cv.imwrite(checkPath(os.path.join(baseDir, 'cannyNormalizedBoxFiltered', fileName)), normalizedBoxFiltered)
        
def checkPath(path: str = None) -> str:
    '''
    '''
    dirPath = path
    if not os.path.isdir(path):
        dirPath = os.path.dirname(path)

    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    return path

if __name__ == '__main__':
    '''
    Entry function.
    '''
    main()
    