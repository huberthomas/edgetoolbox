from ast import IsNot
from fileinput import filename
import os
import unittest
import sys
import cv2 as cv
import numpy as np
sys.path.append(".")
from edgelib import Canny

class TestCanny(unittest.TestCase):
    '''
    Test camera methods.
    '''
    def testCanny(self):
        self.assertTrue('FOO'.isupper())
        
        baseDir = 'D:\\Nextcloud\\master\\master_thesis\\assets\\chapter03\\test_dataset\\tum\\rgbd_dataset_freiburg2_xyz\\rgb'
        outputDir = 'D:\\Nextcloud\\master\\master_thesis\\assets\\chapter03\\canny_steps\\inter-image-exposure\\'
        baseDir = 'D:\\Nextcloud\\master\\master_thesis\\assets\\chapter03\\canny_steps\\inter-image-exposure\\'
        
        fileName = '1311867277.948933.png'
        # fileName = 'test.png'
        img = cv.imread(os.path.join(baseDir, fileName), cv.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError('Image must be set.')

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cannyEdgeDetector = Canny.Canny(sigma = 1, kernelSize = 3, weakPixel = 100, strongPixel = 255, lowThreshold = 0.05, highThreshold = 0.09)
        edge = cannyEdgeDetector.detect(gray)

        # cv.imshow("nonMaxImg",cannyEdgeDetector.nonMaxImg.astype(np.double))
        # cv.imshow("thresholdImg",cannyEdgeDetector.thresholdImg.astype(np.double))
        # cv.imshow("gradientMat",cannyEdgeDetector.gradientMat.astype(np.double))
        # cv.imshow("thetaMat",cannyEdgeDetector.thetaMat.astype(np.double))
        # cv.imshow("edge",edge.astype(np.double))
        # cv.imshow("gray",gray.astype(np.double))
        # cv.imshow("smoothed",cannyEdgeDetector.imgSmoothed.astype(np.double))
        # key = cv.waitKey(0)
        # cv.destroyAllWindows()

        cv.imwrite(os.path.join(outputDir, "gray_" + fileName), gray.astype(np.uint8))
        cv.imwrite(os.path.join(outputDir, "smoothed_" + fileName), cannyEdgeDetector.imgSmoothed.astype(np.uint8))
        cv.imwrite(os.path.join(outputDir, "grad_" + fileName), 255 - cannyEdgeDetector.gradientMat.astype(np.uint8))
        cv.imwrite(os.path.join(outputDir, "mag_" + fileName), cannyEdgeDetector.thetaMat.astype(np.uint8))
        cv.imwrite(os.path.join(outputDir, "nonMax_" + fileName), 255 - cannyEdgeDetector.nonMaxImg.astype(np.uint8))
        cv.imwrite(os.path.join(outputDir, "threshold_" + fileName), 255 - cannyEdgeDetector.thresholdImg.astype(np.uint8))
        cv.imwrite(os.path.join(outputDir, "edge_" + fileName), 255 - edge.astype(np.uint8))

    # def testCannySettings(self):
    #     baseDir = 'D:\\Nextcloud\\master\\master_thesis\\assets\\chapter03\\test_dataset\\tum\\rgbd_dataset_freiburg2_xyz\\rgb'
    #     outputDir = 'D:\\Nextcloud\\master\\master_thesis\\assets\\chapter03\\canny_steps'
    #     fileName = '1311867277.948933.png'
    #     img = cv.imread(os.path.join(baseDir, fileName), cv.IMREAD_UNCHANGED)

    #     if img is None:
    #         raise ValueError('Image must be set.')

    #     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     thresholds=[50,100,150,200]
    #     sigmas=[3]
    #     for thres in thresholds:
    #         for sigma in sigmas:
    #             edge = cv.Canny(gray, thres, thres, None, sigma, True)
    #             inv_edge = cv.bitwise_not(edge)
    #             cv.imwrite(os.path.join(outputDir, "edge_" + str(thres) + "_" + str(sigma) + "_" + fileName), inv_edge)

if __name__ == '__main__':
    '''
    Entry function.

    python -m unittest test/cannyTest.py -v
    '''
    unittest.main()