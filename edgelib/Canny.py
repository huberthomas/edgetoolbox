import numpy as np
from scipy.ndimage import filters
from typing import List,Tuple
import cv2 as cv
import copy

class Canny:
    '''
    Canny algorithm.

    Refactored from https://github.com/FienSoP/canny_edge_detector/blob/master/canny_edge_detector.py

    https://blog.sicara.com/opencv-edge-detection-tutorial-7c3303f10788
    '''

    def __init__(self, sigma: float = 1, kernelSize: int = 5, weakPixel: int = 25, strongPixel: int = 255, lowThreshold: float = 0.05, highThreshold: float = 0.09):
        '''
        Constructor. Initialize values.

        imgs Array of input images.

        sigma Gaussian kernel parameter for smoothing.

        kernelSize Gaussian kernel size.

        weakPixel Weak pixel theshold for hysteresis.

        strongPixel Strong pixel threshold for hysteresis.

        lowThreshold Lower threshold.

        highThreshold High threshold.
        '''
        self.imgSmoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weakPixel = weakPixel
        self.strongPixel = strongPixel
        self.sigma = sigma
        self.kernelSize = kernelSize
        self.lowThreshold = lowThreshold
        self.highThreshold = highThreshold

    def toString(self) -> None:
        '''
        Output current Canny settings.
        '''
        print("")
        print("Weak pixel:", self.weakPixel)
        print("Strong pixel:", self.strongPixel)
        print("Sigma: ", self.sigma)
        print("Kernel size: ", self.kernelSize)
        print("Low threshold: ", self.lowThreshold)
        print("High threshold: ", self.highThreshold)

    def gaussianKernel(self, kernelSize: int = 5, sigma: float = 1) -> np.ndarray:
        '''
        Gaussian kernel.

        kernelSize Gaussian kernel size.

        sigma Gaussian kernel parameter.

        Returns a Gaussian kernel.
        '''
        kernelSize = int(kernelSize // 2)
        x, y = np.mgrid[-kernelSize:(kernelSize + 1), -kernelSize:(kernelSize + 1)]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g

    def sobelFilters(self, img: np.ndarray = None) -> tuple((np.ndarray, np.ndarray)):
        '''
        Filter an image with Sobel in x and y direction and return the gradient and the gradient direction.

        img Input image.

        Returns the gradient and the gradient direction.
        '''
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = filters.convolve(img, Kx)
        Iy = filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)

    def nonMaxSuppression(self, img: np.ndarray = None, direction: np.ndarray = None) -> np.ndarray:
        '''
        Non maximum suppression of a single channel image.

        img Single channel image that is non max suppressed.

        direction Gradient directions of the image.

        Returns the non max suppressed image.
        '''
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = direction * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i, j] >= q) and (img[i, j] >= r):
                        Z[i, j] = img[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        return Z

    def threshold(self, img: np.ndarray = None) -> np.ndarray:
        '''
        Image thresholding.

        img Input image that is thresholded.

        Returns thresholded image.
        '''
        highThreshold = img.max() * self.highThreshold
        lowThreshold = highThreshold * self.lowThreshold

        M, N = img.shape
        res = np.zeros((M, N), dtype=np.int32)

        weak = np.int32(self.weakPixel)
        strong = np.int32(self.strongPixel)

        strong_i, strong_j = np.where(img >= highThreshold)
        weak_i, weak_j = np.where(np.logical_and((img <= highThreshold), (img >= lowThreshold)))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)

    def hysteresis(self, img: np.ndarray = None) -> np.ndarray:
        '''
        Hysteresis process.

        img Single channel input image.

        Returns filtered image.
        '''
        M, N = img.shape
        weak = self.weakPixel
        strong = self.strongPixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i, j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                                or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img

    def detect(self, img: np.ndarray = None) -> np.ndarray:
        '''
        Detect edges.

        img Input image.

        Returns detected edge image.
        '''
        self.toString()
        img = img.astype(np.double)
        self.imgSmoothed = filters.convolve(img, self.gaussianKernel(self.kernelSize, self.sigma)) #cv.blur(img, (self.kernelSize, self.kernelSize))
        self.gradientMat, self.thetaMat = self.sobelFilters(self.imgSmoothed.astype(np.double))
        self.nonMaxImg = self.nonMaxSuppression(self.gradientMat.astype(np.double), self.thetaMat.astype(np.double))
        self.thresholdImg = self.threshold(self.nonMaxImg.astype(np.double))
        edge = self.hysteresis(self.thresholdImg.astype(np.double))

        # self.thetaMat = self.thetaMat * 180. / np.pi
        # self.thetaMat[self.thetaMat < 0] += 180
        #self.thetaMat = self.thetaMat.astype(np.double)
        #minVal = 0
        #maxVal = 0
        #min_val,max_val,min_indx,max_indx=cv.minMaxLoc(self.thetaMat)
        #self.thetaMat = (self.thetaMat / max_val)*255

        # print("img ",cv.minMaxLoc(img))
        # print("imgSmoothed" ,cv.minMaxLoc(self.imgSmoothed))
        # print("gradientMat ",cv.minMaxLoc(self.gradientMat.astype(np.double)))
        # print("thetaMat ",cv.minMaxLoc(self.thetaMat.astype(np.double)))
        # print("thresholdImg ",cv.minMaxLoc(self.thresholdImg))
        # print("edge ",cv.minMaxLoc(self.hysteresis(self.thresholdImg)))

        # x=264
        # y=309
        # print("")
        # print("imgSmoothed: ", self.imgSmoothed[x,y])
        # print("gradientMat: ", self.gradientMat[x,y])
        # print("thetaMat: ", self.thetaMat[x,y])
        # print("nonMaxImg: ", self.nonMaxImg[x,y])
        # print("thresholdImg: ", self.thresholdImg[x,y])
        return edge.astype(np.double)
