import cv2
import numpy as np


def canny(img: np.ndarray = None,
          threshold1: int = 100,
          threshold2: int = 150,
          kernelSize: int = 3,
          highAccuracy: bool = True,
          blurKernelSize: int = 3) -> np.ndarray:
    '''
    Process Canny edge detection on a defined input image.

    img OpenCV input image.

    threshold1 First threshold for the hysteresis procedure.

    threshold2 Second threshold for the hysteresis procedure.

    kernelSize Kernel size for the sobel operator.

    highAccuracy If true, L2 gradient will be used for more accuracy.

    blurKernelSize Kernel size for the Sobel operator.

    Returns OpenCV image that contains edges.
    '''
    if img is None:
        raise ValueError('Input image is empty.')

    if threshold1 < 0 or threshold2 < 0:
        raise ValueError('Threshold must be greater than 0.')

    if threshold2 < threshold1:
        raise ValueError('Threshold 2 must be greater than threshold 1.')

    if kernelSize % 2 == 0 or kernelSize < 3:
        raise ValueError('Wrong kernel size. Allowed are 3, 5, 7, ...')

    if blurKernelSize % 2 == 0 or blurKernelSize < 3:
        raise ValueError('Wrong blur kernel size. Allowed are 3, 5, 7, ...')

    c = img.ndim

    if c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif c == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    blurredImg = cv2.blur(img, (blurKernelSize, blurKernelSize))

    return cv2.Canny(blurredImg, threshold1, threshold2, None, kernelSize, highAccuracy)


def cannyAscendingThreshold(img: np.ndarray = None,
                            threshold1: int = 0,
                            threshold2: int = 0,
                            kernelSize: int = 3,
                            highAccuracy: bool = True,
                            blurKernelSize: int = 3,
                            stepRange: int = 50,
                            validEdgesThreshold: float = 0.5) -> np.ndarray:
    '''
    Process Canny edge detection on a defined input image. The second threshold will be
    increased by the step range. The result will be accumulated until no edge is found. 
    Afterwards the valid edges threshold defines the point in which edges are counted as 
    good ones.

    img OpenCV input image.

    threshold1 First threshold for the hysteresis procedure.

    threshold2 Second threshold for the hysteresis procedure.

    kernelSize Kernel size for the sobel operator.

    highAccuracy If true, L2 gradient will be used for more accuracy.

    blurKernelSize Kernel size for the Sobel operator.

    stepRange Increasing step range for the second threshold.

    validEdgesThreshold Threshold that defines valid edges. Must be between 0 and 1.

    Returns OpenCV image that contains edges.
    '''
    if img is None:
        raise ValueError('Input image is empty.')

    if threshold1 < 0 or threshold2 < 0:
        raise ValueError('Threshold must be greater than 0.')

    if threshold2 < threshold1:
        raise ValueError('Threshold 2 must be greater than threshold 1.')

    if kernelSize % 2 == 0 or kernelSize < 3:
        raise ValueError('Wrong kernel size. Allowed are 3, 5, 7, ...')

    if blurKernelSize % 2 == 0 or blurKernelSize < 3:
        raise ValueError('Wrong blur kernel size. Allowed are 3, 5, 7, ...')

    if stepRange <= 0:
        raise ValueError('Invalid step. Must be greater than 0.')

    if validEdgesThreshold < 0 or validEdgesThreshold > 1:
        raise ValueError('Invalid threshold. Must be between 0 and less or equal 1.')

    h, w = img.shape[:2]
    c = img.ndim
    
    if c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif c == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    blurredImg = cv2.blur(img, (blurKernelSize, blurKernelSize))
    edgeImg = np.zeros((h, w), np.float64)
    resImg = np.zeros((h, w), np.float64)

    i = 0
    while(True):
        ascendingThreshold2 = threshold2 + i*stepRange

        edgeImg = cv2.Canny(blurredImg, threshold1, ascendingThreshold2, None, kernelSize, highAccuracy) / 255.0

        if np.sum(edgeImg) == 0:
            break

        resImg = cv2.add(resImg, edgeImg)
        i = i + 1

    validEdgesThreshold = i * validEdgesThreshold

    cv2.threshold(resImg, validEdgesThreshold, 255, cv2.THRESH_BINARY, edgeImg)

    return edgeImg
