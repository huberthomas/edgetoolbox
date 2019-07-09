import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(img: any = np.array,
          threshold1: int = 100,
          threshold2: int = 150,
          kernelSize: int = 3,
          highAccuracy: bool = True,
          blurKernelSize: int = 3) -> np.array:
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

    c = img.shape[2]

    if c >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edgeImg = cv2.blur(img, (blurKernelSize, blurKernelSize))
    cv2.Canny(edgeImg, threshold1, threshold2, edgeImg, kernelSize, highAccuracy)

    return edgeImg
