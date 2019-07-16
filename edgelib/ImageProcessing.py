import cv2
import numpy as np
import matplotlib.pyplot as plt


def reconstructDepthImg(porousDepthImg: np.array = None, 
                        inpaintRadius: int = 5, 
                        inpaintMethod: int = cv2.INPAINT_TELEA) -> np.array:
    '''
    Reconstruct porous areas in a depth image. Uses inpainting to fill up
    undefined regions in a depth image.

    porousDepthImage Single channel depth image.

    Returns inpainted depth image.
    '''
    if porousDepthImg is None:
        raise ValueError('Input image is empty.')

    if len(porousDepthImg.shape) > 2:
        raise ValueError('Invalid input image. Must be single channel.')

    _, mask = cv2.threshold(porousDepthImg, 0, 1, cv2.THRESH_BINARY_INV)
    mask = np.uint8(mask)

    restoredImg = cv2.inpaint(porousDepthImg, mask, inpaintRadius, inpaintMethod)

    # plt.figure(1)
    # plt.subplot(131)
    # plt.imshow(porousDepthImg)
    # plt.subplot(132)
    # plt.imshow(mask)
    # plt.subplot(133)
    # plt.imshow(restoredImg)
    # plt.show()

    return restoredImg
