import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from Camera import Camera


def reconstructDepthImg(porousDepthImg: np.ndarray = None,
                        inpaintRadius: int = 5,
                        inpaintMethod: int = cv2.INPAINT_TELEA) -> np.ndarray:
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


def projectToImage(camera: Camera = None, xyzCoords: np.ndarray = None) -> np.ndarray:
    '''
    Project 3D coordinates to an image.

    camera Camera instance that contains focal length and camera center.

    xyzCoords 3D coordinates that should be projected to an image plane.

    Returns 2D coordinates on an image plane.
    '''
    if camera is None:
        raise ValueError('Invalid camera.')

    if camera.fx() == 0 or camera.fy() == 0:
        raise ValueError('Focal length of camera must be greater than 0.')

    if type(xyzCoords) is not np.ndarray:
        raise ValueError('Invalid coordinate type. Must be from type numpy.')

    z = xyzCoords[2]

    if z == 0:
        raise ValueError('Invalid z coordinate. Must be greater than 0.')

    u = (xyzCoords[0] / z) * camera.fx() + camera.cx()
    v = (xyzCoords[1] / z) * camera.fy() + camera.cy()

    return np.array([u, v])


def projectToWorld(camera: Camera = None, uvzCoords: np.ndarray = None) -> np.ndarray:
    '''
    Project 2D coordinates to the world coordinate system.

    camera Camera instance that contains focal length and camera center.

    uvzCoords 2D coordinates and depth that should be projected to the world coordinate system.

    Returns 3D coordinates in the world coordinate system.
    '''
    if camera is None:
        raise ValueError('Invalid camera.')

    if camera.fx() == 0 or camera.fy() == 0:
        raise ValueError('Focal length of camera must be greater than 0.')

    if type(uvzCoords) is not np.ndarray:
        raise ValueError('Invalid coordinate type. Must be from type numpy.')

    Z = uvzCoords[2]
    X = (uvzCoords[0] - camera.cx()) * Z / camera.fx()
    Y = (uvzCoords[1] - camera.cy()) * Z / camera.fy()

    return np.array([X, Y, Z])


def getInterpolatedElement(mat: np.ndarray = None, x: float = None, y: float = None, width: int = None) -> float:
    '''
    Bilinear interpolation
    see https: // github.com/JakobEngel/dso/blob/master/src/util/globalFuncs.h
    '''
    ix = np.int(x)
    iy = np.int(y)
    dx = x - ix
    dy = y - iy
    dxdy = dx * dy
    bp = mat + ix + (iy * width)

    res = dxdy * bp[1 + width]
    + (dy - dxdy) * bp[width]
    + (dx - dxdy) * bp[1]
    + (1 - dx - dy + dxdy) * bp[0]

    return res

def createHeatmap(img: np.array = None, colormap: int = cv2.COLORMAP_HOT) -> np.ndarray:
    '''
    Create heatmap of an image with a defined color mapping.

    img Input image.

    colormap OpenCV colormap, e.g. COLORMAP_HOT
    '''
    if img is None:
        raise ValueError('Invalid input image.')

    h, w, _ = img.shape
    heatmap = np.zeros((h, w))
    cv2.normalize(img, heatmap, 0, 255, cv2.NORM_MINMAX)
    cv2.applyColorMap(heatmap, heatmap, colormap)
