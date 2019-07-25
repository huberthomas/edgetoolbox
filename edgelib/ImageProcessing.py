import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from .Camera import Camera


def reconstructDepthImg(porousDepthImg: np.ndarray = None,
                        inpaintRadius: int = 5,
                        inpaintMethod: int = cv.INPAINT_TELEA) -> np.ndarray:
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

    _, mask = cv.threshold(porousDepthImg, 0, 1, cv.THRESH_BINARY_INV)
    mask = np.uint8(mask)

    restoredImg = cv.inpaint(porousDepthImg, mask, inpaintRadius, inpaintMethod)

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

    return np.array([u, v], np.float64)


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

    return np.array([X, Y, Z], np.float64)


def getInterpolatedElement(mat: np.ndarray = None, x: float = None, y: float = None) -> float:
    '''
    Bilinear interpolation
    see https: // github.com/JakobEngel/dso/blob/master/src/util/globalFuncs.h
    '''
    if mat is None:
        raise ValueError('Invalid input matrix.')

    h, w = mat.shape

    if x < 0 or x > (w - 1) or y < 0 or y > (h - 1):
        raise ValueError('Out of bounds.')

    ix = np.int(x)
    iy = np.int(y)
    dx = np.float64(x - ix)
    dy = np.float64(y - iy)
    dxdy = dx * dy
    index = np.int(ix + (iy * w))

    res = dxdy * mat.item(index + 1 + w)
    + (dy - dxdy) * mat.item(index + w)
    + (dx - dxdy) * mat.item(index + 1)
    + (1 - dx - dy + dxdy) * mat.item(index)

    return res


def createHeatmap(img: np.ndarray = None, colormap: int = cv.COLORMAP_JET) -> np.ndarray:
    '''
    Create heatmap of an image with a defined color mapping.

    img Input image.

    colormap OpenCV colormap, e.g. COLORMAP_HOT
    '''
    if img is None:
        raise ValueError('Invalid input image.')
      
    h, w = img.shape[:2]

    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif img.ndim == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

    heatmap = np.zeros((h, w), np.uint8)
    cv.normalize(img, heatmap, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    
    return cv.applyColorMap(heatmap, colormap)
