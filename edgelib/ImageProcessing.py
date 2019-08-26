import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from .Camera import Camera
from .Frame import Frame
from .EdgeMatcherFrame import EdgeMatcherFrame
from typing import List

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

    Returns interpolated value.
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

    Returns heatmap of image.
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

def getGradientInformation(img: np.ndarray = None) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    '''
    Calculate gradient magnitude and orientation of an image by central differences.

    img Image. Multichannel images will be converted to single channel.

    Returns dx, dy, gradient magnitude and orientation.
    '''
    if img is None:
        raise ValueError('Invalid image.')

    c = img.ndim

    if c == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif c == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    
    # multidirectional sobel https://www.researchgate.net/publication/314446743_A_Novel_Approach_for_Color_Image_Edge_Detection_Using_Multidirectional_Sobel_Filter_on_HSV_Color_Space
    # https://pdfs.semanticscholar.org/7e87/b109d5c0205c0fc36a521ef88a860b4b5acf.pdf
    #tmpRgb = destFrame.depth()
    #tmpRgb = cv.blur(tmpRgb, (blurKernelSize, blurKernelSize))
    #tmpRgb = cv.cvtColor(tmpRgb, cv.COLOR_BGR2HSV)

    # dxImg = []
    # dyImg = []
    # magImg = []
    # angleImg = []
    #for c in range(tmpRgb.ndim):
        #if tmpRgb.ndim >= 3:
        #    dy, dx = np.gradient(tmpRgb[:,:,c].astype(np.float64))
        #else:
            # dy, dx = np.gradient(tmpRgb.astype(np.float64))
            # mag = np.sqrt(dx*dx + dy*dy)
            # angle = np.dot(np.arctan(np.divide(dy, dx)), 180.0/np.pi)
    # dxImg.append(dx)
    # dyImg.append(dy)
    # magImg.append(mag)
    # angleImg.append(angle)

    # dxImg = np.stack(dxImg, axis=2)
    # dyImg = np.stack(dyImg, axis=2)
    # magImg = np.stack(magImg, axis=2)
    # angleImg = np.stack(angleImg, axis=2)

    dy, dx = np.gradient(img.astype(np.float64))
    #magnitude = np.sqrt(dx**2 + dy**2)
    magnitude = np.hypot(dx, dy)
    magnitude = np.divide(magnitude, magnitude.max())
    #orientation = np.dot(np.arctan(np.divide(dy, dx)), 180.0/np.pi) # -90 < x < 90
    orientation = np.dot(np.arctan2(dy, dx), 180.0/np.pi) # -180 < x < 180

    #angle[np.where(meaningfulEdges==0)] = None
    #mag[np.where(meaningfulEdges==0)] = 0
    # amax = np.amax(mag)
    # mag[np.where(mag<amax/2.0)] = None
    # im_conv = np.stack(ims, axis=2).astype("uint8")

    return (dx, dy, magnitude, orientation)


def projectEdges(frameFrom: EdgeMatcherFrame = None, 
                 frameTo: EdgeMatcherFrame = None, 
                 takeValuesFromDistTrans: bool = False, 
                 camera: Camera = None, 
                 edgeDistanceBoundaries: tuple = (0,0),
                 result: {} = None) -> np.ndarray:
    '''
    Project edges from one frame to another.

    frameFrom Frame that should be projected to.

    frameTo Frame on that is projected.

    distTransMat Distance matrix that contains the distance transform values.

    takeValuesFromDistTrans Take the distance from the reprojected coordinates.

    camera Camera matrix.

    edgeDistanceBoundaries Boundaries for edge classification: best, good, worse.

    result Stores result in dictionary: key = frameTo.uid, value = see return value

    Returns matrix that contains the result of the reprojection.
    1 channel: best with distance <= edgeDistanceLowerBoundary
    2 channel: good with edgeDistanceLowerBoundary < distance <= edgeDistanceUpperBoundary
    3 channel: worse with distance > edgeDistanceUpperBoundary
    '''
    if not frameFrom.isValid():
        raise ValueError('Invalid frame from.')

    if not frameTo.isValid():
        raise ValueError('Invalid frame to.')

    if camera is None:
        raise ValueError('Invalid camera.')

    if camera.depthScaleFactor() == 0:
        raise ValueError('Invalid depth scale factor.')

    distTransMat = frameTo.distanceTransform()

    h, w = frameFrom.boundaries().shape
    reprojectedEdges = np.zeros((h, w, 3))

    for u in range(0, w):
        for v in range(0, h):

            if frameFrom.boundaries().item(v, u) == 0:
                continue

            if frameFrom.depth().item(v, u) == 0:
                continue

            z = np.float64(frameFrom.depth().item(v, u)) / np.float64(camera.depthScaleFactor())

            if z == 0:
                continue

            # project to world coordinate system
            X = (u - camera.cx()) * z / camera.fx()
            Y = (v - camera.cy()) * z / camera.fy()
            # rotate, translate to other frame
            frameFromT = frameFrom.T()
            frameToInvT = frameTo.invT()
            p1 = frameFromT.item((0, 0)) * X + frameFromT.item((0, 1)) * Y + frameFromT.item((0, 2)) * z + frameFromT.item((0, 3))
            p2 = frameFromT.item((1, 0)) * X + frameFromT.item((1, 1)) * Y + frameFromT.item((1, 2)) * z + frameFromT.item((1, 3))
            p3 = frameFromT.item((2, 0)) * X + frameFromT.item((2, 1)) * Y + frameFromT.item((2, 2)) * z + frameFromT.item((2, 3))
            q1 = frameToInvT.item((0, 0)) * p1 + frameToInvT.item((0, 1)) * p2 + frameToInvT.item((0, 2)) * p3 + frameToInvT.item((0, 3))
            q2 = frameToInvT.item((1, 0)) * p1 + frameToInvT.item((1, 1)) * p2 + frameToInvT.item((1, 2)) * p3 + frameToInvT.item((1, 3))
            q3 = frameToInvT.item((2, 0)) * p1 + frameToInvT.item((2, 1)) * p2 + frameToInvT.item((2, 2)) * p3 + frameToInvT.item((2, 3))
            # project found 3d point Q back to the image plane
            U = (q1 / q3) * camera.fx() + camera.cx()
            V = (q2 / q3) * camera.fy() + camera.cy()
            # boundary check
            if U < 0 or V < 0 or U > (w-1) or V > (h-1):
                continue

            distVal = getInterpolatedElement(distTransMat, U, V)
        
            poi = np.array([u, v])

            if takeValuesFromDistTrans:
                poi = np.array([int(U), int(V)])

            edgeDistanceLowerBoundary = edgeDistanceBoundaries[0]
            edgeDistanceUpperBoundary = edgeDistanceBoundaries[1]

            if distVal <= edgeDistanceLowerBoundary:
                reprojectedEdges.itemset((poi[1], poi[0], 0), reprojectedEdges.item((poi[1], poi[0], 0)) + 1)
            elif distVal > edgeDistanceLowerBoundary and distVal <= edgeDistanceUpperBoundary:
                reprojectedEdges.itemset((poi[1], poi[0], 1), reprojectedEdges.item((poi[1], poi[0], 1)) + 1)
            elif distVal > edgeDistanceUpperBoundary:
                reprojectedEdges.itemset((poi[1], poi[0], 2), reprojectedEdges.item((poi[1], poi[0], 2)) + 1)

    if result is not None:
        result[frameTo.uid] = reprojectedEdges

    return reprojectedEdges