import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from .Camera import Camera
from .Frame import Frame
from . import Canny
from .EdgeMatcherFrame import EdgeMatcherFrame
from typing import List
import copy
import multiprocessing as mp
from multiprocessing import Manager

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


def removeIsolatedPixels(img: np.ndarray = None, area: int = 1, connectivity: int = 8):
    '''
    Removes isolated pixels from an image.

    area Isolated pixel area that should be removed.

    connectivity 4 and 8 neighbour connectivity. Default is 8.

    Returns cleared image.
    '''
    output = cv.connectedComponentsWithStats(img, connectivity, cv.CV_32S)

    numStats = output[0]
    labels = output[1]
    stats = output[2]

    newImg = img.copy()

    for label in range(numStats):
        if stats[label, cv.CC_STAT_AREA] <= area:
            newImg[labels == label] = 0

    return newImg

def getInterpolatedElement(mat: np.ndarray = None, x: float = None, y: float = None) -> float:
    '''
    Bilinear interpolation
    see https://github.com/JakobEngel/dso/blob/master/src/util/globalFuncs.h#L39

    mat Image.

    x Float value in x direction.

    y Float value in y direction.

    Returns interpolated value.
    '''
    if mat is None:
        raise ValueError('Invalid input matrix.')

    h, w = mat.shape

    if x < 0 or x > (w - 1) or y < 0 or y > (h - 1):
        raise ValueError('Out of bounds. x:%f, y:%f, w:%f, h:%f' % (x, y, w, h))

    ix = np.int(x)
    iy = np.int(y)
    dx = np.float64(x - ix)
    dy = np.float64(y - iy)
    dxdy = dx * dy

    # index = np.int(ix + (iy * w))
    # res2 = dxdy * mat.item(index + 1 + w) + (dy - dxdy) * mat.item(index + w) + (dx - dxdy) * mat.item(index + 1) + (1 - dx - dy + dxdy) * mat.item(index)

    res = dxdy * mat.item((iy + 1, ix + 1)) 
    res += (dy-dxdy) * mat.item((iy + 1, ix)) 
    res += (dx-dxdy) * mat.item((iy, ix + 1)) 
    res += (1 - dx - dy + dxdy) * mat.item((iy, ix))

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
    # for c in range(tmpRgb.ndim):
        # if tmpRgb.ndim >= 3:
        #    dy, dx = np.gradient(tmpRgb[:,:,c].astype(np.float64))
        # else:
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
    # orientation = np.dot(np.arctan(np.divide(dy, dx)), 180.0/np.pi) # -90 < x < 90
    orientation = np.dot(np.arctan2(dy, dx), 180.0/np.pi)  # -180 < x < 180

    #angle[np.where(meaningfulEdges==0)] = None
    #mag[np.where(meaningfulEdges==0)] = 0
    # amax = np.amax(mag)
    # mag[np.where(mag<amax/2.0)] = None
    # im_conv = np.stack(ims, axis=2).astype("uint8")

    return (dx, dy, magnitude, orientation)

def projectMultiscaleEdges(frameFrom: EdgeMatcherFrame = None,
                 frameTo: EdgeMatcherFrame = None,
                 camera: Camera = None,
                 edgeDistanceBoundaries: tuple = (0, 0),
                 scales: List[float] = [1],
                 result: List[any] = None) -> dict:
    '''
    '''
    edgeDistanceLowerBoundary = edgeDistanceBoundaries[0]
    edgeDistanceUpperBoundary = edgeDistanceBoundaries[1]
    edgeThresMin = 50
    edgeThresMax = 100
    edgeKernelSize = 3
    blurKernelSize = 3
    scaledReprojectedEdgesList = []

    for s in range(0, len(scales)):
        scale = scales[s]
        assert (scale != 0),'Invalid scale "%f". Must be greater than 0.'%(scale)

        scaledFrameTo = copy.deepcopy(frameTo)
        scaledFrameTo.setRgb(cv.resize(frameTo.rgb(), (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA))
        scaledFrameTo.setBoundaries(Canny.canny(scaledFrameTo.rgb(), edgeThresMin, edgeThresMax, edgeKernelSize, True, blurKernelSize))
        
        scaledCamera = copy.deepcopy(camera)
        scaledCamera.rescale(scale)
        
        # rescale values
        scaledEdgeDistanceLowerBoundary = edgeDistanceLowerBoundary * scale
        scaledEdgeDistanceUpperBoundary = edgeDistanceUpperBoundary * scale

        scaledReprojectedEdgesList.append(projectEdges(frameFrom, scaledFrameTo, camera, scaledCamera, (scaledEdgeDistanceLowerBoundary, scaledEdgeDistanceUpperBoundary)))

    result.append(scaledReprojectedEdgesList)

    return scaledReprojectedEdgesList


def projectEdges(frameFrom: EdgeMatcherFrame = None,
                 frameTo: EdgeMatcherFrame = None,
                 cameraFrom: Camera = None,
                 cameraTo: Camera = None,
                 edgeDistanceBoundaries: tuple = (0, 0),
                 result: {} = None) -> np.ndarray:
    '''
    Project edges from one frame to another.

    frameFrom Frame that should be projected to.

    frameTo Frame on that is projected.

    distTransMat Distance matrix that contains the distance transform values.

    takeInterpolatedPoint Take interpolated point for counting up rating.

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

    if cameraFrom is None:
        raise ValueError('Invalid camera.')

    if cameraFrom.depthScaleFactor() == 0:
        raise ValueError('Invalid depth scale factor.')

    edgeDistanceLowerBoundary = edgeDistanceBoundaries[0]
    edgeDistanceUpperBoundary = edgeDistanceBoundaries[1]

    frameFromH, frameFromW = frameFrom.boundaries().shape
    frameToH, frameToW = frameTo.boundaries().shape

    reprojectedEdges = np.zeros((frameFromH, frameFromW, 3))
    frameToDistanceTransform = frameTo.distanceTransform()

    ## draw matches
    keyPointsFrameFrom = []
    keyPointsFrameTo = []
    matches = []
    ## end of draw matches

    for u in range(0, frameFromW):
        for v in range(0, frameFromH):
            if frameFrom.boundaries().item((v, u)) == 0 or frameFrom.depth().item((v, u)) == 0:
                continue

            z = np.float64(frameFrom.depth().item((v, u))) / np.float64(cameraFrom.depthScaleFactor())
            # project to world
            P = projectToWorld(cameraFrom, np.array((u, v, z), np.float64))
            # transform
            Q = np.dot(frameTo.invT_R(), np.dot(frameFrom.R(), P) + frameFrom.t()) + frameTo.invT_t()
            # reproject to image plane
            q = projectToImage(cameraTo, Q)
            # boundary check
            if q[0] < 0 or q[1] < 0 or q[0] > (frameToW - 1) or q[1] > (frameToH - 1):
                continue

            distVal = getInterpolatedElement(frameToDistanceTransform, q[0], q[1])

            if distVal <= edgeDistanceLowerBoundary:
                reprojectedEdges.itemset((v, u, 0), reprojectedEdges.item((v, u, 0)) + 1)
                ## draw matches
                matchIndex = len(keyPointsFrameFrom)
                keyPointsFrameFrom.append(cv.KeyPoint(u, v, 1))
                keyPointsFrameTo.append(cv.KeyPoint(int(q[0]), int(q[1]), 1))

                if matchIndex % 50 == 0:
                    matches.append(cv.DMatch(matchIndex, matchIndex, 1))
                ## end of draw matches
            elif distVal > edgeDistanceLowerBoundary and distVal <= edgeDistanceUpperBoundary:
                reprojectedEdges.itemset((v, u, 1), reprojectedEdges.item((v, u, 1)) + 1)
            elif distVal > edgeDistanceUpperBoundary:
                reprojectedEdges.itemset((v, u, 2), reprojectedEdges.item((v, u, 2)) + 1)


    if result is not None:
        result[frameTo.uid] = reprojectedEdges

    # best, good, worse = cv.split(reprojectedEdges)
    # overlayRgb = copy.deepcopy(cv.cvtColor(frameFrom.rgb(), cv.COLOR_BGR2RGB))
    # overlayRgb[np.where(best > 0)]=[0, 255, 0]
    # overlayRgb[np.where(good > 0)]=[255, 255, 0]
    # overlayRgb[np.where(worse > 0)]=[255, 0, 0]
    # scaledFrameToRgb = copy.deepcopy(frameTo.rgb())
    # scaledFrameToBoundaries = copy.deepcopy(frameTo.boundaries())
    # scaledFrameToDistanceTransform = copy.deepcopy(frameTo.distanceTransform())
    # scaledFrameToRgb[np.where(scaledFrameToBoundaries > 0)] = [255, 255, 255]
    # scaledFrameToDistanceTransform[np.where(scaledFrameToBoundaries > 0)] = [255]
    # matchesImg = cv.drawMatches(overlayRgb, keyPointsFrameFrom, scaledFrameToRgb, keyPointsFrameTo, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # rgbDepthEdges = cv.addWeighted(cv.cvtColor(frameFrom.rgb(), cv.COLOR_BGR2RGBA), 0.7, cv.cvtColor(createHeatmap(frameFrom.depth().copy()), cv.COLOR_BGR2BGRA), 0.3, 0)
    # rgbDepthEdges[np.where(best > 0)]=[0, 255, 0, 255]
    # rgbDepthEdges[np.where(good > 0)]=[255, 255, 0, 255]
    # rgbDepthEdges[np.where(worse > 0)]=[255, 0, 0, 255]
    # fig = plt.figure(1)
    # fig.suptitle('Multiscale Algo: %d %d'%(scaledFrameToBoundaries.shape))
    # plt.subplot(211)
    # plt.axis('off')
    # plt.title('Output')
    # plt.imshow(rgbDepthEdges)
    # plt.subplot(212)
    # plt.axis('off')
    # plt.title('Matches')
    # plt.imshow(matchesImg)
    # plt.show()

    return reprojectedEdges


def projectEdgesOld(frameFrom: EdgeMatcherFrame = None,
                 frameTo: EdgeMatcherFrame = None,
                 takeInterpolatedPoint: bool = False,
                 camera: Camera = None,
                 edgeDistanceBoundaries: tuple = (0, 0),
                 scale: float = 1,
                 result: {} = None) -> np.ndarray:
    '''
    Project edges from one frame to another.

    frameFrom Frame that should be projected to.

    frameTo Frame on that is projected.

    distTransMat Distance matrix that contains the distance transform values.

    takeInterpolatedPoint Take interpolated point for counting up rating.

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

    h, w = frameTo.boundaries().shape
    reprojectedEdges = np.zeros((h, w, 3))

    for u in range(0, w):
        for v in range(0, h):

            if frameFrom.boundaries().item((v, u)) == 0:
                continue

            if frameFrom.depth().item((v, u)) == 0:
                continue

            z = np.float64(frameFrom.depth().item((v, u))) / np.float64(camera.depthScaleFactor())

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
            # rescale camera if necessary
            camera.rescale(scale)
            # project found 3d point Q back to the image plane
            U = (q1 / q3) * camera.fx() + camera.cx()
            V = (q2 / q3) * camera.fy() + camera.cy()
            # boundary check
            if U < 0 or V < 0 or U > (w-1) or V > (h-1):
                continue

            distVal = getInterpolatedElement(distTransMat, U, V)

            poi = np.array([u, v])

            if takeInterpolatedPoint:
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
