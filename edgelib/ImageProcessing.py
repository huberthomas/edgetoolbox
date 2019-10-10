import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
import copy
from typing import List
from .Camera import Camera
from .Frame import Frame
from .EdgeMatcherFrame import EdgeMatcherFrame
import time

'''
Image processing helper functions.
'''


def hysteresis(img: np.ndarray = None, lowThreshold: float = 0.05, highThreshold: float = 0.15) -> None:
    '''
    Perform hysteresis on an image.

    Single channel image.

    weakPixel Weak pixel theshold for hysteresis.

    strongPixel Strong pixel threshold for hysteresis.

    lowThreshold Lower threshold.

    highThreshold High threshold.
    '''
    if img is None:
        raise ValueError('Invalid image.')

    highThreshold = highThreshold
    lowThreshold = highThreshold * lowThreshold

    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where(np.logical_and((img <= highThreshold), (img >= lowThreshold)))

    strongPixel = 1
    weakPixel = 0.5
    M, N = img.shape
    thres = np.zeros((M, N), dtype=np.float64)
    thres[strong_i, strong_j] = strongPixel
    thres[weak_i, weak_j] = weakPixel

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (thres[i, j] == weakPixel):
                try:
                    if ((thres[i+1, j-1] == strongPixel) or (thres[i+1, j] == strongPixel) or (thres[i+1, j+1] == strongPixel)
                        or (thres[i, j-1] == strongPixel) or (thres[i, j+1] == strongPixel)
                            or (thres[i-1, j-1] == strongPixel) or (thres[i-1, j] == strongPixel) or (thres[i-1, j+1] == strongPixel)):
                        thres[i, j] = strongPixel
                    else:
                        thres[i, j] = 0
                except IndexError:
                    pass

    return thres


def extendImageCanvas(img: np.array = None, cols: int = 0, rows: int = 0) -> np.ndarray:
    '''
    Extend image canvas by mirroring the image in all directions.

    img Input image.

    cols New width.

    rows New height.

    Returns extended image with new dimensions.
    '''
    r, c = img.shape[:2]

    assert(cols > c or rows > r, 'New image size must be greater than the old one.')

    dr = rows - r
    dc = cols - c

    assert(dr >= 0 or dc >= 0, 'New image size is too big. Must be smaller or equal than twice the image size.')

    # horizontal
    flipudImg = np.flipud(img)
    # vertical
    fliplrImg = np.fliplr(img)
    flipudlrImg = np.fliplr(flipudImg)
    
    if img.ndim < 3:
        extendedImg = np.ones((rows, cols), img.dtype)
        extendedImg[:r, :c] = img
        extendedImg[r:, c:] = flipudlrImg[:dr, :dc]
        extendedImg[r:, :c] = flipudImg[:dr, :c]
        extendedImg[:r, c:] = fliplrImg[:r, :dc]
    else:
        extendedImg = np.ones((rows, cols, img.ndim), img.dtype)
        extendedImg[:r, :c, :] = img
        extendedImg[r:, c:, :] = flipudlrImg[:dr, :dc, :]
        extendedImg[r:, :c, :] = flipudImg[:dr, :c, :]
        extendedImg[:r, c:, :] = fliplrImg[:r, :dc, :]

    return extendedImg


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


def removeIsolatedPixels(img: np.ndarray = None, minIsolatedPixelArea: int = 1, connectivity: int = 8) -> (np.ndarray, np.ndarray):
    '''
    Removes isolated pixels from an image.

    area Isolated pixel area that should be removed.

    connectivity 4 and 8 neighbour connectivity. Default is 8.

    Returns cleaned image and cleared pixel mask.
    '''
    h, w = img.shape[:2]
    remained = img.copy()
    removed = np.zeros((h, w), np.uint8)

    if minIsolatedPixelArea == 0:
        return (remained, removed)

    if minIsolatedPixelArea < 0:
        minIsolatedPixelArea = abs(minIsolatedPixelArea)

    if connectivity != 4 and connectivity != 8:
        raise ValueError('Invalid connectivity value "%d".' % (connectivity))

    output = cv.connectedComponentsWithStats(img, connectivity, cv.CV_32S)

    numStats = output[0]
    labels = output[1]
    stats = output[2]

    for label in range(numStats):
        if stats[label, cv.CC_STAT_AREA] <= minIsolatedPixelArea:
            remained[labels == label] = 0
            removed[labels == label] = 1

    return (remained, removed)


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


'''
https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
https://github.com/FienSoP/canny_edge_detector
'''


def canny(img: np.ndarray = None,
          threshold1: int = 100,
          threshold2: int = 150,
          kernelSize: int = 3,
          highAccuracy: bool = True,
          blurKernelSize: int = 3) -> np.ndarray:
    '''
    
    Exlaination
    https://blog.sicara.com/opencv-edge-detection-tutorial-7c3303f10788


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
    edgePreservedBlurred = cv.edgePreservingFilter(img, None, flags=2, sigma_r=0.6)

    if c == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edgePreservedBlurred = cv.cvtColor(edgePreservedBlurred, cv.COLOR_BGR2GRAY)
    elif c == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
        edgePreservedBlurred = cv.cvtColor(edgePreservedBlurred, cv.COLOR_BGRA2GRAY)

    #blurredImg = cv.blur(img, (blurKernelSize, blurKernelSize))
    # edge preserving
    #blurredImg = cv.bilateralFilter(img, 9, 150, 150)

    # fig = plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(edgePreservedBlurred)
    # plt.subplot(122)
    # plt.imshow(otsuCanny(edgePreservedBlurred))
    # plt.show()

    #return cv.Canny(blurredImg, threshold1, threshold2, None, kernelSize, highAccuracy)
    #return medianCanny(blurredImg)
    return otsuCanny(edgePreservedBlurred)


def otsuCanny(img) -> np.ndarray:
    '''
    Automatic Canny threshold detection via Otsu's method.
    See https://en.wikipedia.org/wiki/Otsu%27s_method for more information.
    https://www.meccanismocomplesso.org/en/opencv-python-the-otsus-binarization-for-thresholding/

    img Input image.

    Returns edge image.
    '''
    c = img.ndim

    if c == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif c == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

    otsuThresMax, _ = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    otsuThresMin = 0.5 * otsuThresMax

    return cv.Canny(img, otsuThresMin, otsuThresMax, None, 3, True)


def medianCanny(img: np.ndarray = None,
                sigma: float = 0.33) -> np.ndarray:
    '''
    Automatic Canny threshold detection via statistical distribution.
    https://stackoverflow.com/questions/21324950/how-to-select-the-best-set-of-parameters-in-canny-edge-detection-algorithm-imple
    https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/

    img Input image.

    sigma For threshold determination. Default is 0.33 which is typical used in datascience.

    Returns edge image.
    '''
    # compute the median of the single channel pixel intensities
    c = img.ndim

    if c == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif c == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

    v = np.median(img)
    sigma = 0.33

    # apply automatic Canny edge detection using the computed median
    edgeThresMin = int(max(0, (1.0 - sigma) * v))
    edgeThresMax = int(min(255, (1.0 + sigma) * v))

    return cv.Canny(img, edgeThresMin, edgeThresMax, None, 3, True)


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
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif c == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)

    blurredImg = cv.blur(img, (blurKernelSize, blurKernelSize))
    edgeImg = np.zeros((h, w), np.float64)
    resImg = np.zeros((h, w), np.float64)

    i = 0
    while(True):
        ascendingThreshold2 = threshold2 + i*stepRange

        edgeImg = cv.Canny(blurredImg, threshold1, ascendingThreshold2, None, kernelSize, highAccuracy) / 255.0

        if np.sum(edgeImg) == 0:
            break

        resImg = cv.add(resImg, edgeImg)
        i = i + 1

    validEdgesThreshold = i * validEdgesThreshold

    cv.threshold(resImg, validEdgesThreshold, 255, cv.THRESH_BINARY, edgeImg)

    return edgeImg.astype(np.uint8)


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
                           cannyThresholds: List[tuple] = [(50, 100)],
                           cannyKernelSizes: List[tuple] = [(3, 3)],
                           result: List[any] = None) -> dict:
    '''
    '''
    edgeDistanceLowerBoundary = edgeDistanceBoundaries[0]
    edgeDistanceUpperBoundary = edgeDistanceBoundaries[1]
    scaledReprojectedEdgesList = []

    if len(cannyThresholds) < len(scales) and len(cannyKernelSizes) < len(scales):
        raise ValueError('Invalid parameter settings.')

    # fig = plt.figure(3)

    for s in range(0, len(scales)):
        edgeThresMin = cannyThresholds[s][0]
        edgeThresMax = cannyThresholds[s][1]
        edgeKernelSize = cannyKernelSizes[s][0]
        blurKernelSize = cannyKernelSizes[s][1]
        scale = scales[s]
        assert (scale != 0), 'Invalid scale "%f". Must be greater than 0.' % (scale)

        scaledFrameTo = copy.deepcopy(frameTo)

        if scale != 1:
            scaledFrameTo.setRgb(cv.resize(frameTo.rgb(), (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA))

        scaledBoundaries = canny(scaledFrameTo.rgb(), edgeThresMin, edgeThresMax, edgeKernelSize, True, blurKernelSize)
        #scaledBoundaries = autoCanny(scaledFrameTo.rgb())
        scaledFrameTo.setBoundaries(scaledBoundaries)

        # fig.add_subplot(1, len(scales), s+1)
        # plt.title('%d'%s)
        # plt.imshow(scaledFrameTo.boundaries())

        scaledCamera = copy.deepcopy(camera)
        scaledCamera.rescale(scale)

        # rescale values
        scaledEdgeDistanceLowerBoundary = edgeDistanceLowerBoundary * scale
        scaledEdgeDistanceUpperBoundary = edgeDistanceUpperBoundary * scale

        scaledReprojectedEdgesList.append(projectEdges(frameFrom, scaledFrameTo, camera, scaledCamera, (scaledEdgeDistanceLowerBoundary, scaledEdgeDistanceUpperBoundary)))

    # plt.show()
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
    # keyPointsFrameFrom = []
    # keyPointsFrameTo = []
    # matches = []
    ## end of draw matches

    '''
     ## draw matches
    # keyPointsFrameFrom = []
    # keyPointsFrameTo = []
    # matches = []
    ## end of draw matches
    validV, validU = np.where(np.bitwise_and(frameFrom.boundaries() > 0, frameFrom.depth() > 0))

    validEdges = np.zeros(frameFrom.boundaries().shape, np.float64)
    validEdges[validV, validU] = 1

    transform = np.dot(cameraTo.cameraMatrix(), np.dot(frameTo.R(), frameFrom.t()) + frameTo.invT_t())

    frameFromDepth = frameFrom.depth() / cameraFrom.depthScaleFactor()
    #txChannel = validEdges * transform[0]
    #tyChannel = validEdges * transform[1]
    #tzChannel = validEdges * transform[2]
    zChannel = validEdges * frameFromDepth
    q = np.array([validU, validV, np.ones_like(validU)])
    qq = np.array([frameFromDepth[validV, validU], frameFromDepth[validV, validU], frameFromDepth[validV, validU]])
    
    q = q*qq

    #q = q * [frameFromDepth[validV, validU], frameFromDepth[validV, validU], frameFromDepth[validV, validU]]
    q = np.dot(cameraTo.cameraMatrix(), np.dot(frameTo.invT_R(), np.dot(frameFrom.R(), np.dot(np.linalg.inv(cameraFrom.cameraMatrix()), q))))
    Q = q + [np.full_like(validU, transform[0], np.float64), np.full_like(validU, transform[1], np.float64), np.full_like(validU, transform[2], np.float64)]
    Q = np.divide(Q[:2, :], Q[2, :])
    #q = np.add(q, [txChannel, tyChannel, tzChannel])
    '''
    validV, validU = np.where(np.bitwise_and(frameFrom.boundaries() > 0, frameFrom.depth() > 0))
    validZ = frameFrom.depth()[validV, validU] / cameraFrom.depthScaleFactor()
   
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d') 
    # ax.scatter(tmpP[0], tmpP[1], tmpP[2])
    # plt.show()

    ## working
    # start = time.time()
    # validUZ = validU * validZ
    # validVZ = validV * validZ
    # tmpP = np.array([validUZ, validVZ, validZ], np.float64)
    # CqInvTq = np.dot(cameraTo.cameraMatrix(), frameTo.invT_t())
    # CqInvRq = np.dot(cameraTo.cameraMatrix(), frameTo.invT_R())
    # CqInvRqRpInvCp = np.dot(CqInvRq, np.dot(frameFrom.R(), cameraFrom.invCameraMatrix()))
    # CqInvRqTp = np.dot(CqInvRq, frameTo.t())
    # CqInvTqCqInvRqTp = CqInvRqTp + CqInvTq
    # q_hat = np.dot(CqInvRqRpInvCp, tmpP)
    # q_hat = np.add(q_hat, np.full_like(q_hat, [np.full_like(validUZ, CqInvTqCqInvRqTp[0], np.float64), np.full_like(validUZ, CqInvTqCqInvRqTp[1], np.float64), np.full_like(validUZ, CqInvTqCqInvRqTp[2], np.float64)], np.float64))
    # Q = np.divide(q_hat[:2, :], q_hat[2, :])
    # QX = Q[0]
    # QY = Q[1]
    # version1 = time.time() - start
    ## working end

    ## working
    # q_hat = Cq * Tq * invTp⁻¹ * Cp⁻¹_hat * p_hat
    # start = time.time()
    tmpP = np.array([validU * validZ, validV * validZ, validZ], np.float64)
    C_hat = np.vstack([cameraFrom.invCameraMatrix(), [0,0,1]])
    invTC_hat = np.vstack([np.dot(frameFrom.T(), C_hat), [0,0,1]])
    q_hat = np.dot(np.dot(cameraTo.cameraMatrix(), np.dot(frameTo.invT(), invTC_hat)), tmpP)
    Q = np.divide(q_hat[:2, :], q_hat[2, :])
    QX = Q[0]
    QY = Q[1]
    # validIndices = np.where(np.logical_and(np.logical_and(QX >= 0, QX <= frameToW-1), np.logical_and(QY >= 0, QY <= frameToH-1)))[0]
    # version2 = time.time() - start
    ## working end

    ## working
    # project to world
    # start = time.time()
    # PZ = frameFrom.depth()[validV, validU] / cameraFrom.depthScaleFactor()
    # PX = (validU - np.full_like(validU, cameraFrom.cx(), np.float64)) * (PZ / cameraFrom.fx())
    # PY = (validV - np.full_like(validV, cameraFrom.cy(), np.float64)) * (PZ / cameraFrom.fy())

    # # transform
    # A = np.dot(frameFrom.R(), [PX, PY, PZ]) + [np.full_like(PX, frameFrom.t()[0], np.float64), np.full_like(PY, frameFrom.t()[1], np.float64), np.full_like(PZ, frameFrom.t()[2], np.float64)]
    # Q = np.dot(frameTo.invT_R(), A) + [np.full_like(PX, frameTo.invT_t()[0], np.float64), np.full_like(PY, frameTo.invT_t()[1], np.float64), np.full_like(PZ, frameTo.invT_t()[2], np.float64)]
    # #Q = np.dot(frameTo.invT_R(), A) + frameTo.invT_t()
    # # rescale by dividing the z coordinate
    # Q = np.divide(Q[:2, :], Q[2, :])
    # # reproject to image plane
    # QX = Q[0] * cameraTo.fx() + np.full_like(PX, cameraTo.cx(), np.float64)
    # QY = Q[1] * cameraTo.fy() + np.full_like(PY, cameraTo.cy(), np.float64)
    # version3 = time.time() - start
    # working end

    ## version2 fastest
    # print('%.6f %.6f %.6f'%(version1, version2, version3))

    validIndices = np.where(np.logical_and(np.logical_and(QX >= 0, QX <= frameToW-1), np.logical_and(QY >= 0, QY <= frameToH-1)))[0]
    # working
    # qy = np.floor(QY[validIndices]).astype(np.int64)
    # qx = np.floor(QX[validIndices]).astype(np.int64)
    # working end
   
    #validIndices = np.where(np.bitwise_or(invalidQX, invalidQY))
    # distVal = frameToDistanceTransform[np.ix_(qy, qx)]

    # fig = plt.figure(3)
    # plt.imshow(distVal)
    # plt.show()

    # print('asdfasdfasdf')

    # reprojectedEdges[np.where(distVal <= edgeDistanceLowerBoundary), 0] = 1
    # reprojectedEdges[np.logical_and(np.where(distVal > edgeDistanceLowerBoundary), np.where(distVal <= edgeDistanceUpperBoundary)), 1] = 1
    # reprojectedEdges[np.where(distVal > edgeDistanceUpperBoundary), 2] = 1

    for i in range(0, len(validIndices)):
        index = validIndices[i]
        qx = QX.item(index)
        qy = QY.item(index)
        u = validU.item(index)
        v = validV.item(index)

        #distVal = frameToDistanceTransform.item(int(qy), int(qx))
        distVal = getInterpolatedElement(frameToDistanceTransform, qx, qy)

        if distVal <= edgeDistanceLowerBoundary:
            reprojectedEdges.itemset((v, u, 0), reprojectedEdges.item((v, u, 0)) + 1)
        elif distVal > edgeDistanceLowerBoundary and distVal <= edgeDistanceUpperBoundary:
            reprojectedEdges.itemset((v, u, 1), reprojectedEdges.item((v, u, 1)) + 1)
        elif distVal > edgeDistanceUpperBoundary:
            reprojectedEdges.itemset((v, u, 2), reprojectedEdges.item((v, u, 2)) + 1)
            ## draw matches
            # matchIndex = len(keyPointsFrameFrom)
            # keyPointsFrameFrom.append(cv.KeyPoint(u, v, 1))
            # keyPointsFrameTo.append(cv.KeyPoint(int(q[0]), int(q[1]), 1))

            # if matchIndex % 200 == 0:
            #     matches.append(cv.DMatch(matchIndex, matchIndex, 1))
            ## end of draw matches

    if result is not None:
        result[frameTo.uid] = reprojectedEdges

    # best, good, worse = cv.split(reprojectedEdges)
    # overlayRgb = cv.cvtColor(frameFrom.rgb(), cv.COLOR_BGR2RGB)
    # overlayRgb[np.where(best > 0)]=[0, 255, 0]
    # overlayRgb[np.where(good > 0)]=[255, 255, 0]
    # overlayRgb[np.where(worse > 0)]=[255, 0, 0]
    # scaledFrameToRgb = frameTo.rgb().copy()
    # scaledFrameToBoundaries = frameTo.boundaries().copy()
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
    # plt.title('%s'%(frameFrom.uid))
    # plt.imshow(rgbDepthEdges)
    # plt.subplot(212)
    # plt.axis('off')
    # plt.title('Matches %s -> %s'%(frameFrom.uid, frameTo.uid))
    # plt.imshow(matchesImg)
    # plt.show()

    return reprojectedEdges
