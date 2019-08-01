import os
import time
import logging
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from typing import List
from enum import IntEnum
from .Camera import Camera
from .Frame import Frame
from . import ImageProcessing
from . import Canny


class EdgeMatcherMode(IntEnum):
    '''
    These modes are used to switch between the re-/backprojection of the edges to a defined frame.
    '''
    # back to front projection
    REPROJECT = 1
    # front to back projetion
    BACKPROJECT = 2
    # back to front and front to back projection to a center frame (re- and backproject)
    CENTERPROJECT = 3


class EdgeMatcher:
    '''
    This class implements various edge matching algorithm based on re- or backprojection of
    edge from differenent frames to a defined one.
    '''

    def __init__(self, camera: Camera = None):
        '''
        Constructor.
        '''
        if camera is None or type(camera) is not Camera:
            raise ValueError('Invalid camera.')

        self.__frameSet: List[Frame] = []
        self.__edgeDistanceLowerBoundary = 1
        self.__edgeDistanceUpperBoundary = 10
        self.__frameOffset = 1
        self.__camera = camera

    def setEdgeDistanceBoundaries(self, lowerBoundary: float = None, upperBoundary: float = None) -> None:
        '''
        During processing the distance of the reprojected edge is compared to the reference boundary. Below
        this boundary are defined as the best edges.

        edgeDistanceLowerBoundary Below this boundary are the best edges.
        '''
        if lowerBoundary is None:
            raise ValueError('Invalid edge distance lower boundary value.')

        if lowerBoundary < 0:
            lowerBoundary = abs(lowerBoundary)

        if upperBoundary is None:
            raise ValueError('Invalid edge distance upper boundary value.')

        if upperBoundary < 0:
            upperBoundary = abs(upperBoundary)

        if upperBoundary < lowerBoundary:
            raise ValueError('Upper boundary must be greater than the lower boundary.')

        self.__edgeDistanceLowerBoundary = lowerBoundary
        self.__edgeDistanceUpperBoundary = upperBoundary

    def setFrameOffset(self, frameOffset: int = None) -> None:
        '''
        Set the offset between the matched frames.

        frameOffset Offset that is used for edge matching. Distance between frames.
        '''
        if frameOffset is None:
            raise ValueError('Invalid frame offset.')

        if frameOffset == 0 or frameOffset > 255:
            raise ValueError('Frame offset must be greater than 0 and smaller than 256.')

        if frameOffset < 0:
            frameOffset = abs(frameOffset)

        self.__frameOffset = frameOffset

    def reprojectEdgesByAscendingCannyThreshold(self,
                                                frame: Frame = None,
                                                stepRange: int = 50,
                                                validEdgesThreshold: float = 0.5,
                                                cannyThres1: int = 0,
                                                cannyThres2: int = 0,
                                                cannyKernelSize: int = 3,
                                                cannyHighAccuracy: bool = True,
                                                cannyBlurKernelSize: int = 3) -> np.ndarray:
        '''
        Reproject edges of varying Canny detected edges. The threshold is dynamically set.

        stepRange Increasing step range for the second threshold.

        validEdgesThreshold Threshold that defines valid edges. Must be between 0 and 1.

        cannyThreshold1 First threshold for the hysteresis procedure.

        cannyThreshold2 Second threshold for the hysteresis procedure.

        cannyKernelSize Kernel size for the sobel operator.

        cannyHighAccuracy If true, L2 gradient will be used for more accuracy.

        cannyBlurKernelSize Kernel size for the Sobel operator.

        Returns meaningful edges.
        '''
        if frame is None or not frame.isValid():
            raise ValueError('Invalid frame.')

        # fill frame set
        self.__frameSet.append(frame)

        if len(self.__frameSet) <= self.__frameOffset:
            return None

        h, w = frame.mask.shape
        meaningfulEdges = np.zeros((h, w, 3))
        distTransMat = cv.distanceTransform(255 - frame.mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)

        try:
            for i in range(0, self.__frameOffset):
                start = time.time()
                mask = Canny.cannyAscendingThreshold(self.__frameSet[i].rgb,
                                                     cannyThres1,
                                                     cannyThres2,
                                                     cannyKernelSize,
                                                     cannyHighAccuracy,
                                                     cannyBlurKernelSize,
                                                     stepRange,
                                                     validEdgesThreshold)

                tmpFrame = self.__frameSet[i]
                tmpFrame.mask = mask

                projectedEdges = self.projectEdges(tmpFrame, frame, distTransMat)
                logging.info('Projected %d in %f sec.' % (i, time.time() - start))

                # accumulate values
                meaningfulEdges = cv.add(meaningfulEdges, projectedEdges)

            best, good, worse = cv.split(meaningfulEdges)
            meaningfulEdges = cv.add(best, good)
            # remove values less the n times seen
            minNumOfFramesDetected = self.__frameOffset * 0.25

            _, meaningfulEdges = cv.threshold(meaningfulEdges, minNumOfFramesDetected, 255, cv.THRESH_BINARY)

        except Exception as e:
            raise e

        self.__frameSet.pop(0)

        return meaningfulEdges

    def reprojectEdgesToConsecutiveFrameSet(self, frame: Frame = None, mode: EdgeMatcherMode = EdgeMatcherMode.REPROJECT, outputDir: str = None) -> np.ndarray:
        '''
        Reproject edges from a destination frame based on the frame offset to a set of consecutive frames.

        frame Current frame information.

        Returns meaningful edges. Channel 1: below or equal lower boundary,
        channel 2: between lower and upper boundary, channel 3: above upper boundary and
        channel 4 distance between mask and reprojected point value.

        Returns meaningful edges.
        '''
        if not frame.isValid():
            raise ValueError('Invalid frame.')

        # fill frame set
        self.__frameSet.append(frame)

        maxFrameOffset = self.__frameOffset

        if mode == EdgeMatcherMode.CENTERPROJECT:
            maxFrameOffset = 2 * self.__frameOffset

        if len(self.__frameSet) <= maxFrameOffset:
            return None

        if mode == EdgeMatcherMode.BACKPROJECT or mode == EdgeMatcherMode.CENTERPROJECT:
            destFrame = self.__frameSet[self.__frameOffset]
        elif mode == EdgeMatcherMode.REPROJECT:
            destFrame = self.__frameSet[0]

        h, w = destFrame.mask.shape
        meaningfulEdges = np.zeros((h, w, 3))
        maxFrameOffset = len(self.__frameSet)

        for i in range(0, maxFrameOffset):
            # skip own projection
            if mode == EdgeMatcherMode.BACKPROJECT or mode == EdgeMatcherMode.CENTERPROJECT:
                if i == self.__frameOffset:
                    continue
            elif mode == EdgeMatcherMode.REPROJECT:
                if i == 0:
                    continue

            start = time.time()

            projectedEdges = self.projectEdges(destFrame, self.__frameSet[i])
            logging.info('Projected %d in %f sec.' % (i, time.time() - start))

            # accumulate values
            meaningfulEdges = cv.add(meaningfulEdges, projectedEdges)

        best, good, worse = cv.split(meaningfulEdges)
        meaningfulEdges = cv.add(best, good)

        print('MaxVal', np.amax(meaningfulEdges))
        # remove values less the n times seen
        minNumOfFramesDetected = 0# maxFrameOffset * 0.25
        _, meaningfulEdges = cv.threshold(meaningfulEdges, minNumOfFramesDetected, 255, cv.THRESH_BINARY)

        # PLOTTING
        if outputDir is not None:
            rgbImg = cv.cvtColor(destFrame.rgb.copy(), cv.COLOR_BGR2BGRA)
            depthImg = cv.cvtColor(ImageProcessing.createHeatmap(destFrame.depth.copy()), cv.COLOR_BGR2RGBA)
            #depthImg[np.where((depthImg==[0,0,0,255]).all(axis=2))] = [255,255,255,255]
            combined = cv.addWeighted(rgbImg, 0.7, depthImg, 0.3, 0)
            combined[np.nonzero(destFrame.mask)] = [255, 255, 255, 255]
            combined[np.nonzero(best)] = [0, 255, 0, 255]
            combined[np.nonzero(good)] = [0, 255, 255, 255]
            combined[np.nonzero(worse)] = [0, 0, 255, 255]

            fig = plt.figure(1)
            # plt.subplot(131)
            # plt.axis('off')
            # plt.imshow(rgbImg)
            # plt.subplot(132)
            # plt.axis('off')
            # plt.imshow(depthImg)
            plt.subplot(111)
            plt.axis('off')
            plt.imshow(cv.cvtColor(combined, cv.COLOR_BGRA2RGBA))
            # plt.show()

            if not os.path.exists(os.path.join(outputDir, 'plots')):
                os.makedirs(os.path.join(outputDir, 'plots'))

            fig.savefig(os.path.join(outputDir, 'plots', ('%f.svg' % (time.time()))), dpi=300)
        # END OF PLOTTING

        # remove first frame from frame set
        self.__frameSet.pop(0)

        return meaningfulEdges

    def reprojectEdgesFromConsecutiveFrameSet(self, frame: Frame = None, mode: EdgeMatcherMode = EdgeMatcherMode.REPROJECT, outputDir: str = None) -> np.ndarray:
        '''
        Reproject edges from consectutive frames to a destination frame based on the frame offset.

        frame Current frame information.

        Returns meaningful edges. Channel 1: below or equal lower boundary,
        channel 2: between lower and upper boundary, channel 3: above upper boundary and
        channel 4 distance between mask and reprojected point value.

        Returns meaningful edges.
        '''
        if not frame.isValid():
            raise ValueError('Invalid frame.')

        # fill frame set
        self.__frameSet.append(frame)

        maxFrameOffset = self.__frameOffset

        if mode == EdgeMatcherMode.CENTERPROJECT:
            maxFrameOffset = 2 * self.__frameOffset

        if len(self.__frameSet) <= maxFrameOffset:
            return None

        if mode == EdgeMatcherMode.REPROJECT:
            destFrame = frame
        elif mode == EdgeMatcherMode.BACKPROJECT:
            destFrame = self.__frameSet[0]
        elif mode == EdgeMatcherMode.CENTERPROJECT:
            destFrame = self.__frameSet[self.__frameOffset]
            maxFrameOffset = maxFrameOffset + 1
        else:
            raise ValueError('Unsupported projection mode.')

        h, w = destFrame.mask.shape
        meaningfulEdges = np.zeros((h, w, 3))
        distTransMat = cv.distanceTransform(255 - destFrame.mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)

        for i in range(0, maxFrameOffset):
            if mode == EdgeMatcherMode.CENTERPROJECT:
                # avoid destination frame projection
                if i == self.__frameOffset:
                    continue

            start = time.time()

            projectedEdges = self.projectEdges(self.__frameSet[i], destFrame, distTransMat, True)
            logging.info('Projected %d in %f sec.' % (i, time.time() - start))

            # accumulate values
            meaningfulEdges = cv.add(meaningfulEdges, projectedEdges)

        best, good, worse = cv.split(meaningfulEdges)
        meaningfulEdges = cv.add(best, good)

        print('MaxVal', np.amax(meaningfulEdges))
        # remove values less the n times seen
        minNumOfFramesDetected = 0#maxFrameOffset * 0.25
        _, meaningfulEdges = cv.threshold(meaningfulEdges, minNumOfFramesDetected, 255, cv.THRESH_BINARY)

        # PLOTTING
        if outputDir is not None:
            rgbImg = cv.cvtColor(destFrame.rgb.copy(), cv.COLOR_BGR2BGRA)
            depthImg = cv.cvtColor(ImageProcessing.createHeatmap(destFrame.depth.copy()), cv.COLOR_BGR2RGBA)
            #depthImg[np.where((depthImg==[0,0,0,255]).all(axis=2))] = [255,255,255,255]
            combined = cv.addWeighted(rgbImg, 0.7, depthImg, 0.3, 0)
            combined[np.nonzero(destFrame.mask)] = [255, 255, 255, 255]
            combined[np.nonzero(best)] = [0, 255, 0, 255]
            combined[np.nonzero(good)] = [0, 255, 255, 255]
            combined[np.nonzero(worse)] = [0, 0, 255, 255]

            fig = plt.figure(1)
            # plt.subplot(131)
            # plt.axis('off')
            # plt.imshow(rgbImg)
            # plt.subplot(132)
            # plt.axis('off')
            # plt.imshow(depthImg)
            plt.subplot(111)
            plt.axis('off')
            plt.imshow(cv.cvtColor(combined, cv.COLOR_BGRA2RGBA))
            # plt.show()

            if not os.path.exists(os.path.join(outputDir, 'plots')):
                os.makedirs(os.path.join(outputDir, 'plots'))

            fig.savefig(os.path.join(outputDir, 'plots', ('%f.svg' % (time.time()))), dpi=300)
        # END OF PLOTTING

        # remove first frame from frame set
        self.__frameSet.pop(0)

        return meaningfulEdges

    def projectEdges(self, frameFrom: Frame = None, frameTo: Frame = None, distTransMat: np.ndarray = None, takeValuesFromDistTrans: bool = False) -> np.ndarray:
        '''
        Project edges from one frame to another.

        frameFrom Frame that should be projected to.

        frameTo Frame on that is projected.

        distTransMat Distance matrix that contains the distance transform values.

        takeValuesFromDistTrans Take the distance from the reprojected coordinates.

        Returns matrix that contains the result of the reprojection.
        1 channel: best with distance <= edgeDistanceLowerBoundary
        2 channel: good with edgeDistanceLowerBoundary < distance <= edgeDistanceUpperBoundary
        3 channel: worse with distance > edgeDistanceUpperBoundary
        '''
        if not frameFrom.isValid():
            raise ValueError('Invalid frame from.')

        if not frameTo.isValid():
            raise ValueError('Invalid frame to.')

        if self.__camera.depthScaleFactor() == 0:
            raise ValueError('Invalid depth scale factor.')

        if distTransMat is None:
            distTransMat = cv.distanceTransform(255 - frameTo.mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)

        h, w = frameFrom.mask.shape
        reprojectedEdges = np.zeros((h, w, 3))

        for u in range(0, w):
            for v in range(0, h):

                if frameFrom.mask.item(v, u) == 0:
                    continue

                if frameFrom.depth.item(v, u) == 0:
                    continue

                z = np.float64(frameFrom.depth.item(v, u)) / np.float64(self.__camera.depthScaleFactor())

                if z == 0:
                    continue

                # project to world coordinate system
                X = (u - self.__camera.cx()) * z / self.__camera.fx()
                Y = (v - self.__camera.cy()) * z / self.__camera.fy()
                # rotate, translate to other frame
                p1 = frameFrom.T.item((0, 0)) * X + frameFrom.T.item((0, 1)) * Y + frameFrom.T.item((0, 2)) * z + frameFrom.T.item((0, 3))
                p2 = frameFrom.T.item((1, 0)) * X + frameFrom.T.item((1, 1)) * Y + frameFrom.T.item((1, 2)) * z + frameFrom.T.item((1, 3))
                p3 = frameFrom.T.item((2, 0)) * X + frameFrom.T.item((2, 1)) * Y + frameFrom.T.item((2, 2)) * z + frameFrom.T.item((2, 3))
                q1 = frameTo.invT().item((0, 0)) * p1 + frameTo.invT().item((0, 1)) * p2 + frameTo.invT().item((0, 2)) * p3 + frameTo.invT().item((0, 3))
                q2 = frameTo.invT().item((1, 0)) * p1 + frameTo.invT().item((1, 1)) * p2 + frameTo.invT().item((1, 2)) * p3 + frameTo.invT().item((1, 3))
                q3 = frameTo.invT().item((2, 0)) * p1 + frameTo.invT().item((2, 1)) * p2 + frameTo.invT().item((2, 2)) * p3 + frameTo.invT().item((2, 3))
                # project found 3d point Q back to the image plane
                U = (q1 / q3) * self.__camera.fx() + self.__camera.cx()
                V = (q2 / q3) * self.__camera.fy() + self.__camera.cy()
                # boundary check
                if U < 0 or V < 0 or U > (w-1) or V > (h-1):
                    continue

                distVal = ImageProcessing.getInterpolatedElement(distTransMat, U, V)
           
                poi = np.array([u, v])

                if takeValuesFromDistTrans:
                    poi = np.array([int(U), int(V) ])

                if distVal <= self.__edgeDistanceLowerBoundary:
                    reprojectedEdges.itemset((poi[1], poi[0], 0), reprojectedEdges.item((poi[1], poi[0], 0)) + 1)
                elif distVal > self.__edgeDistanceLowerBoundary and distVal <= self.__edgeDistanceUpperBoundary:
                    reprojectedEdges.itemset((poi[1], poi[0], 1), reprojectedEdges.item((poi[1], poi[0], 1)) + 1)
                elif distVal > self.__edgeDistanceUpperBoundary:
                    reprojectedEdges.itemset((poi[1], poi[0], 2), reprojectedEdges.item((poi[1], poi[0], 2)) + 1)

        return reprojectedEdges

    def __transformPoint(self, frameFrom: Frame = None, frameTo: Frame = None, point3d: np.ndarray = None) -> np.ndarray:
        '''
        Transform position and rotation from on frame position to another.

        frameFrom Frame from.

        frameTo Frame to.

        xyzCoords Point 3D that should be projected from frameFrom to frameTo.
        '''
        if not frameFrom.isValid():
            raise ValueError('Invalid frame from.')

        if not frameTo.isValid():
            raise ValueError('Invalid frame to.')

        if len(point3d) < 3:
            raise ValueError('Invalid point.')

        # rotate, translate point P = [x, y, z] -> uvCoords = frameToInvT * (frameFromT * P))
        p1 = frameFrom.T.item((0, 0)) * point3d[0] + frameFrom.T.item((0, 1)) * point3d[1] + frameFrom.T.item((0, 2)) * point3d[2] + frameFrom.T.item((0, 3))
        p2 = frameFrom.T.item((1, 0)) * point3d[0] + frameFrom.T.item((1, 1)) * point3d[1] + frameFrom.T.item((1, 2)) * point3d[2] + frameFrom.T.item((1, 3))
        p3 = frameFrom.T.item((2, 0)) * point3d[0] + frameFrom.T.item((2, 1)) * point3d[1] + frameFrom.T.item((2, 2)) * point3d[2] + frameFrom.T.item((2, 3))
        q1 = frameTo.invT().item((0, 0)) * p1 + frameTo.invT().item((0, 1)) * p2 + frameTo.invT().item((0, 2)) * p3 + frameTo.invT().item((0, 3))
        q2 = frameTo.invT().item((1, 0)) * p1 + frameTo.invT().item((1, 1)) * p2 + frameTo.invT().item((1, 2)) * p3 + frameTo.invT().item((1, 3))
        q3 = frameTo.invT().item((2, 0)) * p1 + frameTo.invT().item((2, 1)) * p2 + frameTo.invT().item((2, 2)) * p3 + frameTo.invT().item((2, 3))

        # lower performance
        # return np.array([q1, q2, q3], np.float64)
        # point3d = np.array(([point3d[0]],[point3d[1]],[point3d[2]]), np.float64)
        # return np.dot(frameTo.invT_R(), np.dot(frameFrom.R(), point3d) + frameFrom.t()) + frameTo.invT_t()
        # point3d = np.array(([point3d[0]],[point3d[1]],[point3d[2]], [1.0]), np.float64)
        # return np.dot(frameTo.invT(), np.dot(frameFrom.T, point3d))
        return np.array(([q1],[q2],[q3]), np.float64)

