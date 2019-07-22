from edgelib.Camera import Camera
from edgelib.Frame import Frame
import time
import numpy as np
import cv2 as cv
from edgelib import ImageProcessing
import matplotlib.pyplot as plt
from typing import List
import logging
from enum import IntEnum

class EdgeMatcherMode(IntEnum):
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

    def reprojectEdgesFromConsecutiveFrameSet(self, frame: Frame = None, mode: EdgeMatcherMode = EdgeMatcherMode.REPROJECT) -> np.ndarray:
        '''
        Reproject edges based on the frame offset.

        frame Current frame information.

        Returns meaningful edges. Channel 1: below or equal lower boundary, 
        channel 2: between lower and upper boundary, channel 3: above upper boundary and
        channel 4 distance between mask and reprojected point value.
        '''
        if not frame.isValid():
            raise ValueError('Invalid frame.')

        # fill frame set
        self.__frameSet.append(frame)

        maxFrameOffset = self.__frameOffset

        if mode == EdgeMatcherMode.CENTERPROJECT:
            maxFrameOffset = 2 * self.__frameOffset
        
        if len(self.__frameSet) < (maxFrameOffset + 1):
            return None

        if mode == EdgeMatcherMode.REPROJECT:
            destFrame = frame
        elif mode == EdgeMatcherMode.BACKPROJECT:
            destFrame = self.__frameSet[0]
        elif mode == EdgeMatcherMode.CENTERPROJECT:
            destFrame = self.__frameSet[self.__frameOffset]
        else:
            raise ValueError('Unsupported projection mode.')


        h, w = destFrame.mask.shape
        meaningfulEdges = np.zeros((h, w, 4))
        distTransMat = cv.distanceTransform(255 - destFrame.mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)

        for i in range(0, maxFrameOffset):
            if mode == EdgeMatcherMode.CENTERPROJECT:
                # avoid destination frame projection
                if i == self.__frameOffset:
                    continue
            
            start = time.time()

            if mode == EdgeMatcherMode.REPROJECT or (mode == EdgeMatcherMode.CENTERPROJECT and i < self.__frameOffset):
                projectedEdges = self.projectEdges(self.__frameSet[i], destFrame, distTransMat)
                logging.info('Reprojected %d in %f sec.' % (i, time.time() - start))
            elif mode == EdgeMatcherMode.BACKPROJECT or (mode == EdgeMatcherMode.CENTERPROJECT and i > self.__frameOffset):
                projectedEdges = self.projectEdges(self.__frameSet[maxFrameOffset - i - 1], destFrame, distTransMat)
                logging.info('Backprojected %d in %f sec.' % (i, time.time() - start))

            # accumulate values
            meaningfulEdges = cv.add(meaningfulEdges, projectedEdges)

        best, good, worse, _ = cv.split(meaningfulEdges)
        meaningfulEdges = cv.add(best, good)

        # remove values less the n times seen
        minNumOfFramesDetected = self.__frameOffset * 0.25
        _, meaningfulEdges = cv.threshold(meaningfulEdges, minNumOfFramesDetected, 255, cv.THRESH_BINARY)

        # plt.figure(1)
        # plt.subplot(221)
        # plt.imshow(best)
        # plt.subplot(222)
        # plt.imshow(good)
        # plt.subplot(223)
        # plt.imshow(worse)
        # plt.subplot(224)
        # plt.imshow(meaningfulEdges)
        # plt.show()

        # remove first frame from frame set
        self.__frameSet.pop(0)

        return meaningfulEdges

    def projectEdges(self, frameFrom: Frame = None, frameTo: Frame = None, distTransMat: np.ndarray = None) -> np.ndarray:
        '''
        Project edges from one frame to another.

        frameFrom Frame that should be projected to.

        frameTo Frame on that is projected.

        distTransMat Distance matrix that contains the distance transform values.

        Returns matrix that contains the result of the reprojection. 
        1 channel: best with distance <= edgeDistanceLowerBoundary
        2 channel: good with edgeDistanceLowerBoundary < distance <= edgeDistanceUpperBoundary
        3 channel: worse with distance > edgeDistanceUpperBoundary
        4 channel: distance value otherwise -1, e.g. on undefined depth values
        '''
        if not frameFrom.isValid():
            raise ValueError('Invalid frame from.')

        if not frameTo.isValid():
            raise ValueError('Invalid from to.')

        if self.__camera.depthScaleFactor() == 0:
            raise ValueError('Invalid depth scale factor.')

        if distTransMat is None:
            distTransMat = cv.distanceTransform(255 - frameTo.mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)

        h, w = frameFrom.mask.shape
        reprojectedEdges = np.zeros((h, w, 3), np.float64)
        distChannel = np.full((h, w), -1, np.float64)
        reprojectedEdges = np.dstack((reprojectedEdges, distChannel))

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
                xyzCoords = ImageProcessing.projectToWorld(self.__camera, np.array([u, v, z], np.float64))

                # rotate, translate point P = [x, y, z, 1] -> uvCoords = frameToInvT * (frameFromT * P))
                p1 = frameFrom.T.item((0, 0)) * xyzCoords[0] + frameFrom.T.item((0, 1)) * xyzCoords[1] + frameFrom.T.item((0, 2)) * z + frameFrom.T.item((0, 3))
                p2 = frameFrom.T.item((1, 0)) * xyzCoords[0] + frameFrom.T.item((1, 1)) * xyzCoords[1] + frameFrom.T.item((1, 2)) * z + frameFrom.T.item((1, 3))
                p3 = frameFrom.T.item((2, 0)) * xyzCoords[0] + frameFrom.T.item((2, 1)) * xyzCoords[1] + frameFrom.T.item((2, 2)) * z + frameFrom.T.item((2, 3))
                q1 = frameTo.invT().item((0, 0)) * p1 + frameTo.invT().item((0, 1)) * p2 + frameTo.invT().item((0, 2)) * p3 + frameTo.invT().item((0, 3))
                q2 = frameTo.invT().item((1, 0)) * p1 + frameTo.invT().item((1, 1)) * p2 + frameTo.invT().item((1, 2)) * p3 + frameTo.invT().item((1, 3))
                q3 = frameTo.invT().item((2, 0)) * p1 + frameTo.invT().item((2, 1)) * p2 + frameTo.invT().item((2, 2)) * p3 + frameTo.invT().item((2, 3))

                # project found 3d point Q back to the image plane
                uvCoords = ImageProcessing.projectToImage(self.__camera, np.array([q1, q2, q3], np.float64))

                # boundary check
                if uvCoords[0] < 0 or uvCoords[1] < 0 or uvCoords[0] > (w-1) or uvCoords[1] > (h-1):
                    continue

                distVal = ImageProcessing.getInterpolatedElement(distTransMat, uvCoords[0], uvCoords[1])

                iu = int(uvCoords[0])
                iv = int(uvCoords[1])

                if distVal <= self.__edgeDistanceLowerBoundary:
                    reprojectedEdges.itemset((iv, iu, 0), reprojectedEdges.item((iv, iu, 0)) + 1)
                elif distVal > self.__edgeDistanceLowerBoundary and distVal <= self.__edgeDistanceUpperBoundary:
                    reprojectedEdges.itemset((iv, iu, 1), reprojectedEdges.item((iv, iu, 1)) + 1)
                elif distVal > self.__edgeDistanceUpperBoundary:
                    reprojectedEdges.itemset((iv, iu, 2), reprojectedEdges.item((iv, iu, 2)) + 1)

                # store distance value in 4th channel
                reprojectedEdges.itemset((iv, iu, 3), distVal)

        return reprojectedEdges
