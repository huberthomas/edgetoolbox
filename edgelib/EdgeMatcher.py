from edgelib.Camera import Camera
from edgelib.Frame import Frame
import time
import numpy as np
import cv2
from edgelib import ImageProcessing
import matplotlib.pyplot as plt
from typing import List


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

    def reprojectEdgesFromConsecutiveFrameSet(self, frame: Frame = None) -> np.ndarray:
        '''
        Reproject edges based on the frame offset.

        frame Current frame information.

        meaningfulEdges Result edges are stored in this structure.
        '''
        if not frame.isValid():
            raise ValueError('Invalid frame.')

        # fill frame set
        self.__frameSet.append(frame)

        if len(self.__frameSet) <= self.__frameOffset:
            return None

        h, w = frame.mask.shape
        meaningfulEdges = np.zeros((h, w), np.uint8)

        for i in range(0, self.__frameOffset):
            start = time.time()
            reprojectedEdges = self.reprojectEdges(self.__frameSet[i], frame)
            print(time.time() - start)
            best, good, worse, d = cv2.split(reprojectedEdges)

            # plt.figure(1)
            # plt.subplot(131)
            # plt.imshow(best)
            # plt.subplot(132)
            # plt.imshow(good)
            # plt.subplot(133)
            # plt.imshow(worse)
            # plt.show()

            # combine best and good
            # meaningfulEdges = cv2.add(meaningfulEdges, reprojectedEdges.item(: , : , 0))
            # meaningfulEdges = cv2.add(meaningfulEdges, reprojectedEdges.item(: , : , 1))

        self.__frameSet.pop(0)

        # remove values less the n times seen
        # minNumOfFramesDetected = self.__frameOffset * 0.25
        # detectedInFrames = cv2.threshold(detectedInFrames, minNumOfFramesDetected, 255, cv2.CV_8UC1)

        return meaningfulEdges

    def reprojectEdges(self, frameFrom: Frame = None, frameTo: Frame = None) -> np.ndarray:
        '''
        Reproject edges from one frame to another.

        frameFrom Frame that should be projected to.

        frameTo Frame on that is projected.

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

        if self.__camera.depthScaleFactor == 0:
            raise ValueError('Invalid depth scale factor.')

        dist = cv2.distanceTransform(255 - frameTo.mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        invFrameToT = frameTo.T.inv()

        h, w = frameFrom.mask.shape
        reprojectedEdges = np.zeros((h, w, 4))

        for u in range(0, w):
            for v in range(0, h):

                reprojectedEdges.itemset((v, u, 3), -1)

                if frameFrom.mask.item(v, u) == 0:
                    continue

                z = frameFrom.depth.item(v, u) / float(self.__camera.depthScaleFactor)

                if z == 0:
                    continue

                # projection
                xyzCoords = ImageProcessing.projectToWorld(self.__camera, np.array([u, v, z]))
                P = np.array([xyzCoords[0], xyzCoords[1], z, 1])
                Q = invFrameToT * (frameFrom.T * P)
                uvCoords = ImageProcessing.projectToImage(self.__camera, np.array([Q[0], Q[1], Q[2]]))

                # boundary check
                if uvCoords[0] < 0 or uvCoords[1] < 0 or uvCoords[0] >= w or uvCoords[1] >= h:
                    continue

                distVal = ImageProcessing.getInterpolatedElement(dist, uvCoords[0], uvCoords[1], w)

                reprojectedEdges.itemset((v, u, 3), distVal)

                if distVal <= self.__edgeDistanceLowerBoundary:
                    reprojectedEdges.itemset((v, u, 0), reprojectedEdges.item((v, u, 0)) + 1)
                elif distVal > self.__edgeDistanceLowerBoundary and distVal <= self.__edgeDistanceUpperBoundary:
                    reprojectedEdges.itemset((v, u, 1), reprojectedEdges.item((v, u, 1)) + 1)
                elif distVal > self.__edgeDistanceUpperBoundary:
                    reprojectedEdges.itemset((v, u, 2), reprojectedEdges.item((v, u, 2)) + 1)

        return reprojectedEdges
