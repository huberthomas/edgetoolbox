import os
import time
import logging
import numpy as np
import cv2 as cv
import multiprocessing as mp
from multiprocessing import Manager
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List
from enum import IntEnum
from scipy import signal
from .EdgeMatcherFrame import EdgeMatcherFrame
import copy
from .Camera import Camera
from .Frame import Frame
from . import Utilities
from . import ImageProcessing


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
        self.__processedFrameSet: List[Frame] = []
        self.__edgeDistanceLowerBoundary = 1
        self.__edgeDistanceUpperBoundary = 10
        self.__frameOffset = 1
        self.__camera = camera
        self.__numOfThreads = mp.cpu_count()
        self.__minIsolatedPixelArea = 0

    def setMinIsolatedPixelArea(self, minIsolatedPixelArea: int = None) -> None:
        '''
        Set the minimum isolated pixel area. During the process isolated pixel areas 
        less than equal this value are removed.

        minIsolatedPixelArea Minimum isolated pixel areas beyond this value are removed.
        '''
        if minIsolatedPixelArea is None:
            raise ValueError('Invalid minimum isolated pixel area.')

        if minIsolatedPixelArea < 0:
            minIsolatedPixelArea = abs(minIsolatedPixelArea)

        self.__minIsolatedPixelArea = minIsolatedPixelArea

    def minIsolatedPixelArea(self) -> int:
        '''
        Returns the minimum isolated pixel area. During the process isolated pixel areas 
        less than equal this value are removed.
        '''
        return self.__minIsolatedPixelArea

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
                                                frame: EdgeMatcherFrame = None,
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

        h, w = frame.boundaries().shape
        meaningfulEdges = np.zeros((h, w, 3))
        distTransMat = frame.distanceTransform()

        try:
            for i in range(0, self.__frameOffset):
                start = time.time()
                mask = ImageProcessing.cannyAscendingThreshold(self.__frameSet[i].rgb(),
                                                     cannyThres1,
                                                     cannyThres2,
                                                     cannyKernelSize,
                                                     cannyHighAccuracy,
                                                     cannyBlurKernelSize,
                                                     stepRange,
                                                     validEdgesThreshold)

                tmpFrame = self.__frameSet[i]
                tmpFrame.setBoundaries(mask)

                projectedEdges = ImageProcessing.projectEdges(tmpFrame, frame, True, self.__camera, (self.__edgeDistanceLowerBoundary, self.__edgeDistanceUpperBoundary))
                frame.projectedEdgeResults[tmpFrame.uid] = projectedEdges

                logging.info('Projected %d in %f sec.' % (i, time.time() - start))

                # accumulate values
                #meaningfulEdges = cv.add(meaningfulEdges, projectedEdges)

            return frame.getMeaningfulEdges()
            #best, good, worse = cv.split(meaningfulEdges)
            #meaningfulEdges = cv.add(best, good)
            # remove values less the n times seen
            #minNumOfFramesDetected = self.__frameOffset * 0.25

            #_, meaningfulEdges = cv.threshold(meaningfulEdges, minNumOfFramesDetected, 255, cv.THRESH_BINARY)

        except Exception as e:
            raise e

        self.__frameSet.pop(0)

        return meaningfulEdges

    def getProbabilityMap(self, reprojectedEdgesList: List[any] = None) -> np.ndarray:
        '''
        '''
        if reprojectedEdgesList is None or len(reprojectedEdgesList) == 0:
            raise ValueError('Invalid reprojected edges list.')

        scaledMeaningfulEdges = None
        counter = 0

        for scaledResultList in reprojectedEdgesList:
            for scaledResult in scaledResultList:
                counter += 1
                best, good, worse = cv.split(scaledResult)

                if scaledMeaningfulEdges is None:
                    h, w = best.shape[:2]
                    scaledMeaningfulEdges = np.zeros((h, w))

                numBest = np.sum(best)
                numGood = np.sum(good)
                numWorse = np.sum(worse)

                # linear weight function
                weight = (numBest + numGood) / (numBest + numGood + numWorse)
                # sinoid weight function
                #weight = np.sin(Utilities.rescale(weight, 0, 1, 0, np.pi/2.0))
                tmpMeaningfulEdges = cv.add(best, good)
                tmpMeaningfulEdges *= weight
                scaledMeaningfulEdges = cv.add(scaledMeaningfulEdges, tmpMeaningfulEdges)

        if counter > 0:
            scaledMeaningfulEdges /= counter
            return scaledMeaningfulEdges
        else:
            return None


    def reprojectEdgesToConsecutiveFrameSet(self, frame: EdgeMatcherFrame = None,
                                            mode: EdgeMatcherMode = EdgeMatcherMode.REPROJECT,
                                            outputDir: str = None) -> (np.ndarray, np.ndarray):
        '''
        Reproject edges from a destination frame based on the frame offset to a set of consecutive frames.

        frame Current frame information.

        mode Edge matcher mode. Default is reproject, others are backproject and centerproject.

        Returns meaningful edges.
        '''
        if not frame.isValid():
            raise ValueError('Invalid frame.')

        # fill frame set
        self.__frameSet.append(frame)

        maxFrameOffset = self.__frameOffset

        if mode == EdgeMatcherMode.CENTERPROJECT:
            maxFrameOffset = 2 * maxFrameOffset

        if len(self.__frameSet) <= maxFrameOffset:
            return (None, None)

        frameFromIndex = -1

        if mode == EdgeMatcherMode.BACKPROJECT or mode == EdgeMatcherMode.CENTERPROJECT:
            frameFromIndex = self.__frameOffset
        elif mode == EdgeMatcherMode.REPROJECT:
            frameFromIndex = 0
        else:
            raise ValueError('Invalid destination frame index.')

        maxFrameOffset = len(self.__frameSet)
        frameFrom = self.__frameSet[frameFromIndex]
        # PARAMETER SETTING
        scales = [1.0, 0.5, 0.25]
        edgeThresMin = 50
        edgeThresMax = 100
        edgeKernelSize = 3
        blurKernelSize = 5
        # for scale 1 keep the parameters equal to the current frame
        cannyThresholds = [(edgeThresMin, edgeThresMax), (50, 100), (50, 100)]
        cannyKernelSizes = [(edgeKernelSize, blurKernelSize), (3, 3), (3, 3)]
        # todo: optional determine edges
        frameFrom.setBoundaries(ImageProcessing.canny(frameFrom.rgb(), edgeThresMin, edgeThresMax, edgeKernelSize, True, blurKernelSize))

        # SCALED RESPONSES
        reprojectedEdgesList = Manager().list() # needed for multithread pool results

        start = time.time()
        scaledThreadParam = []

        for i in range(0, maxFrameOffset):
            # skip own projection
            if i == frameFromIndex:
                continue

            frameTo = self.__frameSet[i]
            scaledThreadParam.append((frameFrom, frameTo, self.__camera, (self.__edgeDistanceLowerBoundary, self.__edgeDistanceUpperBoundary), scales, cannyThresholds, cannyKernelSizes, reprojectedEdgesList))

        pool = mp.Pool(processes=self.__numOfThreads)
        pool.starmap(ImageProcessing.projectMultiscaleEdges, scaledThreadParam)
        pool.terminate()

        logging.info('Projected %d frames and %d scaled frames in %f sec.' % (len(scaledThreadParam), len(scales), time.time() - start))

        # sum up and weight scaled results
        scaledMeaningfulEdges = self.getProbabilityMap(reprojectedEdgesList)        

        if scaledMeaningfulEdges is None:
            h, w = frameFrom.boundaries().shape[:2]
            scaledMeaningfulEdges = np.zeros((h, w))

        # remove isolated pixel
        if self.__minIsolatedPixelArea > 0:
            _, thresScaledMeaningfulEdges = cv.threshold(scaledMeaningfulEdges, 0, 1, cv.THRESH_BINARY)
            remained, removed = ImageProcessing.removeIsolatedPixels((thresScaledMeaningfulEdges).astype(np.uint8), self.__minIsolatedPixelArea)
            scaledMeaningfulEdges[np.nonzero(removed)] = 0

        frameFrom.scaledMeaningfulEdges = scaledMeaningfulEdges
        # END OF SCALED RESPONSES


        '''
        # refine with only precalculated meaningful edges
        _, scaledMeaningfulEdges = cv.threshold(scaledMeaningfulEdges, 0, 255, cv.THRESH_BINARY)
        
        refinedMeaningfulEdgesList = []
        for i in range(0, maxFrameOffset):
            if i == destFrameIndex:
                continue

            frameTo = self.__frameSet[i]
            
            if frameTo.scaledMeaningfulEdges is None:
                continue

            _, frameToBoundaries = cv.threshold(frameTo.scaledMeaningfulEdges, 0, 255, cv.THRESH_BINARY)
            frameToDistanceTransform = cv.distanceTransform(255 - frameToBoundaries.astype(np.uint8), cv.DIST_L2, cv.DIST_MASK_PRECISE)

            refinedMeaningfulEdges = np.zeros((h, w, 3))
            for u in range(0, w):
                for v in range(0, h):
                    if frameFrom.scaledMeaningfulEdges.item((v, u)) == 0 or frameFrom.depth().item((v, u)) == 0:
                        continue

                    z = np.float64(frameFrom.depth().item((v, u))) / np.float64(self.__camera.depthScaleFactor())
                    # project to world
                    P = ImageProcessing.projectToWorld(self.__camera, np.asarray((u, v, z), np.float64))
                    # transform
                    Q = np.dot(frameTo.invT_R(), np.dot(frameFrom.R(), P) + frameFrom.t()) + frameTo.invT_t()
                    # reproject to image plane
                    q = np.dot(self.__camera.cameraMatrix(), Q)
                    q /= q[2]
                    # boundary check
                    if q[0] < 0 or q[1] < 0 or q[0] > (w - 1) or q[1] > (h - 1):
                        # print(q)
                        continue

                    distVal = ImageProcessing.getInterpolatedElement(frameToDistanceTransform, q[0], q[1])

                    poi = np.array([u, v])

                    if distVal <= self.__edgeDistanceLowerBoundary:
                        refinedMeaningfulEdges.itemset((poi[1], poi[0], 0), refinedMeaningfulEdges.item((poi[1], poi[0], 0)) + 1)
                    elif distVal > self.__edgeDistanceLowerBoundary and distVal <= self.__edgeDistanceUpperBoundary:
                        refinedMeaningfulEdges.itemset((poi[1], poi[0], 1), refinedMeaningfulEdges.item((poi[1], poi[0], 1)) + 1)
                    elif distVal > self.__edgeDistanceUpperBoundary:
                        refinedMeaningfulEdges.itemset((poi[1], poi[0], 2), refinedMeaningfulEdges.item((poi[1], poi[0], 2)) + 1)

            refinedMeaningfulEdgesList.append(refinedMeaningfulEdges)
        
        if len(refinedMeaningfulEdgesList) < self.__frameOffset:
            self.__frameSet.pop(0)
            return (None, None)

        meaningfulEdges = np.zeros((h, w))
        print('Refining %d frames.'%(len(refinedMeaningfulEdgesList)))
        counter = 0
        for refinedMeaningfulEdges in refinedMeaningfulEdgesList:
            best, good, worse = cv.split(refinedMeaningfulEdges)
            numBest = np.sum(best)
            numGood = np.sum(good)
            numWorse = np.sum(worse)

            # linear weight function
            weight = (numBest + numGood) / (numBest + numGood + numWorse)
            # sinoid weight function
            #weight = np.sin(Utilities.rescale(weight, 0, 1, 0, np.pi/2.0))
            tmpMeaningfulEdges = cv.add(best, good)
            tmpMeaningfulEdges *= weight

            fig = plt.figure(counter)
            plt.subplot(211)
            plt.imshow(scaledMeaningfulEdges, cmap='hot')
            plt.subplot(212)
            plt.imshow(tmpMeaningfulEdges, cmap='hot')
            counter += 1

            meaningfulEdges = cv.add(meaningfulEdges, tmpMeaningfulEdges)
            
        fig = plt.figure(len(refinedMeaningfulEdgesList) + 1)
        plt.subplot(211)
        plt.axis('off')
        plt.imshow(frameFrom.scaledMeaningfulEdges, cmap='hot')
        plt.subplot(212)
        plt.axis('off')
        plt.imshow(meaningfulEdges, cmap='hot')
        plt.show()
        '''

        cm_hot = mpl.cm.get_cmap('hot')
        im = cm_hot(frameFrom.scaledMeaningfulEdges)
        im = np.uint8(im * 255)
        im = cv.cvtColor(im, cv.COLOR_BGR2BGRA)
        im[np.where(frameFrom.boundaries() == 0)] = [0, 0, 0, 255]
        rgbResultEdges = cv.addWeighted(cv.cvtColor(frameFrom.rgb(), cv.COLOR_BGR2RGBA), 1, im, 1, 0)
        fig = plt.figure(1)
        plt.axis('off')
        plt.imshow(rgbResultEdges)
        plt.show()

        worseResults = frameFrom.boundaries().copy()
        worseResults[np.nonzero(frameFrom.scaledMeaningfulEdges)] = 0

        self.__frameSet.pop(0)
        return (frameFrom.scaledMeaningfulEdges, worseResults)

    def reprojectEdgesToConsecutiveFrameSet2(self, frame: EdgeMatcherFrame = None, mode: EdgeMatcherMode = EdgeMatcherMode.REPROJECT, outputDir: str = None) -> (np.ndarray, np.ndarray):
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
            return (None, None)

        destFrameIndex = -1

        if mode == EdgeMatcherMode.BACKPROJECT or mode == EdgeMatcherMode.CENTERPROJECT:
            destFrameIndex = self.__frameOffset
        elif mode == EdgeMatcherMode.REPROJECT:
            destFrameIndex = 0
        else:
            raise ValueError('Invalid destination frame index.')

        destFrame = self.__frameSet[destFrameIndex]

        maxFrameOffset = len(self.__frameSet)

        start = time.time()
        # manager needed to share result between threads
        result = Manager().dict()
        param = []

        for i in range(0, maxFrameOffset):
            # skip own projection
            if i == destFrameIndex:
                continue

            param.append((destFrame,
                          self.__frameSet[i],
                          False,
                          self.__camera,
                          (self.__edgeDistanceLowerBoundary, self.__edgeDistanceUpperBoundary),
                          1,
                          result))

        pool = mp.Pool(processes=self.__numOfThreads)
        pool.starmap(ImageProcessing.projectEdges, param)
        pool.terminate()

        logging.info('Projected %d frames in %f sec.' % (maxFrameOffset-1, time.time() - start))
        destFrame.projectedEdgeResults = result
        # destFrame.printProjectedEdgeResults()

        meaningfulEdges = destFrame.getMeaningfulEdges()
        ### REFINE with image pyramids
        # result = Manager().dict()
        # param = []
        # scales = [0.5, 0.25]

        # for s in range(0, len(scales)):
        #     for i in range(0, maxFrameOffset):
        #         # skip own projection
        #         if i == destFrameIndex:
        #             continue

        #         param.append((destFrame,
        #                     self.__frameSet[i],
        #                     False,
        #                     self.__camera,
        #                     (self.__edgeDistanceLowerBoundary, self.__edgeDistanceUpperBoundary),
        #                     1,
        #                     result))

        # pool = mp.Pool(processes=self.__numOfThreads)
        # pool.starmap(ImageProcessing.projectEdges, param)
        # pool.terminate()
        ### REFINE with existing meaningful detections
        start = time.time()
        resultRefined = Manager().dict()
        paramRefined = []

        tmpDestFrame = copy.copy(destFrame)
        _, tmpDestFrameMask = cv.threshold(meaningfulEdges, 0, 255, cv.THRESH_BINARY)
        tmpDestFrame.setBoundaries(tmpDestFrameMask.astype(np.uint8))

        for i in range(0, maxFrameOffset):
            if i == destFrameIndex:
                continue

            tmpFrameTo = copy.copy(self.__frameSet[i])
            refinedMeaningfulEdges = tmpFrameTo.getMeaningfulEdges()

            if refinedMeaningfulEdges is None:
                continue

            _, tmpFrameToMask = cv.threshold(refinedMeaningfulEdges, 0, 255, cv.THRESH_BINARY)
            tmpFrameTo.setBoundaries(tmpFrameToMask.astype(np.uint8))

            paramRefined.append((tmpDestFrame,
                                 tmpFrameTo,
                                 False,
                                 self.__camera,
                                 (self.__edgeDistanceLowerBoundary, self.__edgeDistanceUpperBoundary),
                                 1,
                                 resultRefined))

        if len(paramRefined) < self.__frameOffset:
            self.__frameSet.pop(0)
            return (None, None)

        print('refine', len(paramRefined))
        pool = mp.Pool(processes=self.__numOfThreads)
        pool.starmap(ImageProcessing.projectEdges, paramRefined)
        pool.terminate()

        tmpDestFrame.projectedEdgeResults = resultRefined
        #tmpDestFrame.printProjectedEdgeResults()
        refinedMeaningfulEdges = tmpDestFrame.getMeaningfulEdges()

        logging.info('Projected %d meaningful frames in %f sec.' % (len(resultRefined), time.time() - start))
        ### END OF REFINE
        finalMeaningfulEdges = meaningfulEdges  # refinedMeaningfulEdges
        finalWorseEdges = destFrame.boundaries().copy()
        finalWorseEdges[np.where(meaningfulEdges > 0)] = 0

        ### PLOTTING
        if outputDir is not None:
            rgbImg = cv.cvtColor(destFrame.rgb(), cv.COLOR_BGR2BGRA)
            depthImg = cv.cvtColor(ImageProcessing.createHeatmap(destFrame.depth().copy()), cv.COLOR_BGR2RGBA)
            depthImg[np.where((depthImg == [0, 0, 0, 255]).all(axis=2))] = [255, 255, 255, 255]
            combined = cv.addWeighted(rgbImg, 0.7, depthImg, 0.3, 0)
            combined[np.nonzero(destFrame.boundaries())] = [255, 255, 255, 255]

            fig = plt.figure(1, figsize=(20, 40))
            plt.subplot(321)
            plt.imshow(combined)
            plt.subplot(322)
            plt.title('Out ouf Distance')
            plt.imshow(finalWorseEdges, cmap='hot')
            plt.subplot(323)
            plt.title('Backprojected')
            plt.imshow(meaningfulEdges, cmap='hot')

            diff = cv.add(meaningfulEdges, -1*refinedMeaningfulEdges)
            minVal, maxVal, _, _ = cv.minMaxLoc(diff)
            diff += minVal
            diff /= (maxVal+minVal)

            #_, refinedMeaningfulEdges = cv.threshold(refinedMeaningfulEdges, 0.5, 255, cv.THRESH_BINARY)

            plt.subplot(324)
            plt.title('Refined')
            plt.imshow(refinedMeaningfulEdges, cmap='hot')
            plt.subplot(325)
            plt.title('Backprojected - Refined')
            plt.imshow(diff, cmap='hot')
            plt.subplot(326)
            plt.title('Mask - Refined')
            plt.imshow(cv.add(destFrame.boundaries().astype(np.float64), -255.0*refinedMeaningfulEdges), cmap='hot')

            if not os.path.exists(os.path.join(outputDir, 'plots')):
                os.makedirs(os.path.join(outputDir, 'plots'))

            #fig.savefig(os.path.join(outputDir, 'plots', ('%f.svg' % (time.time()))), dpi=300)
            plt.show()
        # outputDir = None
        # if outputDir is not None:
        #     rgbImg = cv.cvtColor(destFrame.rgb(), cv.COLOR_BGR2BGRA)
        #     depthImg = cv.cvtColor(ImageProcessing.createHeatmap(destFrame.depth().copy()), cv.COLOR_BGR2RGBA)
        #     #depthImg[np.where((depthImg==[0,0,0,255]).all(axis=2))] = [255,255,255,255]
        #     combined = cv.addWeighted(rgbImg, 0.7, depthImg, 0.3, 0)
        #     combined[np.nonzero(destFrame.boundaries())] = [255, 255, 255, 255]
        #     # combined[np.nonzero(best)] = [0, 255, 0, 255]
        #     # combined[np.nonzero(good)] = [0, 255, 255, 255]
        #     # combined[np.nonzero(worse)] = [0, 0, 255, 255]

        #     # gradient (central differences)
        #     tmp = destFrame.rgb().copy()
        #     #tmp = cv.normalize(tmp, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
        #     dx, dy, mag, orientation = ImageProcessing.getGradientInformation(tmp)
        #     # mag[np.where(destFrame.depth==0)] = 0
        #     # orientation[np.where(destFrame.depth==0)] = 0
        #     #ImageProcessing.canny(tmp, 10, 10, 3, True, 3)

        #     fig = plt.figure(2)
        #     fig.suptitle('Gray Blurred')
        #     plt.subplot(321)
        #     plt.axis('off')
        #     plt.title('Projected Points')
        #     plt.imshow(cv.cvtColor(combined, cv.COLOR_BGRA2RGBA))

        #     plt.subplot(322)
        #     plt.axis('off')
        #     plt.title('Gradient Base Image')
        #     plt.imshow(tmp, cmap='gray')

        #     plt.subplot(323)
        #     plt.axis('off')
        #     plt.title('dx')
        #     plt.imshow(dx, cmap='gray')

        #     plt.subplot(324)
        #     plt.axis('off')
        #     plt.title('dy')
        #     plt.imshow(dy, cmap='gray')

        #     plt.subplot(325)
        #     plt.axis('off')
        #     plt.title('Orientation')
        #     plt.imshow(orientation, cmap='hsv')

        #     plt.subplot(326)
        #     plt.axis('off')
        #     plt.title('Magnitude')
        #     plt.imshow(cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1), cmap='hot')
        #     print('orientation', np.amax(orientation), np.amin(orientation))
        #     print('magnitude', np.amax(mag), np.amin(mag))

        #     plt.show()

        #     if not os.path.exists(os.path.join(outputDir, 'plots')):
        #         os.makedirs(os.path.join(outputDir, 'plots'))

        #     fig.savefig(os.path.join(outputDir, 'plots', ('%f.svg' % (time.time()))), dpi=300)
        ### END OF PLOTTING

        # remove first frame from frame set
        self.__frameSet.pop(0)

        return (finalMeaningfulEdges, finalWorseEdges)

    def reprojectEdgesFromConsecutiveFrameSet(self, frame: EdgeMatcherFrame = None, mode: EdgeMatcherMode = EdgeMatcherMode.REPROJECT, outputDir: str = None) -> (np.ndarray, np.ndarray):
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

        distTransMat = destFrame.distanceTransform()

        start = time.time()
        # needed to share result between threads
        result = Manager().dict()
        param = []

        for i in range(0, maxFrameOffset):
            if mode == EdgeMatcherMode.CENTERPROJECT:
                # avoid destination frame projection
                if i == self.__frameOffset:
                    continue

            param.append((self.__frameSet[i],
                          destFrame,
                          True,
                          self.__camera,
                          (self.__edgeDistanceLowerBoundary, self.__edgeDistanceUpperBoundary),
                          1,
                          result))

        pool = mp.Pool(processes=self.__numOfThreads)
        pool.starmap(ImageProcessing.projectEdges, param)
        pool.terminate()

        logging.info('Projected %d frames in %f sec.' % (maxFrameOffset-1, time.time() - start))
        destFrame.projectedEdgeResults = result
        destFrame.printProjectedEdgeResults()

        meaningfulEdges = destFrame.getMeaningfulEdges()
        finalWorseEdges = destFrame.boundaries().copy()
        finalWorseEdges[np.where(finalMeaningfulEdges > 0)] = 0
        # PLOTTING
        if outputDir is not None:
            rgbImg = cv.cvtColor(destFrame.rgb().copy(), cv.COLOR_BGR2BGRA)
            depthImg = cv.cvtColor(ImageProcessing.createHeatmap(destFrame.depth.copy()), cv.COLOR_BGR2RGBA)
            #depthImg[np.where((depthImg==[0,0,0,255]).all(axis=2))] = [255,255,255,255]
            combined = cv.addWeighted(rgbImg, 0.7, depthImg, 0.3, 0)
            combined[np.nonzero(destFrame.boundaries())] = [255, 255, 255, 255]
            # combined[np.nonzero(best)] = [0, 255, 0, 255]
            # combined[np.nonzero(good)] = [0, 255, 255, 255]
            # combined[np.nonzero(worse)] = [0, 0, 255, 255]

            #fig = plt.figure(1)
            # plt.subplot(131)
            # plt.axis('off')
            # plt.imshow(rgbImg)
            # plt.subplot(132)
            # plt.axis('off')
            # plt.imshow(depthImg)
            #plt.subplot(111)
            #plt.axis('off')
            #plt.imshow(cv.cvtColor(combined, cv.COLOR_BGRA2RGBA))
            # plt.show()

            if not os.path.exists(os.path.join(outputDir, 'plots')):
                os.makedirs(os.path.join(outputDir, 'plots'))

            fig.savefig(os.path.join(outputDir, 'plots', ('%f.svg' % (time.time()))), dpi=300)
        # END OF PLOTTING

        # remove first frame from frame set
        self.__frameSet.pop(0)

        return (meaningfulEdges, finalWorseEdges)

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
        frameFromT = frameFrom.T()
        frameToInvT = frameTo.invT()
        p1 = frameFromT.item((0, 0)) * point3d[0] + frameFromT.item((0, 1)) * point3d[1] + frameFromT.item((0, 2)) * point3d[2] + frameFromT.item((0, 3))
        p2 = frameFromT.item((1, 0)) * point3d[0] + frameFromT.item((1, 1)) * point3d[1] + frameFromT.item((1, 2)) * point3d[2] + frameFromT.item((1, 3))
        p3 = frameFromT.item((2, 0)) * point3d[0] + frameFromT.item((2, 1)) * point3d[1] + frameFromT.item((2, 2)) * point3d[2] + frameFromT.item((2, 3))
        q1 = frameToInvT.item((0, 0)) * p1 + frameToInvT.item((0, 1)) * p2 + frameToInvT.item((0, 2)) * p3 + frameToInvT.item((0, 3))
        q2 = frameToInvT.item((1, 0)) * p1 + frameToInvT.item((1, 1)) * p2 + frameToInvT.item((1, 2)) * p3 + frameToInvT.item((1, 3))
        q3 = frameToInvT.item((2, 0)) * p1 + frameToInvT.item((2, 1)) * p2 + frameToInvT.item((2, 2)) * p3 + frameToInvT.item((2, 3))

        # lower performance
        # point3d = np.array(([point3d[0]],[point3d[1]],[point3d[2]]), np.float64)
        # return np.dot(frameTo.invT_R(), np.dot(frameFrom.R(), point3d) + frameFrom.t()) + frameTo.invT_t()
        # point3d = np.array(([point3d[0]],[point3d[1]],[point3d[2]], [1.0]), np.float64)
        # return np.dot(frameTo.invT(), np.dot(frameFrom.T(), point3d))
        return np.array(([q1], [q2], [q3]), np.float64)
