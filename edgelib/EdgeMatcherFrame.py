import logging
import numpy as np
import cv2 as cv
from .Frame import Frame
from . import Utilities


class EdgeMatcherFrame(Frame):
    '''
    Extension of the frame for the edge matching algorithm.
    '''

    def __init__(self):
        '''
        Constructor.
        '''
        super().__init__()

        # key = uid, value = meaningful edge result
        self.projectedEdgeResults = {}
        self.__meaningfulEdges = None

        self.scaledMeaningfulEdges = None
        self.refinedMeaningfulEdges = None

    def printProjectedEdgeResults(self) -> None:
        '''
        Print projected edge results as best, good and worse in percent.
        '''
        if len(self.projectedEdgeResults) == 0:
            print('No projected edge results.')

        for uid, projectedEdges in self.projectedEdgeResults.items():
            best, good, worse = cv.split(projectedEdges)

            numBest = len(best[np.where(best > 0)])
            numGood = len(good[np.where(good > 0)])
            numWorse = len(worse[np.where(worse > 0)])
            total = numBest + numGood + numWorse

            print(uid, ': b %.2f%%, g %.2f%%, w %.2f%%' % (numBest*100/total, numGood*100/total, numWorse*100/total))

    def concatProjectedResults(self) -> np.ndarray:
        '''
        Concatenate projected results in one single image. Each channel represents one of the following
        conditions: best, good or worse images.
        '''
        if not self.isValid():
            raise ValueError('Invalid frame data.')

        h, w = self.__boundaries.shape
        concatEdges = np.zeros((h, w, 3))

        for projectedResult in self.projectedEdgeResults.values():
            concatEdges = cv.add(concatEdges, projectedResult)

        return concatEdges

    def getMeaningfulEdges(self) -> np.ndarray:
        '''
        Get meaningful edges of the current frame.
        The intermediate results are weighted (sinoid function) by its hit quote and added together.

        Returns probability of a meaningful edge per pixel [0 ... 1]. 1 is good, 0 is no edge.
        '''
        total = len(self.projectedEdgeResults)

        if total == 0:
            return None

        h, w = self.rgb().shape[:2]
        meaningfulEdges = np.zeros((h, w), np.float64)

        for projectedEdges in self.projectedEdgeResults.values():
            best, good, worse = cv.split(projectedEdges)

            numBest = len(best[np.where(best > 0)])
            numGood = len(good[np.where(good > 0)])
            numWorse = len(worse[np.where(worse > 0)])

            # linear weight function
            weight = (numBest + numGood) / (numBest + numGood + numWorse)
            # sinoid weight function
            # weight = np.sin(Utilities.rescale(weight, 0, 1, 0, np.pi/2.0))

            curMeaningfulEdges = cv.add(best, good)
            curMeaningfulEdges *= weight

            meaningfulEdges = cv.add(meaningfulEdges, curMeaningfulEdges)

        meaningfulEdges /= len(self.projectedEdgeResults)
        #print(cv.minMaxLoc(meaningfulEdges))

        # remove values less the n times seen
        #minNumOfFramesDetected = 0
        #_, meaningfulEdges = cv.threshold(meaningfulEdges, minNumOfFramesDetected, 255, cv.THRESH_BINARY)

        return meaningfulEdges
