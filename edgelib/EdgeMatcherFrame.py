import logging
import numpy as np
import cv2 as cv
from .Frame import Frame

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
        self.meaningfulEdges = None
    
    def printProjectedEdgeResults(self) -> None:
        '''
        Print projected edge results as best, good and worse in percent.
        '''
        if len(self.projectedEdgeResults) == 0:
            print('No projected edge results.')

        for uid, projectedEdges in self.projectedEdgeResults.items():
            best, good, worse = cv.split(projectedEdges)
            
            numBest = len(best[np.where(best>0)])
            numGood = len(good[np.where(good>0)])
            numWorse = len(worse[np.where(worse>0)])
            total = numBest + numGood + numWorse

            print(uid, ': b %.2f%%, g %.2f%%, w %.2f%%' % (numBest*100/total, numGood*100/total, numWorse*100/total))

    def concatProjectedResults(self) -> np.ndarray:
        '''
        Concatenate projected results in one single image. Each channel represents one of the following
        conditions: best, good or worse images.
        '''
        if not self.isValid():
            raise ValueError('Invalid frame data.')

        h, w = self.mask.shape
        concatEdges = np.zeros((h, w, 3))

        for projectedResult in self.projectedEdgeResults.values():
            concatEdges = cv.add(concatEdges, projectedResult)
        
        return concatEdges

    def getMeaningfulEdges(self) -> np.ndarray:
        '''
        Get meaningful edges of the current frame
        '''
        total = len(self.projectedEdgeResults)

        if total == 0:
            return None

        projectedEdges = self.concatProjectedResults()
        best, good, worse = cv.split(projectedEdges)        

        maxVal = projectedEdges.max()        
        numBest = len(best[np.where(best>0)])
        numGood = len(good[np.where(good>0)])
        numWorse = len(worse[np.where(worse>0)])
        total = numBest + numGood + numWorse

        weight = (numBest + numGood) / (numBest + numGood + numWorse)
        print(numBest, numGood, numWorse, total, weight)
        meaningfulEdges = cv.add(best, good)

        meaningfulEdges *= weight/maxVal

        print(maxVal, weight, cv.minMaxLoc(meaningfulEdges))
        # remove values less the n times seen
        #minNumOfFramesDetected = 0
        #_, meaningfulEdges = cv.threshold(meaningfulEdges, minNumOfFramesDetected, 255, cv.THRESH_BINARY)


        return meaningfulEdges  