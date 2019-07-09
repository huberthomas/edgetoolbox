import os
import multiprocessing as mp
import cv2

from functools import partial
from typing import List

from edgelib import Utilities

mp.set_start_method('spawn', True)


class DataAugmentation:
    '''
    This class generates augmented data out of an existing image folder.
    '''

    def __init__(self, inputDir: str = None, outputDir: str = None) -> None:
        '''
        Constructor.

        inputDir Input directory.

        outputDir Output directory. Results will be stored here.
        '''
        if inputDir is None or len(inputDir) == 0:
            raise ValueError('Input directory is empty.')

        if not os.path.exists(inputDir):
            raise ValueError('Input directory does not exist.')

        if outputDir is None or len(outputDir) == 0:
            raise ValueError('Output directory is empty.')

        if inputDir == outputDir:
            raise ValueError('Input must be different to output directory.')

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        self.__inputDir = inputDir
        self.__outputDir = outputDir
        self.__angles = self.__calculateRotationAngles(16)
        self.__scales = [1.0]
        self.__flipHorizontal = False
        self.__flipVertical = False
        self.__numOfThreads = mp.cpu_count()
        self.__cropBlackRotationBorder = True

    def setCropBlackRotationBorder(self, enable: bool = True) -> None:
        '''
        If images are rotated black areas fills out empty data. Enabling crops image without black regions.

        enable Enable black border cropping.
        '''
        self.__cropBlackRotationBorder = enable

    def setScales(self, scales: List[float] = None) -> None:
        '''
        Set image scales that should be generated.

        scales Array that contains scale values.
        '''
        if scales is None or len(scales) == 0:
            raise ValueError('Scales must be an array that contains scale values.')

        self.__scales = scales

    def setNumOfThreads(self, numOfThreads: int = None) -> None:
        '''
        Set the number of threads that are used in the thread pool to process the 
        augmented data.
        '''
        if numOfThreads is None or numOfThreads <= 0:
            raise ValueError('Number of threads must be an interger value greater than 0.')

        self.__numOfThreads = numOfThreads

    def generateData(self) -> None:
        '''
        Function to start data generation. A pool of numThreads is transforming the images to
        the output folder.
        '''
        imageFileNames = Utilities.getFileNames(self.__inputDir)

        f = open(os.path.join(self.__outputDir, "data.txt"), "w")

        try:
            for imageFileName in imageFileNames:
                imagePath = os.path.join(self.__inputDir, imageFileName)

                img = cv2.imread(imagePath)

                param = []
                cropBlackRotationBorder = self.__cropBlackRotationBorder

                for scale in self.__scales:
                    for angle in self.__angles:
                        subDir = '%.1f_%d_%d_%.1f' % (
                            angle, False, False, scale)
                        dirPath = os.path.join(self.__outputDir, subDir)
                        outFilePath = [dirPath]
                        param.append((os.path.join(dirPath, imageFileName), img, angle, scale, False, False, cropBlackRotationBorder))
                        f.write(os.path.join(subDir, imageFileName) + '\n')

                        if self.__flipHorizontal:
                            subDir = '%.1f_%d_%d_%.1f' % (
                                angle, True, False, scale)
                            dirPath = os.path.join(self.__outputDir, subDir)
                            outFilePath.append(dirPath)
                            param.append((os.path.join(dirPath, imageFileName), img, angle, scale, True, False, cropBlackRotationBorder))
                            f.write(os.path.join(subDir, imageFileName) + '\n')

                        if self.__flipVertical:
                            subDir = '%.1f_%d_%d_%.1f' % (
                                angle, False, True, scale)
                            dirPath = os.path.join(self.__outputDir, subDir)
                            outFilePath.append(dirPath)
                            param.append((os.path.join(dirPath, imageFileName), img, angle, scale, False, True, cropBlackRotationBorder))
                            f.write(os.path.join(subDir, imageFileName) + '\n')

                        if self.__flipHorizontal and self.__flipVertical:
                            subDir = '%.1f_%d_%d_%.1f' % (
                                angle, True, True, scale)
                            dirPath = os.path.join(self.__outputDir, subDir)
                            outFilePath.append(dirPath)
                            param.append((os.path.join(dirPath, imageFileName), img, angle, scale, True, True, cropBlackRotationBorder))
                            f.write(os.path.join(subDir, imageFileName) + '\n')

                        for dirPath in outFilePath:
                            if not os.path.exists(dirPath):
                                os.makedirs(dirPath)

                pool = mp.Pool(processes=self.__numOfThreads)
                pool.starmap(Utilities.transformAndSaveImage, param)
                pool.terminate()
        except Exception as e:
            f.close()
            raise e

    def enableFlip(self, enableHorizontal: bool = True, enableVertical: bool = True) -> None:
        '''
        Enable horizontal/vertical flipping during data generation.

        enableHorizontal Flag to en-/disable flipping.

        enableVertical Flag to en-/disable flipping.
        '''
        self.__flipHorizontal = enableHorizontal
        self.__flipVertical = enableVertical

    def enableFlipHorizontal(self, enable: bool = True) -> None:
        '''
        Enable horizontal flipping during data generation.

        enable Flag to en-/disable flipping.
        '''
        self.__flipHorizontal = enable

    def enableFlipVertical(self, enable: bool = True) -> None:
        '''
        Enable vertical flipping during data generation.

        enable Flag to en-/disable flipping.
        '''
        self.__flipVertical = enable

    def setNumberOfAngles(self, numOfAngles: int = None) -> None:
        '''
        Calculates automatically the number of angles defined by the input parameter.

        numOfAngles Number angles, e.g. 4 will result in 4 angles: [0 90 180 270]
        '''
        try:
            self.__angles = self.__calculateRotationAngles(numOfAngles)
        except Exception as e:
            raise e

    def setRotationAngles(self, angles: List[float] = None) -> None:
        '''
        Set multiple rotation angles for data augmentation.

        angles Angles that are used during the data augmentation process.
        '''
        if angles is None or len(angles) == 0:
            raise ValueError('No angles defined.')

        self.__angles.clear()

        for a in angles:
            self.__angles.append(abs(a))

    def __calculateRotationAngles(self, numOfAngles: int = None) -> List[float]:
        '''
        Calculate angles given by a number of angles. 
        
        numOfAngles Number angles, e.g. 4 will give [0 90 180 270]

        Returns calculated angles as list of floats.
        '''
        if numOfAngles <= 0 or numOfAngles is None:
            raise ValueError('Parameter must be an integer greater than 0.')

        if numOfAngles == 1:
            return [0]

        factor = 360.0 / float(numOfAngles)

        angles = [0]

        while(angles[len(angles) - 1] < (360 - factor)):
            angles.append(angles[len(angles) - 1] + factor)

        return angles
