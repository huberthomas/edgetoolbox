import os
import cv2 as cv
import numpy as np


class Camera:
    '''
    The camera describe the camera parameters like the camera matrix,
    distortion coefficients and the depth scale factor. It is possible to read and write the 
    parameters to a file. 
    '''

    def __init__(self) -> None:
        '''
        Constructor.
        '''
        self.__cameraMatrix = np.zeros((3, 3), np.float64)
        self.__distortionCoefficients = np.zeros(5, np.float64)
        self.__depthScaleFactor = np.float64(1.0)

        self.__tagCameraMatrix = 'cameraMatrix'
        self.__tagDistortionCoefficients = 'distortionCoefficients'
        self.__tagDepthScaleFactor = 'depthScaleFactor'

    def __str__(self) -> str:
        '''
        Class representation as string.

        Returns formatted class parameters.
        '''
        out = '%s\n%s\n' % (self.__tagCameraMatrix, np.array2string(self.__cameraMatrix))
        out += '%s\n%s\n' % (self.__tagDistortionCoefficients, np.array2string(self.__distortionCoefficients))
        out += '%s\n%f' % (self.__tagDepthScaleFactor, self.__depthScaleFactor)

        return out

    def distortionCoefficients(self) -> np.ndarray:
        '''
        Get the distortion coefficients.

        Returns the distortion coefficients d1 - d5.
        '''
        return self.__distortionCoefficients

    def cameraMatrix(self) -> np.ndarray:
        '''
        Get the camera matrix.

        Returns a 3x3 camera matrix.
        '''
        return self.__cameraMatrix

    def depthScaleFactor(self) -> float:
        '''
        Get the depth scale factor.

        Returns the depth scale factor.
        '''
        return self.__depthScaleFactor

    def setDepthScaleFactor(self, factor: float = None) -> None:
        '''
        Set the depth scale factor.

        factor Depth scale factor.
        '''
        if factor is None or factor == 0:
            raise ValueError('Invalid depth scale factor.')

        if factor < 0:
            factor = abs(factor)

        self.__depthScaleFactor = np.float64(factor)

    def setCameraMatrix(self, cameraMatrix: np.ndarray = None) -> None:
        '''
        Set the 3x3 camera matrix.

        cameraMatrix 3x3 camera matrix.
        '''
        if cameraMatrix is None or not cameraMatrix.shape == (3, 3):
            raise ValueError('Invalid camera matrix. Must be 3x3, not %dx%d.' % (cameraMatrix.shape))

        self.__cameraMatrix = cameraMatrix.astype(np.float64)

    def setDistortionCoefficients(self, distortionCoefficients: np.ndarray = None) -> None:
        '''
        Set the 1x5 distortion coefficients.

        distortionCoefficients 1x5 distortion coefficients.        
        '''
        if distortionCoefficients is None or not distortionCoefficients.shape == (5, 1):
            raise ValueError('Invalid distortion coefficients. Must be 1x5, not %dx%d.' % (distortionCoefficients.shape))

        self.__distortionCoefficients = distortionCoefficients.astype(np.float64)

    def loadFromFile(self, filePath: str = None) -> None:
        '''
        Load camera matrix from OpenCV yaml file, e.g.

        %YAML:1.0
        ---
        cameraMatrix: !!opencv-matrix
        rows: 3
        cols: 3
        dt: f
        data: [ 517.30640, 0., 318.643040, 0., 516.469215, 255.313989, 0., 0.,
            1. ]
        distortionCoefficients: [ 0., 0., 0., 0., 0. ]
        depthScaleFactor: 5000.0

        filePath File path to the OpenCV camera YAML file.
        '''
        if filePath is None or not os.path.exists(filePath):
            raise ValueError('Invalid camera file path "%s".' % (filePath))

        try:
            fs = cv.FileStorage(filePath, cv.FILE_STORAGE_READ)
            # camera matrix
            self.setCameraMatrix(fs.getNode(self.__tagCameraMatrix).mat())
            # distortion coefficients
            dcNode = fs.getNode(self.__tagDistortionCoefficients)
            # can be a vector or an OpenCV matrix
            if dcNode.type() == cv.FileNode_MAP:
                self.setDistortionCoefficients(dcNode.mat())
            else:
                distortionCoefficients = []
                for i in range(0, dcNode.size()):
                    distortionCoefficients.append([dcNode.at(i).real()])

                self.setDistortionCoefficients(np.asarray(distortionCoefficients))
            # depth scale factor
            self.setDepthScaleFactor(fs.getNode(self.__tagDepthScaleFactor).real())

            fs.release()
        except Exception as e:
            raise e

    def writeToFile(self, filePath: str = None) -> None:
        '''
        Write camera data to file.

        filePath File that is used to store the camera data.        
        '''
        if filePath is None or len(filePath) == 0:
            raise ValueError('Invalid file path.')
        try:
            fs = cv.FileStorage(filePath, cv.FILE_STORAGE_WRITE)
            fs.write(self.__tagCameraMatrix, self.__cameraMatrix)
            fs.write(self.__tagDistortionCoefficients, self.__distortionCoefficients)
            fs.write(self.__tagDepthScaleFactor, self.__depthScaleFactor)
            fs.release()
        except Exception as e:
            raise e

    def setFx(self, val: float = None):
        '''
        Set new value.

        val New value.
        '''
        if val is None or val <= 0:
            raise ValueError('Invalid focal length.')

        self.__cameraMatrix.itemset((0, 0), val)

    def setFy(self, val: float = None):
        '''
        Set new value.

        val New value.
        '''
        if val is None or val <= 0:
            raise ValueError('Invalid focal length.')

        self.__cameraMatrix.itemset((1, 1), val)

    def setCx(self, val: float = None):
        '''
        Set new value.

        val New value.
        '''
        if val is None or val <= 0:
            raise ValueError('Invalid principal point.')

        self.__cameraMatrix.itemset((0, 2), val)

    def setCy(self, val: float = None):
        '''
        Set new value.

        val New value.
        '''
        if val is None or val <= 0:
            raise ValueError('Invalid principal point.')

        self.__cameraMatrix.itemset((1, 2), val)

    def fx(self) -> float:
        '''
        Focal length x.

        Returns focal length x.
        '''
        return self.__cameraMatrix.item((0, 0))

    def fy(self) -> float:
        '''
        Focal length y.

        Returns focal length y.
        '''
        return self.__cameraMatrix.item((1, 1))

    def cx(self) -> float:
        '''
        Principal point x.

        Returns principal length x.
        '''
        return self.__cameraMatrix.item((0, 2))

    def cy(self) -> float:
        '''
        Principal point y.

        Returns principal point y.
        '''
        return self.__cameraMatrix.item((1, 2))

    def d1(self) -> float:
        '''
        Distortion coefficient 1.

        Returns distortion coefficient 1.
        '''
        return self.__distortionCoefficients(0)

    def d2(self) -> float:
        '''
        Distortion coefficient 2.

        Returns distortion coefficient 2.
        '''
        return self.__distortionCoefficients(1)

    def d3(self) -> float:
        '''
        Distortion coefficient 3.

        Returns distortion coefficient 3.
        '''
        return self.__distortionCoefficients(2)

    def d4(self) -> float:
        '''
        Distortion coefficient 4.

        Returns distortion coefficient 4.
        '''
        return self.__distortionCoefficients(3)

    def d5(self) -> float:
        '''
        Distortion coefficient 5.

        Returns distortion coefficient 5.
        '''
        return self.__distortionCoefficients(4)
