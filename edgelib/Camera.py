import os
import cv2
import numpy as np


class Camera:
    '''
    '''

    def __init__(self) -> None:
        '''
        Constructor. The camera describe the camera parameters like the camera matrix,
        distortion coefficients and much more. It is possible to read and write the 
        parameters to a file. 
        '''
        self.cameraMatrix = np.zeros((3, 3))
        self.distortionCoefficients = np.zeros(5)
        self.depthScaleFactor = 1

        self.__tagCameraMatrix = 'cameraMatrix'
        self.__tagDistortionCoefficients = 'distortionCoefficients'
        self.__tagDepthScaleFactor = 'depthScaleFactor'

    def __str__(self) -> str:
        '''
        Class representation as string.

        Returns formatted class parameters.
        '''
        out = '%s\n%s\n' % (self.__tagCameraMatrix, np.array2string(self.cameraMatrix))
        out += '%s\n%s\n' % (self.__tagDistortionCoefficients, np.array2string(self.distortionCoefficients))
        out += '%s\n%f' % (self.__tagDepthScaleFactor, self.depthScaleFactor)

        return out

    def setDepthScaleFactor(self, factor: float = None) -> None:
        '''
        Set the depth scale factor.

        factor Depth scale factor.
        '''
        if factor is None or factor == 0:
            raise ValueError('Invalid depth scale factor.')

        if factor < 0:
            factor = abs(factor)

        self.depthScaleFactor = factor

    def setCameraMatrix(self, cameraMatrix: np.array = None) -> None:
        '''
        Set the 3x3 camera matrix.

        cameraMatrix 3x3 camera matrix.
        '''
        if cameraMatrix is None or not cameraMatrix.shape == (3, 3):
            raise ValueError('Invalid camera matrix. Must be 3x3, not %dx%d.' % (cameraMatrix.shape))

        self.cameraMatrix = cameraMatrix

    def setDistortionCoefficients(self, distortionCoefficients: np.array = None) -> None:
        '''
        Set the 1x5 distortion coefficients.

        distortionCoefficients 1x5 distortion coefficients.        
        '''
        if distortionCoefficients is None or not distortionCoefficients.shape == (5, 1):            
            raise ValueError('Invalid distortion coefficients. Must be 1x5, not %dx%d.' % (distortionCoefficients.shape))

        self.distortionCoefficients = distortionCoefficients

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
            fs = cv2.FileStorage(filePath, cv2.FILE_STORAGE_READ)
            # camera matrix
            self.setCameraMatrix(fs.getNode(self.__tagCameraMatrix).mat())
            # distortion coefficients
            dcNode = fs.getNode(self.__tagDistortionCoefficients)
            # can be a vector or an OpenCV matrix
            if dcNode.type() == cv2.FileNode_MAP:
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
            fs = cv2.FileStorage(filePath, cv2.FILE_STORAGE_WRITE)
            fs.write(self.__tagCameraMatrix, self.cameraMatrix)
            fs.write(self.__tagDistortionCoefficients, self.distortionCoefficients)
            fs.write(self.__tagDepthScaleFactor, self.depthScaleFactor)
            fs.release()
        except Exception as e:
            raise e
