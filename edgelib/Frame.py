from typing import List
import numpy as np
import sys
import cv2 as cv
sys.path.insert(0, 'dependencies/sophus/py/')
import sophus


class Frame:
    '''
    A frame contains all information of a current scene at a defined time point. It is the information
    that a possible sensor collects during a record.
    '''

    def __init__(self) -> None:
        '''
        Constructor.
        '''
        self.uid = None
        # images
        self.__rgb = None
        self.__depth = None
        self.__boundaries = None
        self.__distanceTransform = None
        # 4x4 transformation matrix
        self.__T = None
        self.__R = None
        self.__t = None
        self.__invT = None
        self.__invT_R = None
        self.__invT_t = None

    def __str__(self) -> str:
        '''
        Class representation as string.

        Returns formatted class parameters.
        '''
        if not self.isValid():
            raise ValueError('Invalid frame data.')

        out = 'RGB: %dx%d\n' % (self.__rgb.shape[:2])
        out += 'Depth: %dx%d\n' % (self.__depth.shape[:2])
        out += 'Mask: %dx%d\n' % (self.__boundaries.shape[:2])
        out += 'T: %s\nT⁻¹%s\n' % (np.array2string(self.__T), np.array2string(self.__invT))

        return out

    def rgb(self) -> np.ndarray:
        '''
        Get the RGB file.
        '''
        return self.__rgb

    def setRgb(self, rgb: np.ndarray = None) -> None:
        '''
        Set the RGB image file.

        rgb RGB file.
        '''
        if rgb is None:
            raise ValueError('Invalid RGB file.')

        self.__rgb = rgb

    def depth(self) -> np.ndarray:
        '''
        Get the depth file.
        '''
        return self.__depth

    def setDepth(self, depth: np.ndarray = None) -> None:
        '''
        Set the depth image file.

        depth Depth file.
        '''
        if depth is None:
            raise ValueError('Invalid depth file.')

        self.__depth = depth

    def boundaries(self) -> np.ndarray:
        '''
        Get the boundaries file.
        '''
        return self.__boundaries

    def setBoundaries(self, boundaries: np.ndarray = None) -> None:
        '''
        Set the boundaries image file.

        boundaries Boundaries file.
        '''
        if boundaries is None:
            raise ValueError('Invalid boundaries file.')


        self.__boundaries = boundaries
        self.__distanceTransform = cv.distanceTransform(255 - boundaries, cv.DIST_L2, cv.DIST_MASK_PRECISE)

    def distanceTransform(self) -> np.ndarray:
        '''
        Get the distance transform of the boundaries image.
        '''
        return self.__distanceTransform


    def R(self) -> np.ndarray:
        '''
        Rotation matrix.

        Returns the 3x3 rotation matrix.
        '''
        return self.__R

    def invT_R(self) -> np.ndarray:
        '''
        Inverse rotation matrix.

        Returns the 3x3 inverse rotation matrix.
        '''
        return self.__invT_R

    def t(self) -> np.ndarray:
        '''
        Get the translation vector.

        Returns the 3x1 translation vector.
        '''
        return self.__t

    def invT_t(self) -> np.ndarray:
        '''
        Get the inverse translation vector.

        Returns the 3x1 translation vector.
        '''
        return self.__invT_t

    def __getR(self, T: np.ndarray = None) -> np.ndarray:
        '''
        Extract the rotation from the transformation matrix.

        T 4x4 Transformation matrix.

        Returns the 3x3 orientation matrix.
        '''
        if T is None:
            raise ValueError('Invalid transformation matrix.')

        return T[0:3, 0:3].astype(np.float64)

    def __gett(self, T: np.ndarray = None) -> np.ndarray:
        '''
        Extract the translation from the transformation matrix.

        T 4x4 Transformation matrix.

        Returns the 3x1 tranlation vector.
        '''
        if T is None:
            raise ValueError('Invalid transformation matrix.')

        return T[0:3, 3].astype(np.float64)

    def invT(self) -> np.ndarray:
        '''
        Get the inverse 4x4 transformation matrx.
        '''
        return self.__invT

    def T(self) -> np.ndarray:
        '''
        Get the 4x4 transformation matrix.
        '''
        return self.__T

    def setT(self, q: List[float] = None, t: List[float] = None) -> None:
        '''
        Set the transformation matrix based on a quaternion and a translation.

        q Quaternion (w, x, y, z).

        t Translation vector.
        '''
        if q is None or not len(q) == 4:
            raise ValueError('Invalid quaternion vector. Must be 1x4, not 1x%d.' % len(q))

        if t is None or not len(t) == 3:
            raise ValueError('Invalid translation vector. Must be 1x4, not 1x%d.' % len(t))

        t = sophus.Vector3(t[0], t[1], t[2])
        v = sophus.Vector3(q[1], q[2], q[3])
        s = sophus.Se3(sophus.So3(sophus.Quaternion(q[0], v)), t)

        T = s.matrix()
        tmpT = np.array(T, np.float64)
        self.__T = tmpT[0:3, 0:4]
        self.__R = self.__getR(self.__T)
        self.__t = self.__gett(self.__T)

        invT = np.array(T.inv(), np.float64)
        self.__invT = invT[0:3, 0:4]
        self.__invT_R = self.__getR(self.__invT)
        self.__invT_t = self.__gett(self.__invT)

    def isValid(self) -> bool:
        '''
        Check if frame is valid.
        '''
        if(self.__rgb is None or self.__depth is None or self.__boundaries is None):
            return False

        if(type(self.__rgb) is not np.ndarray or type(self.__depth) is not np.ndarray or type(self.__boundaries) is not np.ndarray):
            return False

        if(self.__rgb.size == 0 or self.__depth.size == 0 or self.__boundaries.size == 0):
            return False

        return True
