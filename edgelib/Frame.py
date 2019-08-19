from typing import List
import numpy as np
import sys
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
        self.rgb = None
        self.depth = None
        self.mask = None
        # 4x4 transformation matrix
        self.__T = None
        self.__invT = None

    def __str__(self) -> str:
        '''
        Class representation as string.

        Returns formatted class parameters.
        '''
        out = 'RGB: %dx%d\n' % (self.rgb.shape[:2])
        out += 'Depth: %dx%d\n' % (self.depth.shape[:2])
        out += 'Mask: %dx%d\n' % (self.mask.shape[:2])
        out += 'T: %s\nT⁻¹%s\n' % (np.array2string(self.__T), np.array2string(self.__invT))

        return out

    def R(self) -> np.ndarray:
        '''
        Rotation matrix.

        Returns the 3x3 rotation matrix.
        '''
        return self.__getR(self.__T)

    def invT_R(self) -> np.ndarray:
        '''
        Inverse rotation matrix.

        Returns the 3x3 inverse rotation matrix.
        '''
        return self.__getR(self.__invT)

    def t(self) -> np.ndarray:
        '''
        Get the translation vector.

        Returns the 3x1 translation vector.
        '''
        return self.__gett(self.__T)

    def invT_t(self) -> np.ndarray:
        '''
        Get the inverse translation vector.

        Returns the 3x1 translation vector.
        '''
        return self.__gett(self.__invT)

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
        Get the inverse transformation matrx.
        '''
        return self.__invT

    def T(self) -> np.ndarray:
        '''
        Get the transformation matrix.
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
        self.__T = np.array(T, np.float64)
        self.__invT = np.array(T.inv(), np.float64)

    def isValid(self) -> bool:
        '''
        Check if frame is valid.
        '''
        if(self.rgb is None or self.depth is None or self.mask is None):
            return False

        if(type(self.rgb) is not np.ndarray or type(self.depth) is not np.ndarray or type(self.mask) is not np.ndarray):
            return False

        if(self.rgb.size == 0 or self.depth.size == 0 or self.mask.size == 0):
            return False

        return True
