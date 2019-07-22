import numpy as np

from edgelib.TumGroundTruth import TumGroundTruth

class TumGtAssociated:
    '''
    The GroundTruth struct contains all the information from
    the associated groundtruth.txt file located in each dataset of the TUM RGB-D
    dataset.
    '''

    def __init__(self):
        '''
        Constructor.
        '''
        # number of seconds since the Unix epoch
        self.gt = None
        self.rgb = None
        self.depth = None

    def loadFromStringLine(self, line: str = None):
        '''
        Load data from string line of the form: 
        timestamp tx ty tz qx qy qz qw rgbFileName depthFileName
        1305031102.1758 1.3405 0.6266 1.6575 0.6574 0.6126 -0.2949 -0.3248 1305031102.175304.png 1305031102.160407.png

        line The string line of the form: timestamp tx ty tz qx qy qz qw rgbFileName depthFileName
        '''
        if len(line) == 0:
            raise ValueError('Line is empty.')

        line = line.strip()
        entries = line.split(' ')

        if len(entries) < 10:
            raise ValueError('Missing information.')

        self.gt = TumGroundTruth()
        self.gt.timestamp = np.float64(entries[0])
        self.gt.t[0] = np.float64(entries[1])
        self.gt.t[1] = np.float64(entries[2])
        self.gt.t[2] = np.float64(entries[3])
        self.gt.q[0] = np.float64(entries[7])
        self.gt.q[1] = np.float64(entries[4])
        self.gt.q[2] = np.float64(entries[5])
        self.gt.q[3] = np.float64(entries[6])

        self.rgb = entries[8]
        self.depth = entries[9]