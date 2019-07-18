import numpy as np


class TumGroundTruth:
    '''
    The GroundTruth struct contains all the information from
    the groundtruth.txt file located in each dataset of the TUM RGB-D
    dataset.
    '''

    def __init__(self):
        # number of seconds since the Unix epoch
        self.timestamp = 0
        # tx,ty,tz: the position of the optical center of the color camera with respect to the
        # world origin as defined by the motion capture system
        self.t = np.zeros((3, 1))
        # qx, qy, qz, qw: the orientation of the optical center of the color camera in form of a unit
        # quaternion with respect to the world origin as defined by the motion capture system
        self.q = np.zeros((4, 1))

    def loadFromStringLine(self, line: str = None):
        '''
        Load data from string line of the form: 
        timestamp tx ty tz qx qy qz qw, e.g.
        1305031098.6659 1.3563 0.6305 1.6380 0.6132 0.5962 -0.3311 -0.3986

        line The string line of the form: timestamp tx ty tz qx qy qz qw
        '''
        if len(line) == 0:
            raise ValueError('Line is empty.')

        line = line.strip()
        entries = line.split('\t *')

        if len(entries) < 8:
            raise ValueError('Missing information.')

        self.timestamp = np.float(entries[0])
        self.t[0] = np.float(entries[1])
        self.t[1] = np.float(entries[2])
        self.t[2] = np.float(entries[3])
        self.q[0] = np.float(entries[7])
        self.q[1] = np.float(entries[4])
        self.q[2] = np.float(entries[5])
        self.q[3] = np.float(entries[6])
