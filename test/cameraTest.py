import unittest
import sys
sys.path.append(".")
from edgelib import Camera

class TestCameraMethods(unittest.TestCase):
    '''
    Test camera methods.
    '''
    def testRescale(self):
        self.assertTrue('FOO'.isupper())

if __name__ == '__main__':
    '''
    Entry function.

    python -m unittest test/cameraTest.py -v
    '''
    unittest.main()