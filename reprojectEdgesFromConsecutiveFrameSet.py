import logging
import argparse
import time
import os
import sys
import cv2
from edgelib.TumGroundTruthHandler import TumGroundTruthHandler
from edgelib.EdgeMatcher import EdgeMatcher
from edgelib.Camera import Camera
from edgelib.Frame import Frame
from edgelib import Utilities

def checkInputParameter(args: any) -> any:
    '''
    Check the input parameter from argparse.

    args Arguments.

    Returns parsed arguments. Throws exception if error occurs.
    '''

    if args.rgbDir == None or not os.path.exists(args.rgbDir):
        raise ValueError('Invalid RGB image directory.')

    if args.depthDir == None or not os.path.exists(args.depthDir):
        raise ValueError('Invalid depth image directory.')

    if args.maskDir == None or not os.path.exists(args.maskDir):
        raise ValueError('Invalid mask image directory.')

    if args.groundTruthFile == None or not os.path.exists(args.groundTruthFile):
        raise ValueError('Invalid ground truth file.')

    if args.camCalibFile == None or not os.path.exists(args.camCalibFile):
        raise ValueError('Invalid camera calibration file.')

    if args.outputFile == None or len(args.outputFile) == 0:
        raise ValueError('Invalid output file.')

    if args.frameOffset == 0:
        raise ValueError('Invalid frame offset. Must be greater than 0.')

    if args.frameOffset < 0:
        args.frameOffset = abs(args.frameOffset)

    if args.lowerEdgeDistanceBoundary is None:
        raise ValueError('Invalid edge distance lower boundary value.')

    if args.lowerEdgeDistanceBoundary < 0:
        args.lowerEdgeDistanceBoundary = abs(args.lowerEdgeDistanceBoundary)

    if args.upperEdgeDistanceBoundary is None:
        raise ValueError('Invalid edge distance upper boundary value.')

    if args.upperEdgeDistanceBoundary < 0:
        args.upperEdgeDistanceBoundary = abs(args.upperEdgeDistanceBoundary)

    if args.upperEdgeDistanceBoundary < args.upperEdgeDistanceBoundary:
        raise ValueError('Upper boundary must be greater than the lower boundary.')

    outputDir = os.path.dirname(args.outputFile)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        logging.info('Created output directory "%s".' % (outputDir))

    return args


def parseArgs() -> any:
    '''
    Parse user arguments.
    '''
    parser = argparse.ArgumentParser(description='Reconstruct undefined areas in a depth image.')
    parser.add_argument('-r', '--rgbDir', type=str, default=None, required=True, help='RGB image directory.')
    parser.add_argument('-d', '--depthDir', type=str, default=None, required=True, help='Depth image directory.')
    parser.add_argument('-m', '--maskDir', type=str, default=None, required=True, help='Mask image directory.')
    parser.add_argument('-c', '--camCalibFile', type=str, default=None, required=True, help='Camera calibration file.')
    parser.add_argument('-g', '--groundTruthFile', type=str, default=None, required=True, help='Associated TUM ground truth file, e.g. groundtruth_associated.txt.')
    parser.add_argument('-o', '--outputFile', type=str, default=None, required=True, help='Result output file.')
    parser.add_argument('-f', '--frameOffset', type=int, default=1, help='Frame offset. Offset between the reprojected frames.')
    parser.add_argument('-l', '--lowerEdgeDistanceBoundary', type=float, default=1, help='Edges are counted as best below this reprojected edge distance.')
    parser.add_argument('-u', '--upperEdgeDistanceBoundary', type=float, default=5, help='Edges are counted as worse above this reprojected edge distance.')

    return parser.parse_args()


def displayProgress(val: float = None):
    print('Progress: %.2f %%' % val, '     \r', end='')


def main() -> None:
    '''
    Main function. Parse, check input parameter and process data augmentation.
    '''
    try:
        args = parseArgs()
        args = checkInputParameter(args)
        print(Utilities.argsToStr(args))

        startTime = time.time()
        logging.info('Loading data from camera calibration file.')
        camera = Camera()
        camera.loadFromFile(args.camCalibFile)

        logging.info('Loading data from associated ground truth file.')
        gtHandler = TumGroundTruthHandler()
        gtHandler.progress = displayProgress
        gtHandler.load(args.groundTruthFile)

        logging.info('Starting edge matching.')
        edgeMatcher = EdgeMatcher(camera)
        edgeMatcher.setFrameOffset(args.frameOffset)
        edgeMatcher.setEdgeDistanceBoundaries(args.lowerEdgeDistanceBoundary, args.upperEdgeDistanceBoundary)

        for a in gtHandler.data():
            logging.info('Loading frame at timestamp %f' % (a.gt.timestamp))

            rgb = cv2.imread(os.path.join(args.rgbDir, a.rgb), cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(os.path.join(args.depthDir, a.depth), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(os.path.join(args.maskDir, a.rgb), cv2.IMREAD_GRAYSCALE)

            frame = Frame()
            frame.rgb = rgb
            frame.depth = depth
            frame.mask = mask
            frame.setT(a.gt.q, a.gt.t)

            meaningfulEdges = edgeMatcher.reprojectEdgesFromConsecutiveFrameSet(frame)

        elapsedTime = time.time() - startTime
        print('\n')
        logging.info('Finished in %.4f sec' % (elapsedTime))
        sys.exit(0)
    except Exception as e:
        logging.error(e)
        sys.exit(-1)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:\t%(message)s', level=logging.DEBUG)
    main()
