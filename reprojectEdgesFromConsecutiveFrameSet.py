import logging
import argparse
import time
import os
import sys
import cv2 as cv
from edgelib.TumGroundTruthHandler import TumGroundTruthHandler
from edgelib.EdgeMatcher import EdgeMatcher, EdgeMatcherMode, EdgeMatcherFrame
from edgelib.Camera import Camera
from edgelib import Utilities
import matplotlib.pyplot as plt


def checkInputParameter(args: any) -> any:
    '''
    Check the input parameter from argparse.

    args Arguments.

    Returns parsed arguments. Throws exception if error occurs.
    '''

    if args.rgbDir == None or not os.path.exists(args.rgbDir):
        raise ValueError('Invalid RGB image directory "%s".' % (args.rgbDir))

    if args.depthDir == None or not os.path.exists(args.depthDir):
        raise ValueError('Invalid depth image directory "%s".' % (args.depthDir))

    if args.maskDir == None or not os.path.exists(args.maskDir):
        raise ValueError('Invalid mask image directory "%s".' % (args.maskDir))

    if args.groundTruthFile == None or not os.path.exists(args.groundTruthFile):
        raise ValueError('Invalid ground truth file "%s".' % (args.groundTruthFile))

    if args.camCalibFile == None or not os.path.exists(args.camCalibFile):
        raise ValueError('Invalid camera calibration file "%s".' % (args.camCalibFile))

    if args.outputDir == None or len(args.outputDir) == 0:
        raise ValueError('Invalid output file "%s".' % (args.outputDir))

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

    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)
        logging.info('Created output directory "%s".' % (args.outputDir))

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
    parser.add_argument('-o', '--outputDir', type=str, default=None, required=True, help='Result output directory.')
    parser.add_argument('-f', '--frameOffset', type=int, default=1, help='Frame offset. Offset between the reprojected frames.')
    parser.add_argument('-l', '--lowerEdgeDistanceBoundary', type=float, default=1, help='Edges are counted as best below this reprojected edge distance.')
    parser.add_argument('-u', '--upperEdgeDistanceBoundary', type=float, default=5, help='Edges are counted as worse above this reprojected edge distance.')
    parser.add_argument('-p', '--projectionMode', type=int, choices=[EdgeMatcherMode.REPROJECT, EdgeMatcherMode.BACKPROJECT, EdgeMatcherMode.CENTERPROJECT],
                        default=1, help='Set the frame projection mode. 1 is backprojection, 2 is reprojection and 3 is center frame projection. Default is 1.')

    return parser.parse_args()


def main() -> None:
    '''
    Main function. Parse, check input parameter and process data.
    '''
    try:
        args = parseArgs()
        args = checkInputParameter(args)
        print(Utilities.argsToStr(args))

        # write configuration to output directory
        f = open(os.path.join(args.outputDir, 'settings.txt'), 'w')
        f.write(Utilities.argsToStr(args))
        f.close()

        startTime = time.time()
        logging.info('Loading data from camera calibration file.')
        camera = Camera()
        camera.loadFromFile(args.camCalibFile)

        logging.info('Loading data from associated ground truth file.')
        gtHandler = TumGroundTruthHandler()
        gtHandler.load(args.groundTruthFile)

        if len(gtHandler.data()) < args.frameOffset:
            raise ValueError('Number of input data is "%s" but must be greater than the frame offset "%d".' % (len(gtHandler.data()), args.frameOffset))

        logging.info('Starting edge matching.')
        edgeMatcher = EdgeMatcher(camera)
        edgeMatcher.setFrameOffset(args.frameOffset)
        edgeMatcher.setEdgeDistanceBoundaries(args.lowerEdgeDistanceBoundary, args.upperEdgeDistanceBoundary)

        frameFileNames = []
        for a in gtHandler.data():
            logging.info('Loading frame at timestamp %f (RGB: %s)' % (a.gt.timestamp, a.rgb))

            # used for determining the correct filename, depending on the projection mode
            frameFileNames.append(a.rgb)

            rgb = cv.imread(os.path.join(args.rgbDir, a.rgb), cv.IMREAD_UNCHANGED)
            depth = cv.imread(os.path.join(args.depthDir, a.depth), cv.IMREAD_UNCHANGED)
            mask = cv.imread(os.path.join(args.maskDir, a.rgb), cv.IMREAD_GRAYSCALE)

            frame = EdgeMatcherFrame()
            frame.uid = a.gt.timestamp
            frame.rgb = rgb
            frame.depth = depth
            frame.mask = mask
            frame.setT(a.gt.q, a.gt.t)
            
            meaningfulEdges = edgeMatcher.reprojectEdgesToConsecutiveFrameSet(frame, args.projectionMode, args.outputDir)

            if meaningfulEdges is None:
                continue

            # determine correct filename
            if args.projectionMode == EdgeMatcherMode.REPROJECT:
                frameFileName = frameFileNames[len(frameFileNames) - 1]
            elif args.projectionMode == EdgeMatcherMode.BACKPROJECT:
                frameFileName = frameFileNames[0]
            elif args.projectionMode == EdgeMatcherMode.CENTERPROJECT:
                frameFileName = frameFileNames[args.frameOffset]
            else:
                raise ValueError('Unknown projection mode "%d".' % (args.projectionMode))

            # save result
            cv.imwrite(os.path.join(args.outputDir, frameFileName), meaningfulEdges)
            logging.info('Saving "%s"' % (os.path.join(args.outputDir, frameFileName)))
            frameFileNames.pop(0)

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
