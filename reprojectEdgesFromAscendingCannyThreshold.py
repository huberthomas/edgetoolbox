import logging
import argparse
import sys
import os
import cv2 as cv
import time
import multiprocessing as mp
from edgelib import Utilities
from edgelib import Canny
from edgelib.TumGroundTruthHandler import TumGroundTruthHandler
from edgelib.EdgeMatcher import EdgeMatcher, EdgeMatcherMode, EdgeMatcherFrame
from edgelib.Camera import Camera


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

    if args.threshold1 < 0 or args.threshold2 < 0:
        raise ValueError('Threshold must be greater than 0.')

    if args.threshold2 < args.threshold1:
        raise ValueError('Threshold 2 must be greater than threshold 1.')

    if args.kernelSize % 2 == 0 or args.kernelSize < 3:
        raise ValueError('Wrong kernel size. Allowed are 3, 5, 7, ...')

    if args.blurKernelSize % 2 == 0 or args.blurKernelSize < 3:
        raise ValueError('Wrong blur kernel size. Allowed are 3, 5, 7, ...')

    if args.stepRange <= 0:
        raise ValueError('Invalid step. Must be greater than 0.')

    if args.validEdgesThreshold < 0 or args.validEdgesThreshold > 1:
        raise ValueError('Invalid threshold. Must be between 0 and less or equal 1.')

    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)
        logging.info('Created output directory "%s".' % (args.outputDir))

    return args


def parseArgs() -> any:
    '''
    Parse user arguments.
    '''
    parser = argparse.ArgumentParser(description='Canny edge detection from an input image directory.')
    parser.add_argument('-r', '--rgbDir', type=str, default=None, required=True, help='RGB image directory.')
    parser.add_argument('-d', '--depthDir', type=str, default=None, required=True, help='Depth image directory.')
    parser.add_argument('-m', '--maskDir', type=str, default=None, required=True, help='Mask image directory.')
    parser.add_argument('-c', '--camCalibFile', type=str, default=None, required=True, help='Camera calibration file.')
    parser.add_argument('-g', '--groundTruthFile', type=str, default=None, required=True, help='Associated TUM ground truth file, e.g. groundtruth_associated.txt.')
    parser.add_argument('-o', '--outputDir', type=str, default=None, required=True, help='Result output directory.')
    parser.add_argument('-f', '--frameOffset', type=int, default=1, help='Frame offset. Offset between the reprojected frames.')
    parser.add_argument('-k', '--kernelSize', type=int, default=3, help='Set the Sobel kernel size. Default 3.')
    parser.add_argument('-bk', '--blurKernelSize', type=int, default=3, help='Set the blur kernel size. Default 3.')
    parser.add_argument('-t1', '--threshold1', type=int, default=0, help='First threshold for the hysteresis process. Default 100.')
    parser.add_argument('-t2', '--threshold2', type=int, default=0, help='Second threshold for the hysteresis process. Default 150')
    parser.add_argument('-ha', '--highAccuracy', default=True, action='store_true', help='High accuracy flag. Default true.')
    parser.add_argument('-s', '--stepRange', type=int, default=50, help='Marching step range for the second threshold.')
    parser.add_argument('-v', '--validEdgesThreshold', type=float, default=0.5, help='Threshold that defines valid edges.')

    return parser.parse_args()


def main() -> None:
    '''
    Main function. Parse, check input parameter and process data augmentation.
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

            meaningfulEdges = edgeMatcher.reprojectEdgesByAscendingCannyThreshold(frame,
                                                                                  args.stepRange,
                                                                                  args.validEdgesThreshold,
                                                                                  args.threshold1,
                                                                                  args.threshold2,
                                                                                  args.kernelSize,
                                                                                  args.highAccuracy,
                                                                                  args.blurKernelSize)

            if meaningfulEdges is None:
                continue

            frameFileName = frameFileNames[len(frameFileNames) - 1]

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
