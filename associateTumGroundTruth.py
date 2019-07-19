import logging
import argparse
import time
import os
import sys
from edgelib.TumGroundTruthHandler import TumGroundTruthHandler
from edgelib import Utilities

def checkInputParameter(args: any) -> any:
    '''
    Check the input parameter from argparse.

    args Arguments.

    Returns parsed arguments. Throws exception if error occurs.
    '''
    if args.groundTruth == None or not os.path.exists(args.groundTruth):
        raise ValueError('Invalid ground truth file.')

    if args.rgbDir == None or not os.path.exists(args.rgbDir):
        raise ValueError('Invalid RGB image directory.')

    if args.depthDir == None or not os.path.exists(args.depthDir):
        raise ValueError('Invalid depth image directory.')

    if args.outputFile == None or len(args.outputFile) == 0:
        raise ValueError('Invalid output file.')

    if args.maxDifference < 0:
        args.maxDifference = abs(args.maxDifference)

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
    parser.add_argument('-g', '--groundTruth', type=str, default=None, required=True, help='TUM ground truth file, e.g. groundtruth.txt.')
    parser.add_argument('-r', '--rgbDir', type=str, default=None, required=True, help='RGB image directory.')    
    parser.add_argument('-d', '--depthDir', type=str, default=None, required=True, help='Depth image directory.')
    parser.add_argument('-o', '--outputFile', type=str, default=None, required=True, help='Result associated ground truth file.')    
    parser.add_argument('-m', '--maxDifference', type=float, default=0.2, help='Maximum difference between time entries for synchronization of files.')    

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
        gtHandler = TumGroundTruthHandler()
        gtHandler.progress = displayProgress
        gtHandler.associate(args.groundTruth, args.rgbDir, args.depthDir, args.maxDifference)
        gtHandler.save(args.outputFile)
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
