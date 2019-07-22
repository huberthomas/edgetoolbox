import logging
import argparse
import sys
import os
import cv2 as cv
import time
import multiprocessing as mp
from edgelib import Utilities
from edgelib import Canny


def checkInputParameter(args: any) -> any:
    '''
    Check the input parameter from argparse.

    args Arguments.

    Returns parsed arguments. Throws exception if error occurs.
    '''
    # input directory
    if args.inputDir == None or not os.path.exists(args.inputDir):
        raise ValueError('Invalid input directory.')

    # output directory
    if args.outputDir == None or len(args.outputDir) == 0:
        raise ValueError('Invalid output directory.')

    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)
        logging.info('Created output directory "%s".' % (args.outputDir))

    if args.threshold1 < 0 or args.threshold2 < 0:
        raise ValueError('Threshold must be greater than 0.')

    if args.threshold2 < args.threshold1:
        raise ValueError('Threshold 2 must be greater than threshold 1.')

    if args.kernelSize % 2 == 0 or args.kernelSize < 3:
        raise ValueError('Wrong kernel size. Allowed are 3, 5, 7, ...')

    if args.blurKernelSize % 2 == 0 or args.blurKernelSize < 3:
        raise ValueError('Wrong blur kernel size. Allowed are 3, 5, 7, ...')

    # threads
    args.threads = abs(args.threads)

    if args.threads > mp.cpu_count() or args.threads == 0:
        raise ValueError(
            'Invalid number of threads. Choose a value between 1 and %d.' % (mp.cpu_count()))

    return args


def parseArgs() -> any:
    '''
    Parse user arguments.
    '''
    parser = argparse.ArgumentParser(description='Canny edge detection from an input image directory.')
    parser.add_argument('-i', '--inputDir', type=str, default=None, required=True, help='Input image directory.')
    parser.add_argument('-o', '--outputDir', type=str, default=None, required=True, help='Image output directory.')
    parser.add_argument('-k', '--kernelSize', type=int, default=3, help='Set the Sobel kernel size. Default 3.')
    parser.add_argument('-bk', '--blurKernelSize', type=int, default=3, help='Set the blur kernel size. Default 3.')
    parser.add_argument('-t1', '--threshold1', type=int, default=100, help='First threshold for the hysteresis process. Default 100.')
    parser.add_argument('-t2', '--threshold2', type=int, default=150, help='Second threshold for the hysteresis process. Default 150')
    parser.add_argument('-ha', '--highAccuracy', default=True, action='store_true', help='High accuracy flag. Default true.')
    parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(), help='Number of spawned threads to process data. Default is maximum number.')

    return parser.parse_args()


def processAndSaveCanny(imgFilePath: str = None,
                        outFilePath: str = None,
                        threshold1: int = 100,
                        threshold2: int = 150,
                        kernelSize: int = 3,
                        highAccuracy: bool = True,
                        blurKernelSize: int = 3) -> None:
    '''
    Process Canny edge detection on a defined input image.

    imgFilePath Input image file path.

    outFilePath Output image file path.

    threshold1 First threshold for the hysteresis procedure.

    threshold2 Second threshold for the hysteresis procedure.

    kernelSize Kernel size for the sobel operator.

    highAccuracy If true, L2 gradient will be used for more accuracy.

    blurKernelSize Kernel size for the Sobel operator.
    '''
    try:
        img = cv.imread(imgFilePath)
        edge = Canny.canny(img, threshold1, threshold2, kernelSize, highAccuracy, blurKernelSize)
        cv.imwrite(outFilePath, edge)
    except Exception as e:
        raise e


def main() -> None:
    '''
    Main function. Parse, check input parameter and process data augmentation.
    '''
    try:
        args = parseArgs()
        args = checkInputParameter(args)
        print(Utilities.argsToStr(args))

        imgFileNames = Utilities.getFileNames(args.inputDir, ['png', 'jpg', 'jpeg'])

        param = []

        logging.info('Processing %d image(s).' % (len(imgFileNames)))
        
        for imgFileName in imgFileNames:
            imgFilePath = os.path.join(args.inputDir, imgFileName)
            fileName = imgFileName.split('.')
            fileName = fileName[:len(fileName)-1]
            fileName = '.'.join(fileName) + '.png'
            outFilePath = os.path.join(args.outputDir, fileName)

            param.append((imgFilePath,
                          outFilePath,
                          args.threshold1,
                          args.threshold2,
                          args.kernelSize,
                          args.highAccuracy,
                          args.blurKernelSize))            

        pool = mp.Pool(processes=args.threads)

        startTime = time.time()
        pool.starmap(processAndSaveCanny, param)
        elapsedTime = time.time() - startTime

        pool.terminate()

        # write configuration to output directory
        f = open(os.path.join(args.outputDir, 'settings.txt'), 'w')
        f.write(Utilities.argsToStr(args))
        f.write('\nImages\t%d' % (len(imgFileNames)))
        f.write('\nProcessing time\t%.4f sec' % (elapsedTime))
        f.close()
        
        logging.info('Finished in %.4f sec' % (elapsedTime))
        sys.exit(0)
    except Exception as e:
        logging.error(e)
        sys.exit(-1)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:\t%(message)s', level=logging.DEBUG)
    main()
