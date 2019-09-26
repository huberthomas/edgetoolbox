import logging
import argparse
import sys
import os
import cv2 as cv
import time
import multiprocessing as mp
from edgelib import Utilities
from edgelib import ImageProcessing
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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

    if args.cols <= 0:
        raise ValueError('Invalid cols.')

    if args.rows < 0:
        raise ValueError('Invalid rows.')

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
    parser = argparse.ArgumentParser(description='Reconstruct undefined areas in a depth image.')
    parser.add_argument('-i', '--inputDir', type=str, default=None, required=True, help='Input image directory.')
    parser.add_argument('-o', '--outputDir', type=str, default=None, required=True, help='Image output directory.')
    parser.add_argument('-c', '--cols', type=int, default=0, help='Set the new width.')
    parser.add_argument('-r', '--rows', type=int, default=0, help='Set the new height.')
    parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(), help='Number of spawned threads to process data. Default is maximum number.')

    return parser.parse_args()


def processImage(imgFilePath: str = None,
                 outFilePath: str = None,
                 cols: int = 0,
                 rows: int = 0) -> None:
    '''
    Extend image canvas by mirroring the image.

    img Input image.

    cols New width.

    rows New height.

    Returns extended image with new dimensions.    
    '''
    try:
        img = cv.imread(imgFilePath, cv.IMREAD_UNCHANGED)
        extendedImg = ImageProcessing.extendImageCanvas(img, cols, rows)

        # print(outFilePath)
        # fig = plt.figure(figsize=(1, 2))
        # fig.add_subplot(1,2,1)
        # plt.imshow(img)
        # fig.add_subplot(1,2,2)
        # plt.imshow(extendedImg)
        # plt.show()

        if extendedImg is not None:
            cv.imwrite(outFilePath, extendedImg)

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
                          args.cols, 
                          args.rows))

        pool = mp.Pool(processes=args.threads)

        startTime = time.time()
        pool.starmap(processImage, param)
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
