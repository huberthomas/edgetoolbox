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

    if args.gtDir == None or not os.path.exists(args.gtDir):
        raise ValueError('Invalid groundtruth directory.')

    # output directory
    if args.outputDir == None or len(args.outputDir) == 0:
        raise ValueError('Invalid output directory.')

    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)
        logging.info('Created output directory "%s".' % (args.outputDir))

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
    parser.add_argument('-gt', '--gtDir', type=str, default=None, required=True, help='Groundtruth image directory.')
    parser.add_argument('-o', '--outputDir', type=str, default=None, required=True, help='Image output directory.')
    parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(), help='Number of spawned threads to process data. Default is maximum number.')

    return parser.parse_args()


def processAndSaveReconstructedDepthImg(imgFilePath: str = None,
                                        gtFilePath: str = None,
                                        outFilePath: str = None) -> None:
    '''
    Process input file by reconstructing invalid/undefined depth values.

    imgFilePath Input image file path.

    gtFilePath Groundtruth image file path.

    outFilePath Output image file path.

    '''
    try:
        inputImg = cv.imread(imgFilePath, cv.IMREAD_UNCHANGED)
        inputImg = ImageProcessing.hysteresis(inputImg, 0.5, 0.75)
        _, inputImg = cv.threshold(inputImg, 0, 1, cv.THRESH_BINARY)
        inputImg, _ = ImageProcessing.removeIsolatedPixels(inputImg.astype(np.uint8), 100)

        gtImg = cv.imread(gtFilePath, cv.IMREAD_UNCHANGED)

        h, w = inputImg.shape[:2]
        result = np.full((h, w, 3), 255, np.uint8)
        # TP
        result[np.where(np.bitwise_and(inputImg > 0, gtImg > 0))] = [0, 200, 0]
        # FP
        result[np.where(np.bitwise_and(inputImg > 0, gtImg == 0))] = [0, 0, 255]
        # FN
        result[np.where(np.bitwise_and(inputImg == 0, gtImg > 0))] = [255, 0, 0]

        # fig = plt.figure(figsize=(1, 2))
        # fig.add_subplot(3,1,1)
        # plt.axis('off')
        # plt.imshow(cv.bitwise_not(gtImg), cmap='gray')
        # fig.add_subplot(3,1,2)
        # plt.axis('off')
        # plt.imshow(cv.bitwise_not(inputImg), cmap='gray')
        # fig.add_subplot(3,1,3)
        # plt.axis('off')
        # plt.imshow(result)
        # plt.show()

        #cv.imwrite(outFilePath, cv.bitwise_not(inputImg*255))
        #cv.imwrite(outFilePath, cv.bitwise_not(gtImg*255))
        cv.imwrite(outFilePath, result)
    except Exception as e:
        raise e


def main() -> None:
    '''
    Main function. Parse, check input parameter and process data augmentation.

    python precisionRecallViewer.py -i '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/bsr_bsds500/BSR/BSDS500/data/images/test_png/bsds500mixaug_edge_preserved_otsu_aug-nms' -gt '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/bsr_bsds500/BSR/BSDS500/data/groundTruth/test' -o '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/bsr_bsds500/BSR/BSDS500/data/images/test_png/deleteme' -t 1
    python precisionRecallViewer.py -i '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/bsr_bsds500/BSR/BSDS500/data/groundTruth/test' -gt '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/bsr_bsds500/BSR/BSDS500/data/groundTruth/test' -o '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/bsr_bsds500/BSR/BSDS500/data/images/test_png/deleteme' -t 1
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
            gtFilePath = os.path.join(args.gtDir, fileName)
            param.append((imgFilePath,
                          gtFilePath,
                          outFilePath))

        pool = mp.Pool(processes=args.threads)

        startTime = time.time()
        pool.starmap(processAndSaveReconstructedDepthImg, param)
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
