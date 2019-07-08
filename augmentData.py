import argparse
import logging
import os
import multiprocessing as mp

from edgelib.DataAugmentation import DataAugmentation


def checkInputParameter(args):
    '''
    Check the input parameter from argparse.

    args Arguments.

    Returns parsed arguments.
    '''
    if args.inputDir == None or not os.path.exists(args.inputDir):
        raise ValueError('Invalid input directory.')

    # output directory
    if args.outputDir == None or len(args.outputDir) == 0:
        raise ValueError('Invalid output directory.')

    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)
        logging.info('Created output directory "%s".' % (args.outputDir))

    # scales
    if len(args.scales) == 1:
        args.scales = args.scales[0].split(',')

    args.scales = list(map(float, args.scales))
    args.scales = list(map(abs, args.scales))

    # angles
    if args.angles is not None:
        if len(args.angles) == 1:
            args.angles = args.angles[0].split(',')

        args.angles = list(map(float, args.angles))
        args.angles = list(map(abs, args.angles))

    # number of angels
    args.numberOfAngles = abs(args.numberOfAngles)

    if args.numberOfAngles == 0:
        raise ValueError(
            'Invalid number of angles. Must be a value greater than 0.')

    # threads
    args.threads = abs(args.threads)

    if args.threads > mp.cpu_count() or args.threads == 0:
        raise ValueError(
            'Invalid number of threads. Choose a value between 1 and %d.' % (mp.cpu_count()))

    return args


def main():
    '''
    Main function. Parse, check input parameter and process data augmentation.
    '''
    parser = argparse.ArgumentParser(
        description='Data augmentation from an input image directory.')
    parser.add_argument('-i', '--inputDir', type=str, default=None,
                        required=True, help='Input image directory.')
    parser.add_argument('-o', '--outputDir', type=str, default=None,
                        required=True, help='Augmented image output directory.')
    parser.add_argument('-s', '--scales', type=str, nargs='+', default='1',
                        help='Set generated scales.')
    parser.add_argument('-a', '--angles', type=str, nargs='+', default=None,
                        help='Data rotation angles in degrees.')
    parser.add_argument('-na', '--numberOfAngles', type=int, default=16,
                        help='Number of auto generated angles for rotating data. 360° will be split up to this number of angles.')
    parser.add_argument('-fh', '--flipHorizontal', default=False,
                        action='store_true', help='Flip data horizontally.')
    parser.add_argument('-fv', '--flipVertical', default=False,
                        action='store_true', help='Flip data vertically.')
    parser.add_argument('-c', '--cropBlackRotationBorder', default=True,
                        action='store_true', help='Crop out black rotation border.')
    parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(),
                        help='Number of spawned threads to process data. Default is maximum number.')

    try:
        args = checkInputParameter(parser.parse_args())

        aug = DataAugmentation(args.inputDir, args.outputDir)
        aug.setScales(args.scales)

        if args.angles is None:
            aug.setNumberOfAngles(args.numberOfAngles)
        else:
            aug.setAngles(args.angles)

        if args.flipHorizontal:
            aug.enableFlipHorizontal()

        if args.flipVertical:
            aug.enableFlipVertical()

        aug.setNumOfThreads(args.threads)
        aug.setCropBlackRotationBorder(args.cropBlackRotationBorder)
        aug.generateData()
    except Exception as e:
        logging.error(e)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:\t%(message)s', level=logging.DEBUG)
    print('#################')
    print('Data Augmentation')
    print('#################')
    main()