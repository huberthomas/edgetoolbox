import logging
import argparse
import sys
import os
import cv2 as cv
import time
import numpy as np
from edgelib import Utilities
from edgelib import ImageProcessing

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
    if args.outputFile == None or len(args.outputFile) == 0:
        raise ValueError('Invalid output directory.')

    return args


def parseArgs() -> any:
    '''
    Parse user arguments.
    '''
    parser = argparse.ArgumentParser(description='Reconstruct undefined areas in a depth image.')
    parser.add_argument('-i', '--inputDir', type=str, default=None, required=True, help='Input image directory.')
    parser.add_argument('-o', '--outputFile', type=str, default=None, required=True, help='Output text file.')

    return parser.parse_args()

def main() -> None:
    '''
    Main function. Parse, check input parameter and process data augmentation.
    '''
    try:
        args = parseArgs()
        args = checkInputParameter(args)
        print(Utilities.argsToStr(args))

        baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/all_thinned'
        subDir = [
            'rgbd_dataset_freiburg1_360',
            'rgbd_dataset_freiburg1_desk',
            'rgbd_dataset_freiburg1_desk2',
            'rgbd_dataset_freiburg1_floor',
            'rgbd_dataset_freiburg1_plant',
            'rgbd_dataset_freiburg1_room',
            'rgbd_dataset_freiburg1_rpy',
            'rgbd_dataset_freiburg1_teddy',
            'rgbd_dataset_freiburg1_xyz',
            'rgbd_dataset_freiburg2_360_hemisphere',
            'rgbd_dataset_freiburg2_coke',
            'rgbd_dataset_freiburg2_desk',
            'rgbd_dataset_freiburg2_desk_with_person',
            'rgbd_dataset_freiburg2_dishes',
            'rgbd_dataset_freiburg2_flowerbouquet',
            'rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
            'rgbd_dataset_freiburg2_large_no_loop',
            'rgbd_dataset_freiburg2_metallic_sphere',
            'rgbd_dataset_freiburg2_metallic_sphere2',
            'rgbd_dataset_freiburg2_pioneer_360',
            'rgbd_dataset_freiburg2_pioneer_slam',
            'rgbd_dataset_freiburg2_xyz',
            'rgbd_dataset_freiburg3_cabinet',
            'rgbd_dataset_freiburg3_large_cabinet',
            'rgbd_dataset_freiburg3_long_office_household',
            'rgbd_dataset_freiburg3_nostructure_texture_far',
            'rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
            'rgbd_dataset_freiburg3_sitting_static',
            'rgbd_dataset_freiburg3_structure_notexture_far',
            'rgbd_dataset_freiburg3_structure_notexture_near',
            'rgbd_dataset_freiburg3_structure_texture_far',
            'rgbd_dataset_freiburg3_structure_texture_near',
            'rgbd_dataset_freiburg3_teddy',
            'rgbd_dataset_freiburg3_walking_xyz',
        ]
        edgeDirs = [
            'level0/canny',
            'level0/bdcn',
            'level0/bdcn_sgd_singleScale_gpu_tum_30k_augplus',
        ]

        for inputDir in subDir:

            for edgeDir in edgeDirs:
                args.inputDir = os.path.join(baseDir, inputDir, edgeDir)
                args.outputFile = os.path.join(baseDir, '%s.txt'%(inputDir + '_' + edgeDir.replace('/', '_file_')))
                imgFileNames = Utilities.getFileNames(args.inputDir, ['png', 'jpg', 'jpeg'])
                
                f = open(os.path.join(args.outputFile), 'w')
                f.write('filename;numEdges;total;numEdges_%\n')
                countedEdges = []
                for imageFileName in imgFileNames:
                    img = cv.imread(os.path.join(args.inputDir, imageFileName), flags=cv.IMREAD_GRAYSCALE)
                    h, w = img.shape[:2]
                    total = h*w
                    numEdges = len(np.nonzero(img)[0])
                    countedEdges.append(numEdges)
                    # print(numEdges, total, (numEdges/total)*100)
                    f.write('%s;%d;%d;%f\n'%(imageFileName, numEdges, total, (numEdges/total)*100))

                # f.write('dataset;mean;median;std;min;max\n')
                # f.write('%s;%f;%f;%f;%f;%f\n'%(args.outputFile, np.mean(countedEdges)/total, np.median(countedEdges)/total, np.std(countedEdges)/total, np.min(countedEdges)/total, np.max(countedEdges)/total))
                print('dataset;mean;median;std;min;max')
                print('%s;%f;%f;%f;%f;%f'%(args.outputFile, np.mean(countedEdges)/total, np.median(countedEdges)/total, np.std(countedEdges)/total, np.min(countedEdges)/total, np.max(countedEdges)/total))
                f.close()

    except Exception as e:
        logging.error(e)
        sys.exit(-1)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:\t%(message)s', level=logging.DEBUG)
    main()
