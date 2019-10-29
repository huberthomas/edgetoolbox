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
    if args.groundTruthFile is None or not os.path.exists(args.groundTruthFile):
        raise ValueError('Invalid ground truth file.')

    if args.rgbDir is None or not os.path.exists(args.rgbDir):
        raise ValueError('Invalid RGB image directory.')

    if args.depthDir is None or not os.path.exists(args.depthDir):
        raise ValueError('Invalid depth image directory.')

    if args.outputFile is None or len(args.outputFile) == 0:
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
    parser.add_argument('-g', '--groundTruthFile', type=str, default=None, required=True, help='TUM ground truth file, e.g. groundtruth.txt.')
    parser.add_argument('-r', '--rgbDir', type=str, default=None, required=True, help='RGB image directory.')
    parser.add_argument('-d', '--depthDir', type=str, default=None, required=True, help='Depth image directory.')
    parser.add_argument('-o', '--outputFile', type=str, default=None, required=True, help='Result associated ground truth file.')
    parser.add_argument('-rf', '--rgbFile', type=str, default=None, required=False, help='RGB association file, e.g. rgb.txt.')
    parser.add_argument('-df', '--depthFile', type=str, default=None, required=False, help='Depth association file, e.g. depth.txt.')
    parser.add_argument('-m', '--maxDifference', type=float, default=0.2, help='Maximum difference between time entries for synchronization of files.')

    return parser.parse_args()


def displayProgress(val: float = None):
    print('Progress: %.2f %%' % val, '     \r', end='')


def main() -> None:
    '''
    Main function. Parse, check input parameter and process data augmentation.
    '''
    try:
        datasetBase = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets'

        subDir = [
            # 'rgbd_dataset_freiburg1_desk',
            # 'rgbd_dataset_freiburg1_desk2',
            # 'rgbd_dataset_freiburg1_plant',
            # 'rgbd_dataset_freiburg1_room',
            # 'rgbd_dataset_freiburg1_rpy',
            # 'rgbd_dataset_freiburg1_xyz',
            #'rgbd_dataset_freiburg2_desk',
            # 'rgbd_dataset_freiburg2_xyz',
            # 'rgbd_dataset_freiburg3_long_office_household',
            # 'rgbd_dataset_freiburg1_360',
            # 'rgbd_dataset_freiburg1_floor',
            # 'rgbd_dataset_freiburg1_teddy',
            # 'rgbd_dataset_freiburg2_360_hemisphere',
            # 'rgbd_dataset_freiburg2_coke',
            # 'rgbd_dataset_freiburg2_desk_with_person',
            # 'rgbd_dataset_freiburg2_dishes',
            # 'rgbd_dataset_freiburg2_flowerbouquet',
            # 'rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
            # 'rgbd_dataset_freiburg2_large_no_loop',
            # 'rgbd_dataset_freiburg2_metallic_sphere',
            # 'rgbd_dataset_freiburg2_metallic_sphere2',
            # 'rgbd_dataset_freiburg2_pioneer_360',
            # 'rgbd_dataset_freiburg2_pioneer_slam',
            # 'rgbd_dataset_freiburg3_cabinet',
            # 'rgbd_dataset_freiburg3_large_cabinet',
            # 'rgbd_dataset_freiburg3_nostructure_texture_far',
            # 'rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
            # 'rgbd_dataset_freiburg3_sitting_static',
            # 'rgbd_dataset_freiburg3_structure_notexture_far',
            # 'rgbd_dataset_freiburg3_structure_notexture_near',
            # 'rgbd_dataset_freiburg3_structure_texture_far',
            # 'rgbd_dataset_freiburg3_structure_texture_near',
            # 'rgbd_dataset_freiburg3_teddy',
            # 'rgbd_dataset_freiburg3_walking_xyz',
            #'icl_living_room_0',
            # 'icl_living_room_1',
            # 'icl_living_room_2',
            # 'icl_living_room_3',
            # 'icl_office_0',
            # 'icl_office_1',
            # 'icl_office_2',
            # 'icl_office_3'
            # 'eth3d_cables_1',
            # 'eth3d_cables_2',
            # 'eth3d_einstein_1',
            # 'eth3d_einstein_2',
            # 'eth3d_einstein_global_light_changes_2',
            # 'eth3d_mannequin_3',
            # 'eth3d_mannequin_4',
            # 'eth3d_mannequin_face_1',
            # 'eth3d_mannequin_face_2',
            # 'eth3d_planar_1',
            # 'eth3d_planar_2',
            # 'eth3d_plant_scene_1',
            # 'eth3d_plant_scene_2',
            # 'eth3d_rgbd_dataset_freiburg1_desk2_clean',
            # 'eth3d_sfm_bench',
            # 'eth3d_sofa_1',
            # 'eth3d_sofa_2',
            # 'eth3d_table_3',
            # 'eth3d_table_4',
            # 'eth3d_table_7',
        ]

        args = parseArgs()
        for i in range(0, len(subDir)):
            args.groundTruthFile = os.path.join(datasetBase, subDir[i], 'groundtruth.txt')
            args.rgbDir = os.path.join(datasetBase, subDir[i], 'rgb')
            args.depthDir = os.path.join(datasetBase, subDir[i], 'depth')
            args.outputFile = os.path.join(datasetBase, subDir[i], 'groundtruth_associated.txt')

            #args.rgbFile = os.path.join(datasetBase, subDir[i], 'rgb.txt')
            #args.depthFile = os.path.join(datasetBase, subDir[i], 'depth.txt')

            args = checkInputParameter(args)
            print(Utilities.argsToStr(args))

            startTime = time.time()
            gtHandler = TumGroundTruthHandler()
            gtHandler.progress = displayProgress
            gtHandler.associate(args.groundTruthFile, args.rgbDir, args.depthDir, args.maxDifference, args.rgbFile, args.depthFile)
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
