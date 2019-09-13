import logging
import argparse
import time
import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt
from edgelib.TumGroundTruthHandler import TumGroundTruthHandler
from edgelib.EdgeMatcher import EdgeMatcher, EdgeMatcherMode, EdgeMatcherFrame
from edgelib.Camera import Camera
from edgelib import Utilities
from edgelib import ImageProcessing
import numpy as np
'''
from edgelib import ImageProcessing
import numpy as np
baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz'
hha = cv.imread(os.path.join(baseDir, 'hha/1311867170.450076.png'))
depth = cv.cvtColor(cv.imread(os.path.join(baseDir, 'depth/1311867170.450076.png')), cv.COLOR_BGR2GRAY)
edge = ImageProcessing.canny(hha, 5, 10, 3, True, 5)
nanEdge = edge.copy()
edge[np.where(depth==0)] = 0

fig = plt.figure(1)
fig.suptitle('Gray Blurred')
plt.subplot(221)
plt.axis('off')
plt.title('HHA')
plt.imshow(cv.cvtColor(hha, cv.COLOR_BGRA2RGBA))
plt.subplot(222)
plt.axis('off')
plt.title('Canny')
plt.imshow(edge, cmap='gray')
plt.subplot(223)
plt.axis('off')
plt.title('Depth')
plt.imshow(depth, cmap='jet')
plt.subplot(224)
plt.axis('off')
plt.title('Depth')
plt.imshow(nanEdge, cmap='gray')
plt.show()
exit(0)
'''


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
    parser.add_argument('-id', '--inpaintDepth', type=int, choices=[0, 1, 2], default=0,
                        help='Fill out undefined depth regions by inpainting. 0 ... off, 1 ... CV_INPAINT_NS, 2 ... CV_INPAINT_TELEA. Default is 0.')
    parser.add_argument('-s', '--scales', type=int, choices=[0, 1, 2], default=0, help='Set the scale level. 0 = 1:1, 1 = 1:2, 2 = 1:4')
    parser.add_argument('-mipa', '--minIsolatedPixelArea', type=int, default=0, help='Minimum isolated pixel areas. Cleans contours less or equal this value.')

    return parser.parse_args()


def main() -> None:
    '''
    Main function. Parse, check input parameter and process data.
    '''
    try:
        args = parseArgs()
        args = checkInputParameter(args)

        # python reprojectEdgesFromConsecutiveFrameSet.py  -c  -o  -p 3 -f 3 -l 2 -u 2 -mipa 30

        allBase = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/all'
        trainRgbBase = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/mix'
        datasetBase = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets'

        subDir = [
            # 'rgbd_dataset_freiburg1_desk',
            # 'rgbd_dataset_freiburg1_desk2',
            # 'rgbd_dataset_freiburg1_plant',
            # 'rgbd_dataset_freiburg1_room',
            # 'rgbd_dataset_freiburg1_rpy',
            # 'rgbd_dataset_freiburg1_xyz',
            'rgbd_dataset_freiburg2_desk',
            'rgbd_dataset_freiburg2_xyz',
            'rgbd_dataset_freiburg3_long_office_household',
        ]

        for i in range(0, len(subDir)):
            args.rgbDir = os.path.join(trainRgbBase, 'rgb', subDir[i])
            args.depthDir = os.path.join(datasetBase, subDir[i], 'depth')
            args.maskDir = os.path.join(allBase, subDir[i], 'level0/canny')
            args.camCalibFile = os.path.join(datasetBase, subDir[i], 'camera_calib_schenk.yml')
            args.groundTruthFile = os.path.join(datasetBase, subDir[i], 'groundtruth_associated.txt')
            args.outputDir = os.path.join(trainRgbBase, 'gt_tmp', subDir[i])
            args.frameOffset = 3#4
            args.lowerEdgeDistanceBoundary = 3
            args.upperEdgeDistanceBoundary = args.lowerEdgeDistanceBoundary
            args.projectionMode = 3
            args.inpaintDepth = 2
            args.minIsolatedPixelArea = 30
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
            edgeMatcher.setMinIsolatedPixelArea(args.minIsolatedPixelArea)

            frameFileNames = []
            f = open(os.path.join(args.outputDir, 'records.txt'), 'w')
            f.write('# timestamp d<=%d %d<d<=%d d>%d\n' % (args.lowerEdgeDistanceBoundary, args.lowerEdgeDistanceBoundary, args.upperEdgeDistanceBoundary, args.upperEdgeDistanceBoundary))

            for a in gtHandler.data():
                if not os.path.exists(os.path.join(args.rgbDir, a.rgb)):
                    logging.info('Skipping %s. File not found.' % (a.rgb))
                    continue

                logging.info('Loading frame at timestamp %f (RGB: %s)' % (a.gt.timestamp, a.rgb))

                # used for determining the correct filename, depending on the projection mode
                frameFileNames.append(a.rgb)

                rgb = cv.imread(os.path.join(args.rgbDir, a.rgb), cv.IMREAD_UNCHANGED)
                depth = cv.imread(os.path.join(args.depthDir, a.depth), cv.IMREAD_UNCHANGED)
                #mask = cv.imread(os.path.join(args.maskDir, a.rgb), cv.IMREAD_GRAYSCALE)

                if args.inpaintDepth == 1:
                    depth = ImageProcessing.reconstructDepthImg(depth, 5, cv.INPAINT_NS)
                elif args.inpaintDepth == 2:
                    depth = ImageProcessing.reconstructDepthImg(depth, 5, cv.INPAINT_TELEA)

                frame = EdgeMatcherFrame()
                frame.uid = a.rgb  # a.gt.timestamp
                frame.setRgb(rgb)
                frame.setDepth(depth)
                #frame.setBoundaries(mask)
                frame.setT(a.gt.q, a.gt.t)

                start = time.time()
                resultFrame = edgeMatcher.reprojectEdgesToConsecutiveFrameSet(frame, args.projectionMode, args.outputDir)

                if resultFrame is None:
                    continue

                logging.info('Meaningful edges processed in %f sec.' % (time.time() - start))
                cleanedEdges = resultFrame.boundaries().copy()
                cleanedEdges[np.nonzero(resultFrame.refinedMeaningfulEdges)] = 0

                rgbCpy = resultFrame.rgb().copy()
                rgbCpy[np.nonzero(cleanedEdges)] = [255, 0, 0]
                rgbCpy[np.nonzero(resultFrame.refinedMeaningfulEdges)] = [0, 255, 0]

                fig = plt.figure(2)
                fig.suptitle('Meaningful Edges Results')
                fig.add_subplot(3, 1, 1)
                plt.axis('off')
                plt.title('Scaled Meaningful Edges')
                plt.imshow(resultFrame.scaledMeaningfulEdges, cmap='hot')

                fig.add_subplot(3, 1, 2)
                plt.axis('off')
                plt.title('Refined Meaningful Edges')
                plt.imshow(resultFrame.refinedMeaningfulEdges, cmap='hot')
                
                fig.add_subplot(3, 1, 3)
                plt.axis('off')
                plt.title('Good/Bad Edges')
                plt.imshow(rgbCpy, cmap='hot')
                plt.show()

                # save result
                meaningfulEdges = resultFrame.refinedMeaningfulEdges

                numBest = len(meaningfulEdges[meaningfulEdges > 0])
                numWorse = len(cleanedEdges[cleanedEdges > 0])

                frameFileName = resultFrame.uid
                f.write('%s %d %d %d\n' % (frameFileName, numBest, 0, numWorse))
                f.flush()
                cv.imwrite(os.path.join(args.outputDir, frameFileName), meaningfulEdges * 255)
                logging.info('Saving "%s"' % (os.path.join(args.outputDir, frameFileName)))
                frameFileNames.pop(0)

            elapsedTime = time.time() - startTime
            print('\n')
            logging.info('Finished in %.4f sec' % (elapsedTime))
            f.close()

        sys.exit(0)
    except Exception as e:
        logging.error(e)
        sys.exit(-1)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:\t%(message)s', level=logging.DEBUG)
    main()


'''
python reprojectEdgesFromConsecutiveFrameSet.py -g '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/groundtruth_associated.txt' -r '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/rgb' -d '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/depth_inpaint_cv_talea' -m '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/all/rgbd_dataset_freiburg2_xyz/level0/canny' -c '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/camera_calib_schenk.yml' -o '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/em_output' -p 3 -f 4 -l 1 -u 1
python reprojectEdgesFromConsecutiveFrameSet.py -g '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_xyz/groundtruth_associated.txt' -r '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_xyz/rgb' -d '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_xyz/depth' -m '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_xyz/rgb/bdcn_40k_thinned' -c '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_xyz/camera_calib_schenk.yml' -o '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_xyz/em_output' -p 3 -f 4 -l 1 -u 1 -id 2
python reprojectEdgesFromConsecutiveFrameSet.py -g '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_xyz/groundtruth_associated.txt' -r '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_xyz/rgb' -d '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_xyz/depth' -m '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_xyz/mask/canny' -c '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_xyz/camera_calib_schenk.yml' -o '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_xyz/em_output_canny' -p 3 -f 4 -l 1 -u 1 -id 2

python reprojectEdgesFromConsecutiveFrameSet.py -g '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/groundtruth_associated.txt' -r '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/rgb' -d '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/depth_inpaint_nans_m0' -m '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/all/rgbd_dataset_freiburg2_xyz/level0/canny' -c '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/camera_calib_schenk.yml' -o '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/em_output' -p 3 -f 4 -l 2 -u 2 -mipa 10
python reprojectEdgesFromConsecutiveFrameSet.py -g '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/groundtruth_associated.txt' -r '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/rgb' -d '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/depth_inpaint_nans_m0' -m '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/all/rgbd_dataset_freiburg2_xyz/level0/canny' -c '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/camera_calib_schenk.yml' -o '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/em_output' -p 3 -f 4 -l 2 -u 2 -mipa 10


python reprojectEdgesFromConsecutiveFrameSet.py -g '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_plant/groundtruth_associated.txt' -r '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/mix/rgb/rgbd_dataset_freiburg1_plant/' -d '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_plant/depth' -m '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/all/rgbd_dataset_freiburg1_plant/level0/canny' -c '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_plant/camera_calib_schenk.yml' -o '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/mix/gt/rgbd_dataset_freiburg1_plant' -p 3 -f 3 -l 2 -u 2 -mipa 30
'''
