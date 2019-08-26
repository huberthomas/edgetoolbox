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
from edgelib import Canny
import numpy as np
'''
from edgelib import Canny
import numpy as np
baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz'
hha = cv.imread(os.path.join(baseDir, 'hha/1311867170.450076.png'))
depth = cv.cvtColor(cv.imread(os.path.join(baseDir, 'depth/1311867170.450076.png')), cv.COLOR_BGR2GRAY)
edge = Canny.canny(hha, 5, 10, 3, True, 5)
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
    parser.add_argument('-s', '--scale', type=int, choices=[0,1,2], default=0, help='Set the scale level. 0 = 1:1, 1 = 1:2, 2 = 1:4')

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

        for _ in range(0, args.scale):
            camera.setFx(camera.fx()/2.0)
            camera.setFy(camera.fy()/2.0)
            camera.setCx(camera.cx()/2.0)
            camera.setCy(camera.cy()/2.0)

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
        f = open(os.path.join(args.outputDir, 'records.txt'), 'w')
        f.write('# timestamp d<=%d %d<d<=%d d>%d\n'%(args.lowerEdgeDistanceBoundary, args.lowerEdgeDistanceBoundary, args.upperEdgeDistanceBoundary, args.upperEdgeDistanceBoundary))
        for a in gtHandler.data():
            logging.info('Loading frame at timestamp %f (RGB: %s)' % (a.gt.timestamp, a.rgb))

            # used for determining the correct filename, depending on the projection mode
            frameFileNames.append(a.rgb)

            rgb = cv.imread(os.path.join(args.rgbDir, a.rgb), cv.IMREAD_UNCHANGED)
            depth = cv.imread(os.path.join(args.depthDir, a.depth), cv.IMREAD_UNCHANGED)
            mask = cv.imread(os.path.join(args.maskDir, a.rgb), cv.IMREAD_GRAYSCALE)

            if args.inpaintDepth == 1:
                depth = ImageProcessing.reconstructDepthImg(depth, 5, cv.INPAINT_NS)
            elif args.inpaintDepth == 2:
                depth = ImageProcessing.reconstructDepthImg(depth, 5, cv.INPAINT_TELEA)

            for _ in range(0, args.scale):
                rgb = cv.resize(rgb, (0,0), fx=0.5, fy=0.5) #cv.pyrDown(rgb)
                depth = cv.resize(depth, (0,0), fx=0.5, fy=0.5) #cv.pyrDown(depth)
                mask = Canny.canny(rgb, 50, 100, 3, True, 3) #cv.resize(mask, (0,0), fx=0.5, fy=0.5) #cv.pyrDown(mask) 
                #_, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)

            frame = EdgeMatcherFrame()
            frame.uid = a.gt.timestamp
            frame.setRgb(rgb)
            frame.setDepth(depth)
            frame.setBoundaries(mask)
            frame.setT(a.gt.q, a.gt.t)

            meaningfulEdges, worseEdges = edgeMatcher.reprojectEdgesToConsecutiveFrameSet(frame, args.projectionMode, args.outputDir)

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
            numBest = len(meaningfulEdges[np.where(meaningfulEdges > 0)])
            numWorse = len(worseEdges[np.where(worseEdges > 0)])

            f.write('%s %d %d %d\n'%(frameFileName, numBest, 0, numWorse))
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
'''