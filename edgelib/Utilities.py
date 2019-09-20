import os
import cv2 as cv
import math
import glob
import fnmatch

from typing import List

'''
Utilities and helper functions.
'''


def createTrainList(baseDir: str = None, gtDirs: List[str] = None, rgbDirs: List[str] = None, outputFileName: str = 'train_pair.lst', supportedExtensions: list = ['png', 'jpg', 'jpeg']):
    '''
    Create a train list that is used by BDCN for training.

    inputDirs Directories that should be included in the training list.

    gtDirs Base directory where the images are read.

    rgbDirs RGB directories. Must be in the same order as the gtDirs.

    outputFileName Output file name, train.lst

    supportedExtensions These files are included in the training list.

    e.g.
    baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/mix'
    gtDirs = [
        'gt_edge_preserved/rgbd_dataset_freiburg2_desk',
        'gt_edge_preserved/rgbd_dataset_freiburg2_xyz',
        'gt_edge_preserved/rgbd_dataset_freiburg3_long_office_household',
        'bsds500/gt'
    ]
    rgbDirs = [
        'rgb/rgbd_dataset_freiburg2_desk',
        'rgb/rgbd_dataset_freiburg2_xyz',
        'rgb/rgbd_dataset_freiburg3_long_office_household',
        'bsds500/rgb'
    ]
    outputFileName = 'train_pair_bsds_mix.lst'

    if __name__ == '__main__':
        createTrainList(baseDir, gtDirs, rgbDirs, outputFileName, ['png'])
        
    '''
    if baseDir is None:
        raise ValueError('Invalid base directory.')

    if gtDirs is None:
        raise ValueError('Invalid ground truth directories.')

    if rgbDirs is None:
        raise ValueError('Invalid RGB directories.')

    if outputFileName is None:
        raise ValueError('Invalid train list output filename.')

    if len(gtDirs) != len(rgbDirs):
        raise ValueError('Groundtruth directories must have the same size as the RGB directories.')

    f = open(os.path.join(os.path.join(baseDir, outputFileName)), 'w')
    lenGtDirs = len(gtDirs)
    for i in range(0, lenGtDirs):
        gtDir = gtDirs[i]
        rgbDir = rgbDirs[i]

        fileNames = getFileNames(os.path.join(baseDir, gtDirs[i]), supportedExtensions)

        lenFileNames = len(fileNames)

        for j in range(0, lenFileNames):
            fileName = fileNames[j]
            f.write('%s %s' % (os.path.join(rgbDir, fileName), os.path.join(gtDir, fileName)))

            if j != lenFileNames - 1 or i != lenGtDirs - 1:
                f.write('\n')

    f.close()


def getFileNames(inputDir: str = None, supportedExtensions: list = ['png', 'jpg', 'jpeg']) -> List[str]:
    '''
    Get files e.g. (png, jpg, jpeg) from an input directory. It is case insensitive to the extensions.

    inputDir Input directory that contains images.

    supportedExtensions Only files with supported extensions are included in the final list. Case insensitive.

    Returns a list of images file names.
    '''
    if inputDir is None:
        raise ValueError('Input directory must be set.')

    if supportedExtensions is None or len(supportedExtensions) == 0:
        raise ValueError('Supported extensions must be set.')

    res = []

    dirList = os.listdir(inputDir)

    for extension in supportedExtensions:
        pattern = ''
        for char in extension:
            pattern += ('[%s%s]' % (char.lower(), char.upper()))

        res.extend(fnmatch.filter(dirList, '*.%s' % (pattern)))

    return res


def rotateImage(img: any = None, angle: float = None, removeCropBorders: bool = True) -> any:
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping

    img OpenCV image.

    angle Angle in degrees.
    """
    if img is None:
        raise ValueError('Image must be set.')

    height, width = img.shape[:2]  # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    imageCenter = (width/2, height/2)

    M = cv.getRotationMatrix2D(imageCenter, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    absCos = abs(M[0, 0])
    absSin = abs(M[0, 1])

    # find the new width and height bounds
    boundW = int(height * absSin + width * absCos)
    boundH = int(height * absCos + width * absSin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    M[0, 2] += boundW/2 - imageCenter[0]
    M[1, 2] += boundH/2 - imageCenter[1]

    # rotate image with the new bounds and translated rotation matrix
    rotatedImg = cv.warpAffine(img, M, (boundW, boundH))

    # remove black borders
    if removeCropBorders:
        x, y, bbW, bbH = getCropCoordinates(angle, width, height)
        c = list(map(int, [x, y, bbW, bbH]))
        rotatedImg = rotatedImg[c[1]:c[1]+c[3], c[0]:c[0]+c[2]]

    return rotatedImg


def transformAndSaveImage(outputFilePath: str = None,
                          img: any = None,
                          angle: float = 0.0,
                          scale: float = 1.0,
                          flipHorizontal: bool = False,
                          flipVertical: bool = False,
                          cropBlackBorder: bool = False) -> None:
    '''
    Rotates/flips an image by a defined angle and saves it to the output directory.

    img OpenCV image.

    angle Rotation angle.

    scale Rescale image.

    flipHorizontal Flip image horizontal.

    flipVertical Flip image vertical.

    cropBlackBorder Crop black border that can occur if an image is rotated.
    '''
    if img is None:
        raise ValueError('Image must be set.')

    if scale == 0:
        raise ValueError('Scale must be a value greater than 0.')

    if outputFilePath is None or len(outputFilePath) == 0:
        raise ValueError('Output file path must be set.')

    if flipHorizontal and flipVertical:
        img = cv.flip(img, -1)
    elif flipHorizontal:
        img = cv.flip(img, 0)
    elif flipVertical:
        img = cv.flip(img, 1)

    scale = abs(scale)
    img = cv.resize(img, None, fx=scale, fy=scale)

    rotatedImg = rotateImage(img, angle, cropBlackBorder)

    cv.imwrite(outputFilePath, rotatedImg)


def getCropCoordinates(angle: float = None, width: int = None, height: int = None) -> tuple:
    '''
    Rotation sometimes results in black borders around the image. This function calculates the
    cropping area to avoid black borders. For more information see
    https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

    angle Angle in degrees.

    width Width of the image.

    height Height of the image.

    Returns the cropping tuple (x, y, width, height). 
    '''
    if angle is None:
        raise ValueError('Angle is in degrees and must be set.')

    if width is None:
        raise ValueError('Width must be set.')

    if height is None:
        raise ValueError('Heigh must be set')

    ang = math.radians(angle)
    quadrant = math.floor(ang / (math.pi / 2)) & 3
    sign_alpha = ang if (quadrant & 1) == 0 else math.pi - ang
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb = {
        'w': width * math.cos(alpha) + height * math.sin(alpha),
        'h': width * math.sin(alpha) + height * math.cos(alpha)
    }

    gamma = math.atan2(bb['w'], bb['h']) if width < height else math.atan2(
        bb['h'], bb['w'])
    delta = math.pi - alpha - gamma

    length = height if width < height else width
    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return x, y, bb['w'] - 2 * x, bb['h'] - 2 * y


def argsToStr(args) -> None:
    '''
    Convert arguments to a string.

    args Arguments form input parser.
    '''
    param = '*'*80
    param += '\nParameter\n'
    param += '*'*80
    param += '\n'
    for x in args.__dict__:
        param += ('%s\t%s\n' % (x, str(args.__dict__[x])))
    param += '*'*80
    return param


def rescale(val: float = 0, minVal: float = 0, maxVal: float = 0, newMinVal: float = 0, newMaxVal: float = 0) -> float:
    '''
    Rescale value to new range.

    val Value to rescale.

    minVal Min range of val.

    maxVal Max range of val.

    newMinVal New min range.

    newMaxVal New max range.

    Returns rescaled value.
    '''
    return ((newMaxVal - newMinVal)/(maxVal - minVal)) * (val - maxVal) + newMaxVal


def createHtmlTileOutput(dirList: List[str] = [], htmlFilePath: str = None):
    '''
    Create tile view of images located in defined directory.

    dirList Stringlist of image directories.

    htmlFilePath HTML output file that contains tile view of found images.

    baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/test_dataset'
    subDirs = [
        os.path.join(baseDir, 'hdr_fusion', 'flicker_synthetic', 'flicker_1'),
        os.path.join(baseDir, 'hdr_fusion', 'smooth_synthetic', 'flicker_2'),
        os.path.join(baseDir, 'nyu_depth_v2', 'basements', 'basement_001c'),
        os.path.join(baseDir, 'nyu_depth_v2', 'cafe', 'cafe_0001c'),
        os.path.join(baseDir, 'nyu_depth_v2', 'classrooms', 'classroom_0014'),
        os.path.join(baseDir, 'tum', 'rgbd_dataset_freiburg1_desk', 'rgb'),
        os.path.join(baseDir, 'tum', 'rgbd_dataset_freiburg1_xyz', 'rgb'),
        os.path.join(baseDir, 'tum', 'rgbd_dataset_freiburg2_xyz', 'rgb'),
    ]
    htmlOutputFileNames = [
        'flicker_1',
        'flicker_2',
        'basement_001c',
        'cafe_0001c',
        'classroom_0014',
        'rgbd_dataset_freiburg1_desk',
        'rgbd_dataset_freiburg1_xyz',
        'rgbd_dataset_freiburg2_xyz',
    ]

    for i in range(0, len(subDirs)):
        baseSubDir = subDirs[i]
        inputDirs = [
            baseSubDir,
            os.path.join(baseSubDir, 'bdcn'),
            os.path.join(baseSubDir, 'bdcn_40k'),
            os.path.join(baseSubDir, 'bdcn_30k'),
            os.path.join(baseSubDir, 'bdcn_20k'),
            os.path.join(baseSubDir, 'bdcn_10k'),
            os.path.join(baseSubDir, 'bdcn_5k'),
            os.path.join(baseSubDir, 'bdcn_1k'),
        ]

        createHtmlTileOutput(inputDirs, os.path.join(baseDir, '%s.html'%(os.path.basename(htmlOutputFileNames[i]))))
    '''
    data = []
    counter = 0
    for i in range(0, len(dirList)):
        dirPath = dirList[i]
        fileNames = getFileNames(dirPath)
        counter = len(fileNames)

        data.append([])
        for fileName in fileNames:
            data[i].append(os.path.join(dirPath, fileName))

    th = '<tr>'
    for dirName in dirList:
        th += '<th>%s</th>' % (os.path.basename(dirName))
    th += '</tr>'

    tr = ''
    for i in range(0, counter):
        tr += '<tr>'
        for j in range(0, len(dirList)):
            link = data[j][i]
            tr += '<td>'
            tr += '<a href="%s" target="_blank"><img src="%s" alt="%s" title="%s" style="calc(100%% / 7)"></a>' % (link,
                                                                                                                   link, os.path.basename(link), os.path.join(os.path.basename(dirList[j]), os.path.basename(link)))
            tr += '</td>'
        tr += '</tr>'

    table = '<table style="width:100%">'
    table += th
    table += tr
    table += '</table>'

    html = '''
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body>
    %s
    </body>
    </html>
    ''' % (table)

    f = open(htmlFilePath, 'w')
    f.write(html)
    f.close()
