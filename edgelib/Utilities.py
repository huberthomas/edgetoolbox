import os
import cv2 as cv
import math
import glob
import fnmatch
import shutil
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import random
import re

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
    baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/stableEdgesTest2'
    gtDirs = [
        'gt/rgbd_dataset_freiburg1_360',
        'gt/rgbd_dataset_freiburg1_desk',
        'gt/rgbd_dataset_freiburg1_desk2',
        'gt/rgbd_dataset_freiburg1_floor',
        'gt/rgbd_dataset_freiburg1_plant',
        'gt/rgbd_dataset_freiburg1_room',
        'gt/rgbd_dataset_freiburg1_rpy',
        'gt/rgbd_dataset_freiburg1_teddy',
        'gt/rgbd_dataset_freiburg1_xyz',
        'gt/rgbd_dataset_freiburg2_360_hemisphere',
        'gt/rgbd_dataset_freiburg2_coke',
        'gt/rgbd_dataset_freiburg2_desk',
        'gt/rgbd_dataset_freiburg2_desk_with_person',
        'gt/rgbd_dataset_freiburg2_dishes',
        'gt/rgbd_dataset_freiburg2_flowerbouquet',
        'gt/rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
        'gt/rgbd_dataset_freiburg2_large_no_loop',
        'gt/rgbd_dataset_freiburg2_metallic_sphere',
        'gt/rgbd_dataset_freiburg2_metallic_sphere2',
        'gt/rgbd_dataset_freiburg2_pioneer_360',
        'gt/rgbd_dataset_freiburg2_pioneer_slam',
        'gt/rgbd_dataset_freiburg2_xyz',
        'gt/rgbd_dataset_freiburg3_cabinet',
        'gt/rgbd_dataset_freiburg3_large_cabinet',
        'gt/rgbd_dataset_freiburg3_long_office_household',
        'gt/rgbd_dataset_freiburg3_nostructure_texture_far',
        'gt/rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
        'gt/rgbd_dataset_freiburg3_sitting_static',
        'gt/rgbd_dataset_freiburg3_structure_notexture_far',
        'gt/rgbd_dataset_freiburg3_structure_notexture_near',
        'gt/rgbd_dataset_freiburg3_structure_texture_far',
        'gt/rgbd_dataset_freiburg3_structure_texture_near',
        'gt/rgbd_dataset_freiburg3_teddy',
        'gt/rgbd_dataset_freiburg3_walking_xyz'
    ]
    rgbDirs = [
        'rgb/rgbd_dataset_freiburg1_360',
        'rgb/rgbd_dataset_freiburg1_desk',
        'rgb/rgbd_dataset_freiburg1_desk2',
        'rgb/rgbd_dataset_freiburg1_floor',
        'rgb/rgbd_dataset_freiburg1_plant',
        'rgb/rgbd_dataset_freiburg1_room',
        'rgb/rgbd_dataset_freiburg1_rpy',
        'rgb/rgbd_dataset_freiburg1_teddy',
        'rgb/rgbd_dataset_freiburg1_xyz',
        'rgb/rgbd_dataset_freiburg2_360_hemisphere',
        'rgb/rgbd_dataset_freiburg2_coke',
        'rgb/rgbd_dataset_freiburg2_desk',
        'rgb/rgbd_dataset_freiburg2_desk_with_person',
        'rgb/rgbd_dataset_freiburg2_dishes',
        'rgb/rgbd_dataset_freiburg2_flowerbouquet',
        'rgb/rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
        'rgb/rgbd_dataset_freiburg2_large_no_loop',
        'rgb/rgbd_dataset_freiburg2_metallic_sphere',
        'rgb/rgbd_dataset_freiburg2_metallic_sphere2',
        'rgb/rgbd_dataset_freiburg2_pioneer_360',
        'rgb/rgbd_dataset_freiburg2_pioneer_slam',
        'rgb/rgbd_dataset_freiburg2_xyz',
        'rgb/rgbd_dataset_freiburg3_cabinet',
        'rgb/rgbd_dataset_freiburg3_large_cabinet',
        'rgb/rgbd_dataset_freiburg3_long_office_household',
        'rgb/rgbd_dataset_freiburg3_nostructure_texture_far',
        'rgb/rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
        'rgb/rgbd_dataset_freiburg3_sitting_static',
        'rgb/rgbd_dataset_freiburg3_structure_notexture_far',
        'rgb/rgbd_dataset_freiburg3_structure_notexture_near',
        'rgb/rgbd_dataset_freiburg3_structure_texture_far',
        'rgb/rgbd_dataset_freiburg3_structure_texture_near',
        'rgb/rgbd_dataset_freiburg3_teddy',
        'rgb/rgbd_dataset_freiburg3_walking_xyz'
    ]
    outputFileName = 'train_pair.lst'

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
            os.path.join(baseSubDir, 'bdcn_tum_gpu'),
            os.path.join(baseSubDir, 'bdcn_tum_7k_aug_gpu'),
            os.path.join(baseSubDir, 'bdcn_singleScale_gpu_tum_21k_augplus')
            # os.path.join(baseSubDir, 'bdcn_30k'),
            # os.path.join(baseSubDir, 'bdcn_20k'),
            # os.path.join(baseSubDir, 'bdcn_10k'),
            # os.path.join(baseSubDir, 'bdcn_5k'),
            # os.path.join(baseSubDir, 'bdcn_1k'),
            # os.path.join(baseSubDir, 'bdcn_1k'),
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



def copyRgbFromGtList(rgbSrcDir: str = None, gtSrcDir: str = None, rgbDstDir: str = None) -> None:
    '''
    Copy corresponding RGB file from the GT directory to a new RGB destination dir. Looks in GT directory
    for image filenames and copies RGB equivalent to the destination folder.

    rgbSrcDir RGB image source directory.

    gtSrcDir Ground truth source directory.

    rgbDst RGB image destination directory.

    e.g.
    baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets'
    trainDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/stableEdges2/'
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
        'rgbd_dataset_freiburg3_walking_xyz'
    ]
    for i in range(0, len(subDir)):
        copyRgbFromGtList(os.path.join(baseDir, subDir[i], 'rgb'), os.path.join(trainDir, 'gt', subDir[i]), os.path.join(trainDir, 'rgb', subDir[i]))
    '''
    gtFileNames = getFileNames(gtSrcDir)

    for filename in gtFileNames:
        srcRgbFile = os.path.join(rgbSrcDir, filename)
        if os.path.exists(srcRgbFile):
            shutil.copyfile(srcRgbFile, os.path.join(rgbDstDir, filename))
        else:
            print('Missing file: %s'%(srcRgbFile))


def generateMultiScaleImage(imageFileNames: List[str] = [], srcDirs: List[str] = [], outputDir: str = None) -> None:
    '''
    Combines multiple images to a single one by adding same images from the src directory together and take the mean
    as result. All images will be scaled to the resolution of the image in the first directory.

    imageFilesNames Image file names that are located in all src directories.

    srcDirs Source image directory.

    outputDir Result output directory.

    e.g.
    baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/stableEdgesTest2/all_rgb/'
    subDir = 'canny_multiScale_cpu'
    levelDirs = [
        os.path.join(baseDir, subDir, 'level0'),
        os.path.join(baseDir, subDir, 'level1'),
        os.path.join(baseDir, subDir, 'level2')
    ]
    imageFileNames = getFileNames(os.path.join(baseDir, subDir, 'level0'))
    generateMultiScaleImage(imageFileNames, levelDirs, os.path.join(baseDir, subDir))
    '''
    scales = len(srcDirs)
    total = len(imageFileNames)
    for i in range(0, total):
        imageFileName = imageFileNames[i]

        print('Processing %s %.2f%%'%(imageFileName, ((i+1)/total)*100))
        
        resImg = None
        shape = (0, 0)
        for srcDir in srcDirs:
            imgPath = os.path.join(srcDir, imageFileName)

            if not os.path.exists(imgPath):
                raise ValueError('Invalid image %s' % (imgPath))

            img = np.array(cv.imread(imgPath, cv.IMREAD_GRAYSCALE))

            if resImg is None:
                # rescale all images afterwards to this image
                resImg = np.array(img).astype(np.float64)
                shape = resImg.shape[:2]
                continue

            if img.shape[:2] != shape:
                img = cv.resize(img, (shape[1], shape[0]), interpolation=cv.INTER_LINEAR)

            resImg += img
        
        resImg /= scales

        cv.imwrite(os.path.join(outputDir, imageFileName), resImg)


def writeStringList(filePath, strList):
    '''
    Write list of strings to a file.

    filePath Output file path.

    strList Array that contains a string in each line.
    '''
    f = open(filePath, 'w')
    for line in strList:
        f.write('%s\n'%(line.strip()))
    f.close()

def createValidationLst(dataRootDir: str = None, dataLstFile: str = None):
    '''
    Creates a validation list out of a data list file. The validation files
    are not located in the datalist anymore.

    dataRootDir Data root directory.

    dataLstFile File that contains RGB/ground truth correspondences in each line, e.g.
    rgb_aug/0.0_0_1_1.0/1311867171.026274.png gt_aug/0.0_0_1_1.0/1311867171.026274.png
    rgb_aug/22.5_0_0_1.0/1311867171.026274.png gt_aug/22.5_0_0_1.0/1311867171.026274.png
    rgb_aug/22.5_0_1_1.0/1311867171.026274.png gt_aug/22.5_0_1_1.0/1311867171.026274.png

    e.g.
    dataRootDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/stableEdges2'
    dataLstFile = 'train_pair_.lst'
    createValidationLst(dataRootDir, dataLstFile)
    '''
    filePath = os.path.join(dataRootDir, dataLstFile)
    trainDataLstFile = os.path.join(dataRootDir, 'auto_generated_train_' + dataLstFile)
    valDataLstFile = os.path.join(dataRootDir, 'auto_generated_val_' + dataLstFile)
    trainDataLst = []
    valDataLst = []
    with open(filePath, 'r') as f:
        files = f.readlines()
        #files = [line.strip().split('\n') for line in files]
        total = len(files)
        #print(total, files[:4])
        # 10% of dataset is validation
        valTotal = total * 0.1

        if valTotal > 100:
            # limit val to max 100 images
            valTotal = 100

        while len(valDataLst) < valTotal:
            k = random.randint(0, total-2)
            # create new lists
            trainDataLst = files[:k]
            valDataLst.append(files[k])
            trainDataLst.extend(files[k+1:])
            files = trainDataLst
            total -= 1

    writeStringList(trainDataLstFile, trainDataLst)
    writeStringList(valDataLstFile, valDataLst)

def naturalSort(l: List[str]): 
    '''
    Sorting dictionary naturally, 
    e.g. [1.png 2.png, 10.png, 100.png]
    sort -> 1.png 10.png 100.png 2.png
    naturalSort -> 1.png 2.png 10.png 100.png

    l List with strings.

    Returns sorted list
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

