import os
import cv2
import math
import glob
import fnmatch

from typing import List

'''
Utilities and helper functions.
'''
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
            pattern += ('[%s%s]'%(char.lower(), char.upper()))

        res.extend(fnmatch.filter(dirList, '*.%s'%(pattern)))
        
    return res

def rotateImage(img: any = None, angle: float = None, removeCropBorders: bool = True) -> any:
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping

    img OpenCV image.

    angle Angle in degrees.
    """
    if img is None:
        raise ValueError('Image must be set.')

    height, width = img.shape[:2] # image shape has 3 dimensions
    imageCenter = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    M = cv2.getRotationMatrix2D(imageCenter, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    absCos = abs(M[0,0]) 
    absSin = abs(M[0,1])

    # find the new width and height bounds
    boundW = int(height * absSin + width * absCos)
    boundH = int(height * absCos + width * absSin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    M[0, 2] += boundW/2 - imageCenter[0]
    M[1, 2] += boundH/2 - imageCenter[1]

    # rotate image with the new bounds and translated rotation matrix
    rotatedImg = cv2.warpAffine(img, M, (boundW, boundH))
    
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
        img = cv2.flip(img, -1)
    elif flipHorizontal:
        img = cv2.flip(img, 0)
    elif flipVertical:
        img = cv2.flip(img, 1)

    scale = abs(scale)
    img = cv2.resize(img, None, fx=scale, fy=scale)

    rotatedImg = rotateImage(img, angle, cropBlackBorder)

    cv2.imwrite(outputFilePath, rotatedImg)

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

    gamma = math.atan2(bb['w'], bb['h']) if width < height else math.atan2(bb['h'], bb['w'])
    delta = math.pi - alpha - gamma

    length = height if width < height else width
    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return x, y, bb['w'] - 2 * x, bb['h'] - 2 * y
