import os

'''
Utilities and helper functions.
'''
def getImageFileNames(inputDir, supportedExtensions=['.png', '.jpg', '.jpeg']):
    '''
    Get image files (png, jpg, jpeg) from an input directory.
    
    inputDir Input directory that contains images.

    supportedExtensions Only files with supported extensions are included in the final list.

    Returns a list of images file names.
    '''
    res = []

    for root, directories, files in os.walk(inputDir):
        for f in files:
            for extension in supportedExtensions:
                fn, ext = os.path.splitext(f.lower())

                if extension == ext:
                    res.append(f)

    return res