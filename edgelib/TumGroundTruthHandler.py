import os
import sys
import time
import numpy as np
import logging
import collections
from pathlib import Path
from typing import List
from .TumGtAssociated import TumGtAssociated
from .TumGroundTruth import TumGroundTruth
from . import Utilities


class TumGroundTruthHandler:
    '''
    TumGroundTruth object finds correspondences between files
    that are captured by 30Hz and the high-speed tracking ground truth
    system 100Hz.
    '''

    def __init__(self):
        '''
        Constructor.
        '''
        self.__associations = {}

    def data(self) -> List[TumGtAssociated]:
        '''
        Get the associated data.
        '''
        return self.__associations.values()

    def progress(self, percentage: float = None) -> None:
        '''
        Function stup that can be overwritten to get the progress of some functions.

        percentage Current progress in percent.
        '''
        pass

    def __fileCorrespondence(self, filePath: str = None) -> dict:
        '''
        Reads correspondence file and returns dictionary with entries.

        filePath Path to the correspondence file, e.g. rgb.txt file like in the ETH3D dataset.
            11784.337488890 depth/11784.337488.png
            11784.374352455 depth/11784.374352.png
            11784.411215782 depth/11784.411215.png
            11784.448079109 depth/11784.448079.png

        Returns dictionary with readen entries.
        '''
        if(filePath is None):
            raise ValueError('Invalid file path.')

        correspondences = {}

        try:
            f = open(filePath, 'r')

            for line in f:
                if line.count('#') or len(line) == 0:
                    continue

                entries = line.split(' ')

                if len(entries) < 2:
                    logging.info('Wrong file format? Less than 2 entries. Skipping line "%s"' % (line))
                    continue

                timestamp = np.float64(entries[0])
                relFilePath = Path(entries[1]).stem
                correspondences.update({relFilePath: timestamp})

            f.close()
        except Exception as e:
            raise e

        return correspondences

    def associate(self, groundTruthPath: str = None,
                  rgbDirPath: str = None,
                  depthDirPath: str = None,
                  maxDifference: float = 0.2,
                  rgbFile: str = None,
                  depthFile: str = None) -> None:
        '''
        TumGroundTruth object finds correspondences between files
        that are captured by 30Hz and the high-speed tracking ground truth
        system 100Hz.

        groundTruthPath Absolute file path to the groundtruth.txt file.

        rgbDirPath Absolute directory path to the RGB directory.

        depthDirPath Absolute directory path to the depth directory.

        maxDifference Max time difference that is allowed to hold the value, otherwise it is skipped.

        rgbFile RGB association file, e.g. rgb.txt.

        depthFile Depth association file, e.g. depth.txt.
        '''
        gtList = self.__readRawGroundTruthFile(groundTruthPath)


        rgbCorrespondences = {}
        depthCorrespondences = {}

        if rgbFile is not None:
            rgbCorrespondences = self.__fileCorrespondence(rgbFile)

        if depthFile is not None: 
            depthCorrespondences = self.__fileCorrespondence(depthFile)

        dirAssociations = self.__associateDirectoryFiles(rgbDirPath, depthDirPath, maxDifference, rgbCorrespondences, depthCorrespondences)
        self.__associations = self.__associateGroundTruthWithDir(gtList, dirAssociations)

    def save(self, filePath: str = None) -> None:
        '''
        Save associated ground truth to a file in the format
        timestamp tx ty tz qx qy qz qw rgbFileName depthFileName

        filePath Data is stored to this file.
        '''
        if filePath is None or len(filePath) == 0:
            raise ValueError('Invalid file path.')

        if self.__associations is None:
            raise ValueError('No associations set.')

        try:
            f = open(filePath, 'w')
            
            f.write('# ground truth trajectory\n')
            f.write('# timestamp tx ty tz qx qy qz qw rgbFileName depthFileName\n')

            counter = 0
            total = len(self.__associations)
            for a in self.__associations.values():
                # recommended for ICL dataset
                # f.write('%f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %s %s\n' % (a.gt.timestamp,
                #                                              a.gt.t[0], a.gt.t[1], a.gt.t[2],
                #                                              a.gt.q[1], a.gt.q[2], a.gt.q[3], a.gt.q[0],
                #                                              a.rgb, a.depth))                                                             
                # recommended for TUM dataset
                # f.write('%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %s %s\n' % (a.gt.timestamp,
                #                                              a.gt.t[0], a.gt.t[1], a.gt.t[2],
                #                                              a.gt.q[1], a.gt.q[2], a.gt.q[3], a.gt.q[0],
                #                                              a.rgb, a.depth))
                # recommended for ETH dataset
                f.write('%.9f %.14f %.14f %.14f %.14f %.14f %.14f %.14f %s %s\n' % (a.gt.timestamp,
                                                             a.gt.t[0], a.gt.t[1], a.gt.t[2],
                                                             a.gt.q[1], a.gt.q[2], a.gt.q[3], a.gt.q[0],
                                                             a.rgb, a.depth))
                counter = counter + 1
                self.progress(counter / total * 100)

            f.close()
        except Exception as e:
            raise e

    def load(self, filePath: str = None) -> None:
        '''
        Load data from a file. The entries must be of the form
        timestamp tx ty tz qx qy qz qw rgbFileName depthFileName

        filePath File path to the associated ground truth file.
        '''
        if filePath is None or len(filePath) == 0:
            raise ValueError('Invalid file path.')

        associations = {}

        try:
            f = open(filePath, 'r')

            counter = -1
            total = len(f.readlines())

            # reset file pointer after readlines
            f.seek(0)

            for line in f:
                counter = counter + 1
                self.progress((counter + 1) / total)

                if line.count('#') or len(line) == 0:
                    continue

                entries = line.split(' ')

                if len(entries) < 10:
                    logging.info('Wrong file format? Less than 10 entries. Skipping line "%s"' % (line))
                    continue

                association = TumGtAssociated()
                association.loadFromStringLine(line)

                associations.update({('%f' % association.gt.timestamp): association})

            f.close()
        except Exception as e:
            raise e

        # sort associations to be sure that the timestamps are increasing
        self.__associations = collections.OrderedDict((associations.items()))

    def __readRawGroundTruthFile(self, tumGtPath: str = None) -> List[TumGroundTruth]:
        '''
        Reads the ground truth information out of a defined file.

        tumGtPath File path to the ground truth file, e.g.
        # timestamp tx ty tz qx qy qz qw
        11873.240000000 -1.2150835027514 0.75432647351722 1.4685295511364 -0.22243192269128 0.86350037964587 -0.38800252585617 0.23312051400195
        11873.250000000 -1.2098883980529 0.74163830982265 1.4657813349346 -0.22270440288169 0.86362409233908 -0.38739438237336 0.2334132999085
        11873.260000000 -1.2046015590052 0.72900775516528 1.4631370750175 -0.22283701229784 0.86372016942534 -0.38699356489541 0.23359605220888

        Returns datastructure that contains ground truth.
        '''
        if tumGtPath is None:
            raise ValueError('Invalid file path.')

        result = []

        try:
            f = open(tumGtPath, 'r')

            for line in f:
                if len(line) == 0:
                    continue
                # ignore comments
                if line.count('#') > 0:
                    continue

                gt = TumGroundTruth()
                gt.loadFromStringLine(line)

                result.append(gt)

            f.close()

        except Exception as e:
            raise e

        return result

    def __readAssociatedGroundTruth(self, tumGtAsPath: str = None,
                                    rgbDirPath: str = None,
                                    depthDirPath: str = None) -> None:
        '''
        Read the associated ground truth from a file and store it internally.

        tumGtAsPath TUM grouth truth associated path.

        rgbDirPath RGB directory path of the RGB files located in the ground truth file.

        depthDirPath Depth directory path of the depth files located in the ground truth file.
        '''
        if tumGtAsPath is None or len(tumGtAsPath) == 0:
            raise ValueError('Invalid associated ground truth path.')

        if rgbDirPath is None or len(rgbDirPath) == 0:
            raise ValueError('Invalid RGB directory path.')

        if depthDirPath is None or len(depthDirPath) == 0:
            raise ValueError('Invalid depth directory path.')

        associated = {}

        try:
            f = open(tumGtAsPath, 'r')

            for line in f:
                if len(line) == 0:
                    continue
                # ignore comments
                if line.count('#') > 0:
                    continue

                gta = TumGtAssociated()
                gta.loadFromStringLine(line)
                gta.rgb = os.path.join(rgbDirPath, gta.rgb)
                gta.depth = os.path.join(depthDirPath, gta.depth)

                associated['%f' % gta.gt.timestamp] = gta

            f.close()

        except Exception as e:
            raise e

        self.__associated = associated

    def __associateDirectoryFiles(self, rgbDirPath: str = None,
                                  depthDirPath: str = None,
                                  maxDifference: float = 0.2,
                                  rgbCorrespondences: dict = {},
                                  depthCorrespondences: dict = {}) -> List[List[str]]:
        '''
        Associate files of the RGB and depth directory by their filename which is a timestamp of the record.

        rgbDirPath Directory of the RGB files.

        depthDirPath Directory of the depth files.

        maxDifference Max time difference that is allowed to hold the value, otherwise it is skipped.

        Returns list of corresponding rgb and depth files with minimum time difference.
        '''
        if rgbDirPath is None or len(rgbDirPath) == 0:
            raise ValueError('Invalid RGB directory path.')

        if depthDirPath is None or len(depthDirPath) == 0:
            raise ValueError('Invalid depth directory path.')

        if maxDifference is None:
            raise ValueError('Invalid maximum difference set.')

        if maxDifference < 0:
            maxDifference = abs(maxDifference)

        rgbFiles = Utilities.getFileNames(rgbDirPath)
        depthFiles = Utilities.getFileNames(depthDirPath)

        # rgbFiles.sort()
        # depthFiles.sort()

        rgbFiles = Utilities.naturalSort(rgbFiles)
        depthFiles = Utilities.naturalSort(depthFiles)

        associations = []
        ascIndex = 0


        for rgbFile in rgbFiles:
            rgbTimestamp = Path(rgbFile).resolve().stem

            if len(rgbCorrespondences) != 0:
                rgbTimestamp = rgbCorrespondences[rgbTimestamp]
            else:
                rgbTimestamp = np.float64(rgbTimestamp)


            foundCandidateIndex = -1
            minDifference = sys.float_info.max

            for i in range(ascIndex, len(depthFiles)):
                depthFile = depthFiles[i]
                depthTimestamp = Path(depthFile).resolve().stem
                
                if len(depthCorrespondences) != 0:
                    depthTimestamp = depthCorrespondences[depthTimestamp]
                else:
                    depthTimestamp = np.float64(depthTimestamp)


                diff = abs(rgbTimestamp - depthTimestamp)

                if diff <= maxDifference:
                    if diff < minDifference:
                        minDifference = diff
                        foundCandidateIndex = i

            if foundCandidateIndex < 0:
                continue

            associations.append([rgbFile, depthFiles[foundCandidateIndex]])
            ascIndex = foundCandidateIndex

        return associations

    def __associateGroundTruthWithDir(self, gtList: List[TumGroundTruth] = None,
                                      dirAssociations: List[List[str]] = None) -> dict:
        '''
        Associates ground truth entries to the lowest time difference of the files.

        gtList Ground truth list.

        dirAssociations Directory associations, QPair is RGB and Depth.

        Returns dictionary containing timestamp (key) and association (value) sorted by timestamp.
        '''
        associations = {}
        ascIndex = 0
        for files in dirAssociations:
            rgbTimestamp = np.float64(Path(files[0]).resolve().stem)
            depthTimestamp = np.float64(Path(files[1]).resolve().stem)

            foundCandidateIndex = -1
            minDiff = sys.float_info.max

            for i in range(ascIndex, len(gtList)):
                gt = gtList[i]
                rgbDiff = abs(gt.timestamp - rgbTimestamp)
                depthDiff = abs(gt.timestamp - depthTimestamp)

                if rgbDiff < depthDiff:
                    if rgbDiff < minDiff:
                        minDiff = rgbDiff
                        foundCandidateIndex = i
                else:
                    if depthDiff < minDiff:
                        minDiff = depthDiff
                        foundCandidateIndex = i

            if foundCandidateIndex < 0:
                continue

            association = TumGtAssociated()
            association.gt = gtList[foundCandidateIndex]
            association.rgb = files[0]
            association.depth = files[1]

            associations.update({('%f' % association.gt.timestamp): association})
            ascIndex = foundCandidateIndex
        
        return collections.OrderedDict((associations.items()))
