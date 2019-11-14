from edgelib import Utilities
import os

'''
Copy depth files related to rgb files by using the lookup table in TUM ground truth handler.
'''

baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets'
trainDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/stableEdgesIndependentTest2/'
subDir = [
    # 'rgbd_dataset_freiburg1_360',
    # 'rgbd_dataset_freiburg1_desk',
    # 'rgbd_dataset_freiburg1_desk2',
    # 'rgbd_dataset_freiburg1_floor',
    # 'rgbd_dataset_freiburg1_plant',
    # 'rgbd_dataset_freiburg1_room',
    # 'rgbd_dataset_freiburg1_rpy',
    # 'rgbd_dataset_freiburg1_teddy',
    # 'rgbd_dataset_freiburg1_xyz',
    # 'rgbd_dataset_freiburg2_360_hemisphere',
    # 'rgbd_dataset_freiburg2_coke',
    # 'rgbd_dataset_freiburg2_desk',
    # 'rgbd_dataset_freiburg2_desk_with_person',
    # 'rgbd_dataset_freiburg2_dishes',
    # 'rgbd_dataset_freiburg2_flowerbouquet',
    # 'rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
    # 'rgbd_dataset_freiburg2_large_no_loop',
    # 'rgbd_dataset_freiburg2_metallic_sphere',
    # 'rgbd_dataset_freiburg2_metallic_sphere2',
    # 'rgbd_dataset_freiburg2_pioneer_360',
    # 'rgbd_dataset_freiburg2_pioneer_slam',
    # 'rgbd_dataset_freiburg2_xyz',
    # 'rgbd_dataset_freiburg3_cabinet',
    # 'rgbd_dataset_freiburg3_large_cabinet',
    # 'rgbd_dataset_freiburg3_long_office_household',
    # 'rgbd_dataset_freiburg3_nostructure_texture_far',
    # 'rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
    # 'rgbd_dataset_freiburg3_sitting_static',
    # 'rgbd_dataset_freiburg3_structure_notexture_far',
    # 'rgbd_dataset_freiburg3_structure_notexture_near',
    # 'rgbd_dataset_freiburg3_structure_texture_far',
    # 'rgbd_dataset_freiburg3_structure_texture_near',
    # 'rgbd_dataset_freiburg3_teddy',
    # 'rgbd_dataset_freiburg3_walking_xyz',
    'eth3d_cables_1',
    'eth3d_cables_2',
    'eth3d_einstein_1',
    'eth3d_einstein_2',
    'eth3d_einstein_global_light_changes_2',
    'eth3d_mannequin_3',
    'eth3d_mannequin_4',
    'eth3d_mannequin_face_1',
    'eth3d_mannequin_face_2',
    'eth3d_planar_2',
    'eth3d_plant_scene_1',
    'eth3d_plant_scene_2',
    'eth3d_sfm_bench',
    'eth3d_sofa_1',
    'eth3d_sofa_2',
    'eth3d_table_3',
    'eth3d_table_4',
    'eth3d_table_7',
    'icl_living_room_0',
    'icl_living_room_1',
    'icl_living_room_2',
    'icl_living_room_3',
    'icl_office_0',
    'icl_office_1',
    'icl_office_2',
    'icl_office_3',
]
for i in range(0, len(subDir)):
    gtAssFileName = os.path.join(baseDir, subDir[i], 'groundtruth_associated.txt')
    dstPath = os.path.join(trainDir, 'depth', subDir[i])
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)

    Utilities.copyRgbFromGtList(os.path.join(baseDir, subDir[i], 'depth'), os.path.join(trainDir, 'gt', subDir[i]), dstPath, gtAssFileName)