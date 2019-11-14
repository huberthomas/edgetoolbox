import subprocess
import os

baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/'
inputDirs = [
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

for inputDir in inputDirs:
    subprocess.run(['python', 'associate.py', os.path.join(baseDir, inputDir, 'rgb.txt'), os.path.join(baseDir, inputDir, 'depth.txt'), '--output_file', os.path.join(baseDir, inputDir, 'associate.txt')])