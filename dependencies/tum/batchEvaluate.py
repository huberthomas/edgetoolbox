import subprocess
import os

gtDir = '/home/tom/University/repositories/projects/edgetoolbox/evaluation/tum_gt/'
slamDir = '/home/tom/University/repositories/projects/REVO/eval/'
gtFiles = [
    [
        'rgbd_dataset_freiburg1_360',
        'rgbd_dataset_freiburg1_desk',
        'rgbd_dataset_freiburg1_desk2',
        'rgbd_dataset_freiburg1_floor',
        'rgbd_dataset_freiburg1_plant',
        'rgbd_dataset_freiburg1_room',
        'rgbd_dataset_freiburg1_rpy',
        'rgbd_dataset_freiburg1_teddy',
        'rgbd_dataset_freiburg1_xyz',
    ],
    [
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
        'rgbd_dataset_freiburg2_xyz', ],
    [
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
]

tum_fr1 = [
    [
        'poses_rgbd_dataset_freiburg1_360.txt',
        'bdcn_rgbd_dataset_freiburg1_360_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg1_360_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg1_desk.txt',
        'bdcn_rgbd_dataset_freiburg1_desk_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg1_desk_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg1_desk2.txt',
        'bdcn_rgbd_dataset_freiburg1_desk2_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg1_desk2_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg1_floor.txt',
        'bdcn_rgbd_dataset_freiburg1_floor_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg1_floor_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg1_plant.txt',
        'bdcn_rgbd_dataset_freiburg1_plant_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg1_plant_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg1_room.txt',
        'bdcn_rgbd_dataset_freiburg1_room_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg1_room_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg1_rpy.txt',
        'bdcn_rgbd_dataset_freiburg1_rpy_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg1_rpy_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg1_teddy.txt',
        'bdcn_rgbd_dataset_freiburg1_teddy_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg1_teddy_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg1_xyz.txt',
        'bdcn_rgbd_dataset_freiburg1_xyz_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg1_xyz_poses.txt',
    ],
]

tum_fr2 = [
    [
        'poses_rgbd_dataset_freiburg2_360_hemisphere.txt',
        'bdcn_rgbd_dataset_freiburg2_360_hemisphere_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg2_360_hemisphere_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg2_coke.txt',
        'bdcn_rgbd_dataset_freiburg2_coke_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg2_coke_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg2_desk.txt',
        'bdcn_rgbd_dataset_freiburg2_desk_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg2_desk_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg2_desk_with_person.txt',
        'bdcn_rgbd_dataset_freiburg2_desk_with_person_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg2_desk_with_person_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg2_dishes.txt',
        'bdcn_rgbd_dataset_freiburg2_dishes_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg2_dishes_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg2_flowerbouquet.txt',
        'bdcn_rgbd_dataset_freiburg2_flowerbouquet_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg2_flowerbouquet_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg2_flowerbouquet_brownbackground.txt',
        'bdcn_rgbd_dataset_freiburg2_flowerbouquet_brownbackground_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg2_flowerbouquet_brownbackground_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg2_large_no_loop.txt',
        'bdcn_rgbd_dataset_freiburg2_large_no_loop_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg2_large_no_loop_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg2_metallic_sphere.txt',
        'bdcn_rgbd_dataset_freiburg2_metallic_sphere_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg2_metallic_sphere_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg2_metallic_sphere2.txt',
        'bdcn_rgbd_dataset_freiburg2_metallic_sphere2_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg2_metallic_sphere2_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg2_pioneer_360.txt',
        'bdcn_rgbd_dataset_freiburg2_pioneer_360_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg2_pioneer_360_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg2_pioneer_slam.txt',
        'bdcn_rgbd_dataset_freiburg2_pioneer_slam_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg2_pioneer_slam_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg2_xyz.txt',
        'bdcn_rgbd_dataset_freiburg2_xyz_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg2_xyz_poses.txt',
    ],
]

tum_fr3 = [
    [
        'poses_rgbd_dataset_freiburg3_cabinet.txt',
        'bdcn_rgbd_dataset_freiburg3_cabinet_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg3_cabinet_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg3_large_cabinet.txt',
        'bdcn_rgbd_dataset_freiburg3_large_cabinet_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg3_large_cabinet_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg3_long_office_household.txt',
        'bdcn_rgbd_dataset_freiburg3_long_office_household_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg3_long_office_household_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg3_nostructure_texture_far.txt',
        'bdcn_rgbd_dataset_freiburg3_nostructure_texture_far_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg3_nostructure_texture_far_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg3_nostructure_texture_near_withloop.txt',
        'bdcn_rgbd_dataset_freiburg3_nostructure_texture_near_withloop_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg3_nostructure_texture_near_withloop_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg3_sitting_static.txt',
        'bdcn_rgbd_dataset_freiburg3_sitting_static_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg3_sitting_static_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg3_structure_notexture_far.txt',
        'bdcn_rgbd_dataset_freiburg3_structure_notexture_far_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg3_structure_notexture_far_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg3_structure_notexture_near.txt',
        'bdcn_rgbd_dataset_freiburg3_structure_notexture_near_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg3_structure_notexture_near_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg3_structure_texture_far.txt',
        'bdcn_rgbd_dataset_freiburg3_structure_texture_far_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg3_structure_texture_far_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg3_structure_texture_near.txt',
        'bdcn_rgbd_dataset_freiburg3_structure_texture_near_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg3_structure_texture_near_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg3_teddy.txt',
        'bdcn_rgbd_dataset_freiburg3_teddy_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg3_teddy_poses.txt',
    ],

    [
        'poses_rgbd_dataset_freiburg3_walking_xyz.txt',
        'bdcn_rgbd_dataset_freiburg3_walking_xyz_poses.txt',
        'bdcn_sgdss30ktumaug_rgbd_dataset_freiburg3_walking_xyz_poses.txt',
    ],
]

tum = [
    tum_fr1,
    tum_fr2,
    tum_fr3
]
print('dataset;pairs;te_rmse_[m];te_mean_[m];te_median_[m];te_std_[m];te_min_[m];te_max_[m];re_rmse_[deg];re_mean_[deg];re_median_[deg];re_std_[deg];re_min_[deg];re_max_[deg]')
for j in range(0, len(gtFiles)):
    for i in range(0, len(gtFiles[j])):
        inputDir = gtFiles[j][i]
        dataset = tum[j][i]
        for slamFile in dataset:
            baseFilename = slamFile.replace('.txt', '')
            # subprocess.run(['python', 'evaluate_ate.py', os.path.join(gtDir, '%s-groundtruth.txt' % (inputDir)), os.path.join(slamDir, slamFile),
            #                 '--plot', 'results_ate/figure_%s.png' % (baseFilename), '--outputFile', 'results_ate/ate_%s.txt' % (baseFilename), '--offset', '0', '--scale', '1', '--verbose'])
            #subprocess.run(['python', 'evaluate_rpe.py', os.path.join(gtDir, '%s-groundtruth.txt' % (inputDir)), os.path.join(slamDir, slamFile), '--max_pairs', '10000', '--fixed_delta', '--delta', '1', '--delta_unit', 's', '--plot', 'results_rpe/figure_%s.png' % (baseFilename), '--outputFile', 'results_rpe/rpe_%s.txt' % (baseFilename), '--offset', '0', '--scale', '1', '--verbose'])
            subprocess.run(['python', 'evaluate_rpe.py', os.path.join(gtDir, '%s-groundtruth.txt' % (inputDir)), os.path.join(slamDir, slamFile), '--max_pairs', '10000', '--fixed_delta', '--delta', '1', '--delta_unit', 'm', '--plot', 'results_rpe_per_m/figure_%s.png' % (baseFilename), '--outputFile', 'results_rpe_per_m/rpe_%s.txt' % (baseFilename), '--offset', '0', '--scale', '1', '--verbose'])
