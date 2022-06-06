addpath('..\dependencies\edgeval\')
addpath('..\dependencies\edgeval\mex\')
addpath('..\dependencies\edgeval\utils\')

% baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/desk_daylight_xyz/';
% invertImages(fullfile(baseDir, 'mask/bdcn_thinned'), fullfile(baseDir, 'mask/bdcn_thinned_inv'));
% invertImages(fullfile(baseDir, 'mask/hed_thinned'), fullfile(baseDir, 'mask/hed_thinned_inv'));
% invertImages(fullfile(baseDir, 'mask/rcf_thinned'), fullfile(baseDir, 'mask/rcf_thinned_inv'));
% invertImages(fullfile(baseDir, 'mask/structured_forests_thinned'), fullfile(baseDir, 'mask/structured_forests_thinned_inv'));
% invertImages(fullfile(baseDir, 'mask/canny'), fullfile(baseDir, 'mask/canny_inv'));
%
% baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/desk_dimmed_xyz/';
% invertImages(fullfile(baseDir, 'mask/bdcn_thinned'), fullfile(baseDir, 'mask/bdcn_thinned_inv'));
% invertImages(fullfile(baseDir, 'mask/hed_thinned'), fullfile(baseDir, 'mask/hed_thinned_inv'));
% invertImages(fullfile(baseDir, 'mask/rcf_thinned'), fullfile(baseDir, 'mask/rcf_thinned_inv'));
% invertImages(fullfile(baseDir, 'mask/structured_forests_thinned'), fullfile(baseDir, 'mask/structured_forests_thinned_inv'));
% invertImages(fullfile(baseDir, 'mask/canny'), fullfile(baseDir, 'mask/canny_inv'));
%
% baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/desk_neon_xyz/';
% invertImages(fullfile(baseDir, 'mask/bdcn_thinned'), fullfile(baseDir, 'mask/bdcn_thinned_inv'));
% invertImages(fullfile(baseDir, 'mask/hed_thinned'), fullfile(baseDir, 'mask/hed_thinned_inv'));
% invertImages(fullfile(baseDir, 'mask/rcf_thinned'), fullfile(baseDir, 'mask/rcf_thinned_inv'));
% invertImages(fullfile(baseDir, 'mask/structured_forests_thinned'), fullfile(baseDir, 'mask/structured_forests_thinned_inv'));
% invertImages(fullfile(baseDir, 'mask/canny'), fullfile(baseDir, 'mask/canny_inv'));

% inputDir='D:/Nextcloud/master/master_thesis/assets/chapter02/bsds500/thinned/'
% outputDir='D:/Nextcloud/master/master_thesis/assets/chapter02/bsds500/thinned_inv/'

% baseDir='D:\Nextcloud\master\portable-apache\htdocs\test_dataset\'

% inputDir='hdr_fusion\flicker_synthetic\flicker_1\bdcn_tum_gpu\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
% inputDir='hdr_fusion\flicker_synthetic\flicker_1\bdcn_tum_gpu_aug\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
%
% inputDir='hdr_fusion\smooth_synthetic\flicker_2\bdcn_tum_gpu\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
% inputDir='hdr_fusion\smooth_synthetic\flicker_2\bdcn_tum_gpu_aug\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
%
% inputDir='nyu_depth_v2\basements\basement_001c\bdcn_tum_gpu\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
% inputDir='nyu_depth_v2\basements\basement_001c\bdcn_tum_gpu_aug\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
%
% inputDir='nyu_depth_v2\cafe\cafe_0001c\bdcn_tum_gpu\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
% inputDir='nyu_depth_v2\cafe\cafe_0001c\bdcn_tum_gpu_aug\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
%
% inputDir='nyu_depth_v2\classrooms\classroom_0014\bdcn_tum_gpu\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
% inputDir='nyu_depth_v2\classrooms\classroom_0014\bdcn_tum_gpu_aug\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
%
% inputDir='tum\rgbd_dataset_freiburg1_desk\rgb\bdcn_tum_gpu\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
% inputDir='tum\rgbd_dataset_freiburg1_desk\rgb\bdcn_tum_gpu_aug\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
%
% inputDir='tum\rgbd_dataset_freiburg1_xyz\rgb\bdcn_tum_gpu\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
% inputDir='tum\rgbd_dataset_freiburg1_xyz\rgb\bdcn_tum_gpu_aug\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
%
% inputDir='tum\rgbd_dataset_freiburg2_xyz\rgb\bdcn_tum_gpu\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))
% inputDir='tum\rgbd_dataset_freiburg2_xyz\rgb\bdcn_tum_gpu_aug\'
% invertImages(fullfile(baseDir,inputDir), fullfile(baseDir,inputDir))

baseDir='Z:\Master\eval\comparison_learned_vs_detected\'
fileDir='bdcn_30k_aug_gpu'
edgeThinning(fullfile(baseDir, strcat(fileDir)),fullfile(baseDir, strcat(fileDir,'_thinned')))
% invertImages(fullfile(baseDir, fileDir),fullfile(baseDir, fileDir))
% fileDir='gt'
% invertImages(fullfile(baseDir, fileDir),fullfile(baseDir, fileDir))

baseDir='D:\Universit�t\xampp\htdocs\master\test_dataset'
invImg(baseDir, 'hdr_fusion\flicker_synthetic\flicker_1\bdcn_tum_gpu')
invImg(baseDir, 'hdr_fusion\smooth_synthetic\flicker_2\bdcn_tum_gpu')
invImg(baseDir, 'nyu_depth_v2\basements\basement_001c\bdcn_tum_gpu')
invImg(baseDir, 'nyu_depth_v2\cafe\cafe_0001c\bdcn_tum_gpu')
invImg(baseDir, 'nyu_depth_v2\classrooms\classroom_0014\bdcn_tum_gpu')
invImg(baseDir, 'tum\rgbd_dataset_freiburg1_desk\rgb\bdcn_tum_gpu')
invImg(baseDir, 'tum\rgbd_dataset_freiburg1_xyz\rgb\bdcn_tum_gpu')
invImg(baseDir, 'tum\rgbd_dataset_freiburg1_xyz\rgb\bdcn_tum_gpu')



invertImages(fullfile('D:\Universit�t\xampp\htdocs\master\test_dataset\tum\rgbd_dataset_freiburg2_xyz\rgb\bdcn\rgb_thinned'),fullfile('D:\Universit�t\xampp\htdocs\master\test_dataset\tum\rgbd_dataset_freiburg2_xyz\rgb\bdcn\rgb_thinned_inv'))
% aBase='D:/Universit�t/xampp/htdocs/master/test_dataset/tum/rgbd_dataset_freiburg2_xyz/rgb/bdcn';
% a = 'rgb/1311867280.549209.png';
% b = 'rgb_nms_raw/1311867280.549209.png';
% imwrite(ICG.nmsEdgeImage(imread(fullfile(aBase,a))),fullfile(aBase,b));

function [] = invImg(baseDir, fileDir)
% invertImages(fullfile(baseDir, fileDir),fullfile(baseDir, fileDir))
% invertImages(fullfile(baseDir, strcat(fileDir,'aug')),fullfile(baseDir, strcat(fileDir,'aug'))))
end