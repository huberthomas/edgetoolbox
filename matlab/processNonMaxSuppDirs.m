function [ output_args ] = processNonMaxSuppDirs( input_args )
%PROCESSNONMAXSUPPDIRS Summary of this function goes here
%   Detailed explanation goes here
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\hdr_fusion\flicker_synthetic\flicker_1\deepcontour\flicker_1', 'D:\Master-Thesis\datasets\test_dataset\hdr_fusion\flicker_synthetic\flicker_1\deepcontour\flicker_1_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\hdr_fusion\flicker_synthetic\flicker_1\hed\flicker_1', 'D:\Master-Thesis\datasets\test_dataset\hdr_fusion\flicker_synthetic\flicker_1\hed\flicker_1_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\hdr_fusion\flicker_synthetic\flicker_1\structured_forests\flicker_1', 'D:\Master-Thesis\datasets\test_dataset\hdr_fusion\flicker_synthetic\flicker_1\structured_forests\flicker_1_nms_inv', 1)
% 
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\hdr_fusion\smooth_synthetic\flicker_2\deepcontour\flicker_2', 'D:\Master-Thesis\datasets\test_dataset\hdr_fusion\smooth_synthetic\flicker_2\deepcontour\flicker_2_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\hdr_fusion\smooth_synthetic\flicker_2\hed\flicker_2', 'D:\Master-Thesis\datasets\test_dataset\hdr_fusion\smooth_synthetic\flicker_2\hed\flicker_2_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\hdr_fusion\smooth_synthetic\flicker_2\structured_forests\flicker_2', 'D:\Master-Thesis\datasets\test_dataset\hdr_fusion\smooth_synthetic\flicker_2\structured_forests\flicker_2_nms_inv', 1)
% 
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\basements\basement_001c\deepcontour\basement_001c', 'D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\basements\basement_001c\deepcontour\basement_001c_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\basements\basement_001c\hed\basement_001c', 'D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\basements\basement_001c\hed\basement_001c_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\basements\basement_001c\structured_forests\basement_001c', 'D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\basements\basement_001c\structured_forests\basement_001c_nms_inv', 1)
% 
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\cafe\cafe_0001c\deepcontour\cafe_0001c', 'D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\cafe\cafe_0001c\deepcontour\cafe_0001c_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\cafe\cafe_0001c\hed\cafe_0001c', 'D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\cafe\cafe_0001c\hed\cafe_0001c_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\cafe\cafe_0001c\structured_forests\cafe_0001c', 'D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\cafe\cafe_0001c\structured_forests\cafe_0001c_nms_inv', 1)
% 
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\classrooms\classroom_0014\deepcontour\classroom_0014', 'D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\classrooms\classroom_0014\deepcontour\classroom_0014_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\classrooms\classroom_0014\hed\classroom_0014', 'D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\classrooms\classroom_0014\hed\classroom_0014_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\classrooms\classroom_0014\structured_forests\classroom_0014', 'D:\Master-Thesis\datasets\test_dataset\nyu_depth_v2\classrooms\classroom_0014\structured_forests\classroom_0014_nms_inv', 1)
% 
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg1_desk\rgb\deepcontour\rgb', 'D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg1_desk\rgb\deepcontour\rgb_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg1_desk\rgb\hed\rgb', 'D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg1_desk\rgb\hed\rgb_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg1_desk\rgb\structured_forests\rgb', 'D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg1_desk\rgb\structured_forests\rgb_nms_inv', 1)
% 
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg1_xyz\rgb\deepcontour\rgb', 'D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg1_xyz\rgb\deepcontour\rgb_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg1_xyz\rgb\hed\rgb', 'D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg1_xyz\rgb\hed\rgb_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg1_xyz\rgb\structured_forests\rgb', 'D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg1_xyz\rgb\structured_forests\rgb_nms_inv', 1)
% 
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg2_xyz\rgb\deepcontour\rgb', 'D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg2_xyz\rgb\deepcontour\rgb_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg2_xyz\rgb\hed\rgb', 'D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg2_xyz\rgb\hed\rgb_nms_inv', 1)
% nonMaxSuppression('D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg2_xyz\rgb\structured_forests\rgb', 'D:\Master-Thesis\datasets\test_dataset\tum\rgbd_dataset_freiburg2_xyz\rgb\structured_forests\rgb_nms_inv', 1)

end


