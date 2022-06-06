clearvars;
close all;
mainDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg2_xyz/2019-04-15_match_result_schenk';
plotTitle = 'rgbd_dataset_freiburg1_xyz';

files = {
    '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_xyz/em_output/bdcn.txt',...
    '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/rgbd_dataset_freiburg1_xyz/em_output_canny/canny.txt'
%     fullfile(mainDir, 'canny_offset_1.txt'), ...
%     fullfile(mainDir, 'deepcontour_offset_1.txt'), ...
%     fullfile(mainDir, 'rcf_offset_1.txt'), ...
%     fullfile(mainDir, 'hed_offset_1.txt'), ...
%     fullfile(mainDir, 'structured_forests_offset_1.txt')
    %     fullfile(mainDir, 'sobel_offset_1.txt'),
};

% plotTitle = 'fr2xyz algorithm test'
% files = { '/home/tom//University/datasets/sample_rgbd_dataset_freiburg2_xyz/match_result/canny_consecutive_frame_set_offset_4.txt',
%     '/home/tom//University/datasets/sample_rgbd_dataset_freiburg2_xyz/match_result/canny_reprojection_offset_4_offset_4.txt' }

[edgeResults, legendResults] = ICG.getEdgeMatcherResults(files);
ICG.displayEdgeMatcherResultChart(edgeResults, legendResults, plotTitle);
