close all; clear all;
% addpath('/home/tom/University/repositories/projects_archive/edgeval/')
addpath('..\dependencies\edgeval\')
addpath('..\dependencies\edgeval\mex\')
addpath('..\dependencies\edgeval\utils\')

gt_dir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/bsr_bsds500/BSR/BSDS500/data/groundTruth/test'
base_dir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/eval/201905/';

gt_dir='Z:/Master/datasets/bsr_bsds500/BSR/BSDS500/data/groundTruth/test';
base_dir='Z:\Master\eval\201905\';

% % benchmarkBoundary(fullfile(base_dir, 'sharp_net/BSDS500/data/images/test-thres'), gt_dir, 'BDCN');
% % return
% % benchmarkBoundary(fullfile(base_dir, 'sharp_net/BSDS500/data/images/test'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'rcf/BSDS500/data/images/test'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'bdcn/BSDS500/data/images/test'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'hed/BSDS500/data/images/test'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'structured_forests/BSDS500/data/images/test'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'deepcontour/BSDS500/data/images/test'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'canny/BSDS500/data/images/test'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'multiple_canny_thres_25/BSDS500/data/images/test'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'sobel/BSDS500/data/images/test'), gt_dir, 'BDCN');
% 
evalPlotDirs = {
fullfile(base_dir, 'sharp_net/BSDS500/data/images/test-nms'), ...
fullfile(base_dir, 'rcf/BSDS500/data/images/test-nms'), ...
fullfile(base_dir, 'bdcn/BSDS500/data/images/test-nms'), ...
fullfile(base_dir, 'hed/BSDS500/data/images/test-nms'), ...
fullfile(base_dir, 'structured_forests/BSDS500/data/images/test-nms'), ...
fullfile(base_dir, 'deepcontour/BSDS500/data/images/test-nms'), ...
fullfile(base_dir, 'canny_multi_thres/test'), ...
% fullfile(base_dir, 'multiple_canny_thres_25/BSDS500/data/images/test-nms'), ...
% fullfile(base_dir, 'sobel/BSDS500/data/images/test-nms')
};
% % 
% % 
evalPlotTitles = {
'SN (2019)', ...
'RCF (2017)', ...
'BDCN (2019)', ...
'HED (2015)', ...
'SE (2015)', ...
'DC (2015)', ...
'Canny (1986)', ...
% 'Multiple Canny (1986)', ...
% 'Sobel (1986)'
};
% % 
% figure('Name', 'Results', 'Renderer', 'painters', 'Position', [10 10 1200 1200]),
edgesEvalPlot(evalPlotDirs, evalPlotTitles);
% 
% % benchmarkBoundary(fullfile(base_dir, 'sharp_net/BSDS500/data/images/test-thinned'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'rcf/BSDS500/data/images/test-thinned'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'bdcn/BSDS500/data/images/test-thinned'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'hed/BSDS500/data/images/test-thinned'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'structured_forests/BSDS500/data/images/test-thinned'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'deepcontour/BSDS500/data/images/test-thinned'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'canny/BSDS500/data/images/test-thinned'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'multiple_canny_thres_25/BSDS500/data/images/test-thinned'), gt_dir, 'BDCN');
% % benchmarkBoundary(fullfile(base_dir, 'sobel/BSDS500/data/images/test-thinned'), gt_dir, 'BDCN');
% 
% evalPlotDirs = {
% fullfile(base_dir, 'sharp_net/BSDS500/data/images/test-thinned-nms'), ...
% fullfile(base_dir, 'rcf/BSDS500/data/images/test-thinned-nms'), ...
% fullfile(base_dir, 'bdcn/BSDS500/data/images/test-thinned-nms'), ...
% fullfile(base_dir, 'hed/BSDS500/data/images/test-thinned-nms'), ...
% fullfile(base_dir, 'structured_forests/BSDS500/data/images/test-thinned-nms'), ...
% fullfile(base_dir, 'deepcontour/BSDS500/data/images/test-thinned-nms'), ...
% fullfile(base_dir, 'canny/BSDS500/data/images/test-thinned-nms'), ...
% % fullfile(base_dir, 'multiple_canny_thres_25/BSDS500/data/images/test-thinned-nms'), ...
% % fullfile(base_dir, 'sobel/BSDS500/data/images/test-thinned-nms')
% };
% % 
% % 
% evalPlotTitles = {
% 'Sharp Net (2019)', ...
% 'RCF (2017)', ...
% 'BDCN (2019)', ...
% 'HED (2015)', ...
% 'Structured Forests (2015)', ...
% 'Deep Contour (2015)', ...
% 'Canny (1986)', ...
% % 'Multiple Canny (1986)', ...
% % 'Sobel (1986)'
% };
% % 
% % figure('Name', 'Results', 'Renderer', 'painters', 'Position', [10 10 1200 1200]),
% % edgesEvalPlot(evalPlotDirs, evalPlotTitles);

gt_dir='Z:/Master/datasets/bsr_bsds500/BSR/BSDS500/data/groundTruth/test';
nms_dir='Z:\Master\eval\201905\canny_multi_thres\test';
name='Canny';
% benchmarkNmsBoundary(nms_dir, gt_dir, name);
% edgesEvalPlot({'Z:\Master\eval\201905\canny_multi_thres\test'}, name);
function benchmarkNmsBoundary(nms_dir, gt_dir, name)
%%% Not working - wrong image output
nms_dir
param = struct();
param.gtDir = gt_dir;
param.cleanup = 0;
param.thrs = 99;
param.resDir = nms_dir;
%% perform evaluation
try
    [ODS,~,~,~,OIS,~,~,AP,R50] = edgesEvalDir( param );
catch ME
    ME
end
% edgesEvalPlot(nms_dir, name);
end