% convert images
clear vars
folder = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/stableEdgesIndependentTest2/all_gt/'
files = dir(strcat(folder,'*.png'));
gtDir = '/home/tom/Downloads/gt_test/';

for m = 1:length(files)
   fprintf('%s\n', files(m).name);
   filename = strcat(folder,files(m).name);
   img = imread(filename);
   img = im2bw(img, 0);
   
   imwrite(imcomplement(img), fullfile(folder, 'inv', strcat(files(m).name(1:end-4),'.png')));
   
   groundTruth = {};
   gt = struct();
   gt.Boundaries = img;
   groundTruth{1, 1} = gt;
   
   save(fullfile(gtDir, strcat(files(m).name(1:end-4),'.mat')), 'groundTruth');
end


% % %%
% % test
% dirs = {
% %     folder
% %  '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/bsr_bsds500/BSR/BSDS500/data/groundTruth/test_new_groundtruth'
% % '/home/tom/Pictures/deleteme/test',...
% '/home/tom/Pictures/deleteme/test_bin',...
% % '/home/tom/Pictures/deleteme/test_inv'
% }
% 
% names = {
% %     'testEdgeProbabilityMap',...
%     'testEdgeBinarized',...
% %     'bsdsTestInverse'
% }
% 
% for i=1:length(dirs)
%     benchmarkBoundary(dirs{i}, '/home/tom/Pictures/deleteme/test_gt', names{i})
% end
% 
% 
% %%
% for i=1:length(dirs)
%     dirs{i} = strcat(dirs{i}, '-nms')
% end
% 
% edgesEvalPlot(dirs, names)