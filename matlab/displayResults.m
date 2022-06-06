

% close all;
% clearvars;
%
% from = 1855;
% to = 1865;
%
% mainDir = '/home/tom/University/datasets/rgbd_dataset_freiburg2_xyz'
% rgbDir = fullfile(mainDir, '2019-04-15_match_result_schenkz', 'canny', 'rgb');
% maskDir = fullfile(mainDir, '2019-04-15_match_result_schenk');
% maskSubDir = {'canny/rgb'};%, 'rcf', 'hed', 'deepcontour', 'structured_forests'};
% outputDir = fullfile(mainDir, 'rgbd_dataset_freiburg2_xyz_match_result_ros_all');
%
% if(from > 0 && to > 0)
%     outputDir = strcat(outputDir, '_', num2str(from), '-', num2str(to))
% end
%
% fileFilter = {'jpeg', 'jpg', 'bmp', 'png'};
%
% if(~ICG.existsOrCreate(outputDir, true))
%     error('Invalid output directory: "%s".', outputDir);
%     return
% end
%
% maxNum = length(maskSubDir);
%
% for n=1:length(fileFilter)
%     fullfile(rgbDir, strcat('*.', fileFilter{n}))
%     imageFiles = dir(fullfile(rgbDir, strcat('*.', fileFilter{n})));
%     imageFiles = {imageFiles.name};
%     numImages = length(imageFiles);
%
%     for i=1:numImages, imgName = imageFiles{i}(1:end-4);
%
%         if(i < from || i > to)
%            continue;
%         end
%
%         if(exist(fullfile(outputDir, strcat(imgName, '.png')), 'file'))
%             continue;
%         end
%
%         fprintf('%d%% %s\n', floor(((i)/numImages)*100), fullfile(outputDir, strcat(imgName, '.', fileFilter{n})));
%
%         h = figure('Position', get(0, 'Screensize'), 'Name', imageFiles{i}, 'NumberTitle','off')
%         suptitle(imageFiles{i});
%
%         for j=1:length(maskSubDir)
%             maskImg = imread(fullfile(maskDir, maskSubDir{j}, imageFiles{i}));
%             subplot(2,3,j), imshow(maskImg), title(maskSubDir{j}, 'Interpreter', 'none'),
%             clear maskImg;
%         end
%         saveas(h, fullfile(outputDir, strcat(imgName, '.png')));
%
% %          k = waitforbuttonpress
% %        close(h);
%     end
% end
% close all;


% mainDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Backup/2019-02-12_Desktop/Backup/Master-Thesis/datasets/test_dataset/tum/rgbd_dataset_freiburg1_desk/rgb';
%
% files = {'1305031459.259760.png', '1305031459.291690.png','1305031459.327632.png','1305031459.359651.png','1305031459.391667.png','1305031459.427646.png','1305031459.459679.png','1305031459.491555.png','1305031459.527594.png','1305031459.559647.png'};
%
% for n=1:length(files)
%     close all;
%     fileName = files{n};
%     label = fileName(1:end-4);
%
%     rgbImg = imread(fullfile(mainDir, fileName));
%     edgeImg = imread(fullfile(mainDir, 'rcf', 'rgb', fileName));
%     nmsImg = imread(fullfile(mainDir, 'rcf', 'rgb_nms', fileName));
%     nmsThinnedImg = imread(fullfile(mainDir, 'rcf', 'rgb_nms_thinned', fileName));
%
%     h(1) = figure('Position', get(0, 'Screensize')),
%     subplot(2,2,1), imshow(rgbImg), title(strcat('Input: ', fileName), 'Interpreter', 'none'),
%     subplot(2,2,2), imshow(edgeImg), title(strcat('Edges'), 'Interpreter', 'none'),
%     subplot(2,2,3), imshow(nmsImg), title(strcat('Edges NMS'), 'Interpreter', 'none'),
%     subplot(2,2,4), imshow(nmsThinnedImg), title(strcat('NMS Thinned'), 'Interpreter', 'none');
%
%     saveas(h, fullfile('/home/tom/University/datasets/', strcat(label, '.jpg')));
%
% end


% inputDir = '/home/tom/University/datasets/rgbd_dataset_freiburg1_desk/';
%
% if(~exist(inputDir, 'dir'))
%     error('Input directory "%s" does not exist.', inputDir)
% end
%
% fileFilter = {'jpeg', 'jpg', 'bmp', 'png'};
%
% rgbDir = fullfile(inputDir, 'rgb');
% maskDir = fullfile(inputDir, 'mask');
% maskThinned = fullfile(inputDir, 'mask_thinned');
%
% for n=1:length(fileFilter)
%     imageFiles = dir(fullfile(inputDir, strcat('*.', fileFilter{n})));
%     imageFiles = {imageFiles.name};
%     numImages = length(imageFiles);
%
%     for i=1:numImages, imgName = imageFiles{i}(1:end-4);
%
%
%         I = imread(fullfile(inputDir, imageFiles{i}));
%
%     end
% end


% close all;
% clearvars;
% 
% inputDir = '/home/tom//University/datasets/sample_rgbd_dataset_freiburg2_xyz/';
% outputDir = '/home/tom//University/datasets/sample_rgbd_dataset_freiburg2_xyz/unsupervised_comparison_25_33_50';
% 
% if(~exist(inputDir, 'dir'))
%     error('Input directory "%s" does not exist.', inputDir)
% end
% 
% ICG.existsOrCreate(outputDir, true)
% 
% fileFilter = {'jpeg', 'jpg', 'bmp', 'png'};
% 
% subDir = {'rgb', 'unsuvised_canny_asc_thres_25', 'unsuvised_canny_asc_thres_33', 'unsuvised_canny_asc_thres_50'}
% numSubDir = length(subDir);
% 
% for n=1:length(fileFilter)
%     imageFiles = dir(fullfile(inputDir, 'rgb', strcat('*.', fileFilter{n})));
%     imageFiles = {imageFiles.name};
%     numImages = length(imageFiles);
%       
%     for i=1:numImages, imgName = imageFiles{i}(1:end-4);
%                 
%         h(1) = figure('Position', get(0, 'Screensize'), 'Name', imageFiles{i}),
%                
%         for j=1:numSubDir;
%             tmp = imread(fullfile(inputDir, subDir{j}, imageFiles{i}));
%             subplot(1, numSubDir, j), imshow(tmp), title(subDir{j}, 'Interpreter', 'none'),
%         end
%         
%         saveas(h, fullfile(outputDir, strcat(imgName, '.jpg')));
%     end
% end


% close all;
% clearvars;
% 
% baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/desk_daylight_static/rgb/'
% inputDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/desk_daylight_static/mask/';
% outputDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/desk_daylight_static/results_thinned/';
% 
% if(~exist(inputDir, 'dir'))
%     error('Input directory "%s" does not exist.', inputDir)
% end
% 
% ICG.existsOrCreate(outputDir, true)
% 
% fileFilter = {'png'};
% 
% subDir = {'bdcn', 'rcf', 'hed', 'sf', 'sn'}
% numSubDir = length(subDir);
% 
% for n=1:length(fileFilter)
%     imageFiles = dir(fullfile(baseDir, strcat('*.', fileFilter{n})));
%     imageFiles = {imageFiles.name};
%     numImages = length(imageFiles);
%       
%     for i=1:10, imgName = imageFiles{i}(1:end-4);
%         if(exist(fullfile(outputDir, strcat(imgName, '.jpg')), 'file'))
%             continue
%         end
%         
%         h(1) = figure('Position', get(0, 'Screensize'), 'Name', imageFiles{i}),
%         tmp = imread(fullfile(baseDir, imageFiles{i}));
%         subplot(2, 3, 1), imshow(tmp), title('rgb', 'Interpreter', 'none'),
%         
%         for j=1:numSubDir;
%             tmp = imread(fullfile(inputDir, subDir{j}, imageFiles{i}));
%             
%             E = im2double(tmp);
%             I = E;
%             
%             if subDir{j} == "sn"
%                 indices = find(E < 0.6);
%                 E(indices) = 0;
%             end
%             ENms = ICG.nmsEdgeImage(E);
%             EThin = ICG.edgeThinning(ENms);
%             
%             subplot(2, 3, j+1), imshow(EThin), title(subDir{j}, 'Interpreter', 'none'),
%         end
%         
%         saveas(h, fullfile(outputDir, strcat(imgName, '.jpg')));
%         
%         if mod(n, 10) == 0
%             close all;
%         end
%     end    
% end
% close all

close all;
clearvars;

baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/depth/'
inputDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/';
outputDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/results_inpainting/';

if(~exist(inputDir, 'dir'))
    error('Input directory "%s" does not exist.', inputDir)
end

ICG.existsOrCreate(outputDir, true)

fileFilter = {'png'};

subDir = {'depth_inpaint_cv_talea', 'depth_inpaint_cv_ns', 'depth_inpaint_nans_m0', 'depth_inpaint_nans_m1', 'depth_inpaint_nans_m2', 'depth_inpaint_nans_m3', 'depth_inpaint_nans_m4', 'depth_inpaint_nans_m5'}
numSubDir = length(subDir);

for n=1:length(fileFilter)
    imageFiles = dir(fullfile(baseDir, strcat('*.', fileFilter{n})));
    imageFiles = {imageFiles.name};
    numImages = length(imageFiles);
      
    for i=1:10, imgName = imageFiles{i}(1:end-4);
        if(exist(fullfile(outputDir, strcat(imgName, '.jpg')), 'file'))
            continue
        end
        
        h(1) = figure('Position', get(0, 'Screensize'), 'Name', imageFiles{i}),
        tmp = depthToColormap(imread(fullfile(baseDir, imageFiles{i})));
        subplot(3, 3, 1), imshow(tmp), title('depth', 'Interpreter', 'none'),
        
        for j=1:numSubDir;
            tmp = depthToColormap(imread(fullfile(inputDir, subDir{j}, imageFiles{i})));
            
            subplot(3, 3, j+1), imshow(tmp), title(subDir{j}, 'Interpreter', 'none'),
        end
        
        saveas(h, fullfile(outputDir, strcat(imgName, '.jpg')));
                
        if mod(n, 10) == 0
            close all;
            return
        end        
    end    
end
close all

