clear;
clearvars;
close all;

addpath('..\dependencies\edgeval\')
addpath('..\dependencies\edgeval\mex\')
addpath('..\dependencies\edgeval\utils\')


inputDirs = {
    '/home/tom/Pictures/sharp_net/mask', ...
    }

baseDir = 'D:\Nextcloud\master\master_thesis\assets\chapter03\test_dataset\';

inputDirs = {
    fullfile(baseDir, 'hdr_fusion\flicker_synthetic\flicker_1\sharp_net\mask'),
    fullfile(baseDir, 'hdr_fusion\smooth_synthetic\flicker_2\sharp_net\mask')
    fullfile(baseDir, 'nyu_depth_v2\basements\basement_001c\sharp_net\mask')
    fullfile(baseDir, 'nyu_depth_v2\cafe\cafe_0001c\sharp_net\mask')
    fullfile(baseDir, 'nyu_depth_v2\classrooms\classroom_0014\sharp_net\mask')
    fullfile(baseDir, 'tum\rgbd_dataset_freiburg1_desk\rgb\sharp_net\mask')
    fullfile(baseDir, 'tum\rgbd_dataset_freiburg1_xyz\rgb\sharp_net\mask')
    fullfile(baseDir, 'tum\rgbd_dataset_freiburg2_xyz\rgb\sharp_net\mask')
}

displayResults = true;

if displayResults
    figure('Name', 'Results', 'Renderer', 'painters', 'Position', [10 10 1600 600]);
end

thres = 0.6

inputDirs = {
 'Z:\Master\datasets\desk_dimmed_static_ae_off\mask\sn\'   
}

for j=1:length(inputDirs)
    images = dir(fullfile(inputDirs{j}, strcat('*.png')));
    images = { images.name };
    n = length(images);
    
    %         suptitle(inputDirs{j});
    outDir = fullfile(inputDirs{j}, '../nms-inv')
    mkdir(outDir);
    
    for i=1:n, img = images{i}(1:end-4);
        fprintf('%d%% %s\n', floor(((i)/n)*100), fullfile(inputDirs{j}, images{i}));
              
        E = imread(fullfile(inputDirs{j}, images{i}));
        
        [w h c] = size(E);
        
        if(c > 1)
            fprintf('No binary image.');
            break;
        end
        
        E = im2double(E);
        I = E;
        indices = find(E < thres);
        E(indices) = 0;
        ENms = ICG.nmsEdgeImage(E);
        EThin = ICG.edgeThinning(ENms);
        EThin = ICG.edgeThinning(EThin);
          
        if displayResults
            subplot(2, 2, 1), imshow(I), title('Input');
            subplot(2, 2, 2), imshow(E), title(strcat('Threshold', {' '}, num2str(thres)));
            subplot(2, 2, 3), imshow(ENms), title('NMS');
            subplot(2, 2, 4), imshow(EThin), title('Thinned');
            
%             pause(5/1000);
%         pause
        end
         imwrite(imcomplement(EThin), fullfile(outDir,  images{i}));
    end
end
