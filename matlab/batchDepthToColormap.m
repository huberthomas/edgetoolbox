% Z:\Master\linux\University\repositories\projects_archive\nyudv2_data_label_extractor\data\label40
clear all
close all
% baseDir='Z:\Master\linux\University\repositories\projects_archive\nyudv2_data_label_extractor\data\'
% inputDir=fullfile(baseDir,'rawdepth');
% outputDir=fullfile(baseDir,'rawdepthrgb');

baseDir='Z:\Master\datasets\nyud_hdr_fusion\flicker_synthetic\flicker_1\'
baseDir='D:\Nextcloud\master\master_thesis\assets\chapter03\datasets\bsds500\test\sn';
inputDir=fullfile(baseDir,'depth');
outputDir=fullfile(baseDir,'depthrgb');

extension = 'png';

if(~ICG.existsOrCreate(outputDir, true))
    error('Invalid output directory: "%s".', outputDir);
    return;
end

images = dir(fullfile(inputDir, strcat('*.', extension)));
images={images.name};
n=length(images);

for i=1:n;    
    fprintf('%d%% %s\n', floor(((i)/n)*100), fullfile(inputDir, images{i}));   
    
    labeled=imread(fullfile(inputDir,images{i}));
    labeledrgb=depthToColormap(labeled);
%     figure(1), subplot(1,2,1),imshow(labeled,[]),subplot(1,2,2),imshow(labeledrgb,[]);
    imwrite(labeledrgb, fullfile(outputDir, images{i}));
end