% Z:\Master\linux\University\repositories\projects_archive\nyudv2_data_label_extractor\data\label40
clear all
close all
inputDir='..\dependencies\nyudv2_data_label_extractor\data\label40';
outputDir='..\dependencies\nyudv2_data_label_extractor\data\label40rgb';
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
    figure(1), subplot(1,2,1),imshow(labeled,[]),subplot(1,2,2),imshow(labeledrgb,[]);
    imwrite(labeledrgb, fullfile(outputDir, images{i}));
end