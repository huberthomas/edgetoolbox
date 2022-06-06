function [ output_args ] = batchGrayToColormap( inputDir, outputDir, varargin )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if(~exist(inputDir, 'dir'))
    error('Input directory "%s" does not exist.', inputDir)
end

if(~ICG.existsOrCreate(outputDir, true))
    error('Invalid output directory: "%s".', outputDir);
    return;
end

srcExtension = 'png';
dstExtension = 'png';

images = dir(fullfile(inputDir, strcat('*.', srcExtension)));
images={images.name};
n=length(images);

for i=1:n, img = images{i}(1:end-4);    
    fprintf('%d%% %s\n', floor(((i)/n)*100), fullfile(outputDir, strcat(img, '.', dstExtension)));   
    
    E = imread(fullfile(inputDir, images{i}));

    if nargin > 2;
        E = imcomplement(E);
    end
    
    E = depthToColormap(E);
    
    imwrite(E, fullfile(outputDir, strcat(img, '.', dstExtension)));
end
end
