inputDir='D:\Nextcloud\master\master_thesis\assets\chapter02\bsds500\thinned'
outputDir='D:\Nextcloud\master\master_thesis\assets\chapter02\bsds500\thinned_cleaned'

if(~exist(inputDir, 'dir'))
    error('Input directory "%s" does not exist.', inputDir)
end

if(~exist(outputDir, 'dir'))
%     mkdir(outputDir);
    fprintf('Output directory "%s" does not exist.\n', outputDir)
    reply = input('Do you want to create it? Y/N [Y]: ', 's');
    if isempty(reply) | strcmpi(reply, 'y') == 1
        mkdir(outputDir);
    else
        exit(0)
    end
end

srcExtension = 'png';
dstExtension = 'png';

images = dir(fullfile(inputDir, strcat('*.', srcExtension)));
images={images.name};
n=length(images);

for i=1:n, img = images{i}(1:end-4);
    fprintf('Status: %d%%\n', (i/n)*100)
%     if(exist(fullfile(outputDir, strcat(img, '.', dstExtension)), 'file'))
%         continue
%     end
    I = imread(fullfile(inputDir, images{i}));
    % start isolated object cleaning
    [outImg, outImgSeg]=isolatedObjectCleaning(imcomplement(I));
    imwrite(outImg, fullfile(outputDir, strcat(img, '.', dstExtension)));
    imwrite(outImgSeg, fullfile(outputDir, strcat(img, '_seg.', dstExtension)));
    % end isolated object cleaning
    
end

function [outImg, outImgSeg ] = isolatedObjectCleaning(I)
    % start isolated object cleaning
    areaThreshold=30;
%     I = bwmorph(I, 'thin', Inf); % thinning
    E = bwareaopen(I, areaThreshold);
    rgb=I-E;
    outImg=imcomplement(E);
    rgb=ICG.convertBinImage2RGB(rgb,1,255,0,0);
    outImgSeg=imcomplement(E+imcomplement(rgb));
    figure(1), 
    subplot(1,2,1), imshow(I,[]),
    subplot(1,2,2), imshow(outImg,[]);
    % end isolated object cleaning
end