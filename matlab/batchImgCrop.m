baseDir='D:\Nextcloud\master\master_thesis\assets\chapter04\'

fileDirs={ ...
    'frame_offset_eval'
    }
outputDir='D:\Nextcloud\master\master_thesis\assets\chapter04\frame_offset_eval\'

for m=1:length(fileDirs);
    imgDir = fullfile(baseDir, fileDirs{m}, '')
    outDir = fullfile(outputDir)
    
    if(~exist(imgDir, 'dir'))
        error('Input directory "%s" does not exist.', imgDir)
    end
    
    if(~exist(outDir, 'dir'))
        mkdir(outDir);
    end
    
    srcExtension = 'png';
    
    images = dir(fullfile(imgDir, strcat('*.', srcExtension)));
    images={images.name};
    n=length(images);
    
    for i=1:n, img = images{i}(1:end-4);
            fprintf('Status: %d%%\n', (i/n)*100)
%             if(exist(fullfile(outDir, images{i}), 'file'))
%                 continue
%             end
            cropFile(fullfile(imgDir, images{i}), fullfile(outDir, 'scaled', images{i}), [655 250 626 469], 1, 1);
            cropFile(fullfile(imgDir, images{i}), fullfile(outDir, 'refined', images{i}), [655 814 626 469], 1, 1);
            cropFile(fullfile(imgDir, images{i}), fullfile(outDir, 'rgb', images{i}), [655 1377 626 469], 0, 0);
    end
end

function [] = cropFile(inputFilePath,outputFilePath, rec, invert, thres)
    fprintf(inputFilePath, outputFilePath);
    crop = imcrop(imread(inputFilePath), rec);
    if invert == 1
        crop=imcomplement(crop);
    end
    
    if thres == 1
        crop = im2bw(rgb2gray(crop),0.5);
    end
    imwrite(crop, outputFilePath);
end

