function [] = edgeThinning(inputDir, outputDir)
%%
% Thin edges of a binary image mask.
% param inputDir Input directory that contains image masks.
% param outputDir Destination directory of the thinned image masks.
% remark Contours must be white, background black.

if(~exist(inputDir, 'dir'))
    error('Input directory "%s" does not exist.', inputDir)
end

if(~ICG.existsOrCreate(outputDir, true))
    error('Invalid output directory: "%s".', outputDir);
    return
end

fileFilter = {'jpeg', 'jpg', 'bmp', 'png'};

for n=1:length(fileFilter)
    imageFiles = dir(fullfile(inputDir, strcat('*.', fileFilter{n})));
    imageFiles = {imageFiles.name};
    numImages = length(imageFiles);
    
    for i=1:numImages, imgName = imageFiles{i}(1:end-4);
        fprintf('%d%% %s\n', floor(((i)/numImages)*100), fullfile(outputDir, strcat(imgName, '.', fileFilter{n})));
        
        if exist(fullfile(outputDir, strcat(imgName, '.', fileFilter{n})), 'file')
%             continue
        end
        
        E = imread(fullfile(inputDir, imageFiles{i}));
        [w h c] = size(E);
            
        if(c > 1)
            E = rgb2gray(E);
        end
        invert=false;
        if(ICG.isBackgroundWhite(E))
            E = imcomplement(E);
            invert=true;
        end
        
        E = ICG.nmsEdgeImage(E);%fullfile(inputDir, imageFiles{i}));    
        E = ICG.edgeThinning(E);%fullfile(inputDir, imageFiles{i}));
        
        if(invert)
            E=imcomplement(E);
        end
        imwrite(E, fullfile(outputDir, strcat(imgName, '.', fileFilter{n})));
    end
end

end


