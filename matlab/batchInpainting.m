method = 5
inpaintingAction('/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/depth', ...
strcat('/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/depth_inpaint_nans_m', num2str(method)), method)

function [] = inpaintingAction(inputDir, outputDir, method)
%%
% Thin edges of a binary image mask.
% param inputDir Input directory that contains image masks.
% param outputDir Destination directory of the thinned image masks.

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
       
        I = imread(fullfile(inputDir, imageFiles{i}));
        D = double(I);
        % normalize for inpaint_nans
        D(D == 0) = NaN;
        maxD = max(D(:));
        D = inpaint_nans(D./maxD, method)*maxD;
        D = uint16(D);
        
%         diff = imsubtract(D, I);
%         figure(1), 
%         subplot(1,3,1),imshow(I, []), title('Input');%, colormap default;
%         subplot(1,3,2),imshow(D, []), title('Inpainted');%, colormap default;
%         subplot(1,3,3),imshow(diff, []), title('Difference');%, colormap default;
        imwrite(D, fullfile(outputDir, strcat(imgName, '.png')));
    end
end

end