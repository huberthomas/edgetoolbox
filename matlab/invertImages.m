function [ output_args ] = invertImages( inputDir, outputDir, varargin )
% Invert images from an input directory and store them to an output
% directory.
% @inputDir Image input directory.
% @outputDir Image output directory.
% @srcExtension Image extension that is looked for. Default: png.
% @dstExtension Output image extension. Default: png.

% imwrite(imcomplement(imread(fullfile(imgDir,'100007.png'))),fullfile(imgDir,'100007.png'))
% imwrite(imcomplement(imread(fullfile(imgDir,'128035.png'))),fullfile(imgDir,'128035.png'))
% imwrite(imcomplement(imread(fullfile(imgDir,'384022.png'))),fullfile(imgDir,'384022.png'))
% inputDir = 'D:/Nextcloud/master/master_thesis/assets/chapter03/kinect/desk_dimmed_static_ae_off/bdcn/'
% outputDir = inputDir

if(~exist(inputDir, 'dir'))
    error('Input directory "%s" does not exist.', inputDir)
end

if(~exist(outputDir, 'dir'))
    mkdir(outputDir);
%     fprintf('Output directory "%s" does not exist.\n', outputDir)
%     reply = input('Do you want to create it? Y/N [Y]: ', 's');
%     if isempty(reply) | strcmpi(reply, 'y') == 1
%         mkdir(outputDir);
%     end
end

srcExtension = 'png';
dstExtension = 'png';

if(nargin > 2)
    srcExtension = varargin{1}
end

if(nargin > 3)
    dstExtension = varargin{2}
end

images = dir(fullfile(inputDir, strcat('*.', srcExtension)));
images={images.name};
n=length(images);

for i=1:n, img = images{i}(1:end-4);
    fprintf('%d%%\n', (i/n)*100)
%     if(exist(fullfile(outputDir, strcat(img, '.', dstExtension)), 'file'))
%         continue
%     end
    
    I = imread(fullfile(inputDir, images{i}));
    I = imcomplement(I);
    figure(1), imshow(I), title(images{i})
%     w = waitforbuttonpress
    imwrite(I, fullfile(outputDir, strcat(img, '.', dstExtension)));
end

end



