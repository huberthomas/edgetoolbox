function [ output_args ] = edgeDetection( inputDir, outputDir, varargin )

if(~exist(inputDir, 'dir'))
    error('Input directory "%s" does not exist.', inputDir)
end

if(~exist(outputDir, 'dir'))
    fprintf('Output directory "%s" does not exist', outputDir)
    reply = input('Do you want to create it? Y/N [Y]: ', 's');
    if isempty(reply) | strcmpi(reply, 'y') == 1
        mkdir(outputDir);
    end
end

%mkdir(fullfile(outputDir, 'sobel'));
mkdir(fullfile(outputDir, 'canny'));
%mkdir(fullfile(outputDir, 'sobel/rgb'));
%mkdir(fullfile(outputDir, 'sobel/rgb_inv'));
mkdir(fullfile(outputDir, 'canny/rgb'));
mkdir(fullfile(outputDir, 'canny/rgb_inv'));

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
    fprintf('%s\n', fullfile(inputDir, images{i}))
    I = imread(fullfile(inputDir, images{i}));
    I = rgb2gray(I);
    %BW1 = edge(I,'canny');
    BW2 = edge(I,'canny', [0.1 0.3]);
    %figure(1), imshowpair(BW1,BW2,'montage'), title('Sobel Filter                                   Canny Filter');
    
    %w = waitforbuttonpress
    %imwrite(BW1, fullfile(outputDir, 'sobel/rgb/', strcat(img, '.', dstExtension)));
    imwrite(BW2, fullfile(outputDir, 'canny/rgb/', strcat(img, '.', dstExtension)));
    %imwrite(imcomplement(BW1), fullfile(outputDir, 'sobel/rgb_inv/', strcat(img, '.', dstExtension)));
    imwrite(imcomplement(BW2), fullfile(outputDir, 'canny/rgb_inv/', strcat(img, '.', dstExtension)));
end

end

