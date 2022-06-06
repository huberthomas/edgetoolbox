function [ output_args ] = edgeSobel( inputDir, outputDir, varargin )
% Invert images from an input directory and store them to an output
% directory.
% @inputDir Image input directory.
% @outputDir Image output directory.
% @srcExtension Image extension that is looked for. Default: png.
% @dstExtension Output image extension. Default: png.

if(~exist(inputDir, 'dir'))
    error('Input directory "%s" does not exist.', inputDir)
end

if(~exist(outputDir, 'dir'))
    fprintf('Output directory "%s" does not exist.\n', outputDir)
    reply = input('Do you want to create it? Y/N [Y]: ', 's');
    if isempty(reply) | strcmpi(reply, 'y') == 1
        mkdir(outputDir);
    end
end

srcExtension = 'jpg';
dstExtension = 'png';
threshold = 0.6;
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
    %fprintf('%s\n', img)
    E = imread(fullfile(inputDir, images{i}));
    [w h c] = size(E);
    
    if(c==3)
        E = rgb2gray(E);
    end
    
    start = tic;
    edgeImage = edge(E, 'Sobel');
    elapsed = toc(start)
%     figure(1), imshow(edgeImage);
%     imwrite(I, fullfile(outputDir, strcat(img, '.', dstExtension)));
end

end



