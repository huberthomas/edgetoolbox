function [ output_args ] = nyuDatasetViewer( inputDir, varargin )
%NYUDATASETVIEWER Summary of this function goes here
%   Detailed explanation goes here

if(~exist(inputDir, 'dir'))
    error('Input directory "%s" does not exist.', inputDir)
end

% if(~exist(outputDir, 'dir'))
%     fprintf('Output directory "%s" does not exist.\n', outputDir)
%     reply = input('Do you want to create it? Y/N [Y]: ', 's');
%     if isempty(reply) | strcmpi(reply, 'y') == 1
%         mkdir(outputDir);
%     end
% end

srcExtension = 'ppm';
dstExtension = 'png';

images = dir(fullfile(inputDir, strcat('*.', srcExtension)));
images={images.name};
n=length(images);

for i=1:n, img = images{i}(1:end-4);
    fprintf('%s\n', img)
    I = imread(fullfile(inputDir, images{i}));
    figure(1), imshow(I), title(images{i})
       
    %w = waitforbuttonpress
    if(nargin > 1)
        outputDir = varargin{1};
        
        if(~exist(outputDir, 'dir'))
            mkdir(outputDir);
        end
        imwrite(I, fullfile(outputDir, strcat(img, '.', dstExtension)));
    end
    
end

end

