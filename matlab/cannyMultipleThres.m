clear all;

baseDir='Z:\Master\datasets\bsr_bsds500\BSR\BSDS500\data\images\'

fileDirs={ ...
    'test'
    }
outputDir='Z:\Master\datasets\bsr_bsds500\BSR\BSDS500\data\images\test_png\canny\'

for m=1:length(fileDirs);
    imgDir = fullfile(baseDir, fileDirs{m})
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
%         if mod(i,15) == 0 | i == 1 | i == n% copy every 15th image
            fprintf('Status: %d%%\n', (i/n)*100)
            if(exist(fullfile(outDir, images{i}), 'file'))
                continue
            end
            cannyFile(fullfile(imgDir, images{i}), fullfile(outDir, images{i}));
%         end
    end
end

function [] = cannyFile(inputFilePath,outputFilePath)
I = rgb2gray(imread(inputFilePath));
method='Canny';
E = zeros(size(I));
for i=1:1:254
    %     if mod(i,25)==0
    E=E+edge(I,method,i/255.0);
    %     end
end
E=E/max(max(E(:)));
% figure(1), imshow(E,[]);
if length(outputFilePath)>0
    imwrite(E,outputFilePath);
end

end














