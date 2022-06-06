baseDir='Z:/Master/datasets/'

fileDirs={ ...
    'desk_daylight_static_ae_off/rgb', ...
    'desk_daylight_static_ae_on/rgb', ...
    'desk_dimmed_static_ae_off/rgb', ...
    'desk_dimmed_static_ae_on/rgb', ...
    'desk_neon_static_ae_off/rgb', ...
    'desk_neon_static_ae_on/rgb'
    }
outputDir='D:/Nextcloud/master/master_thesis/datasets/kinect/'

for m=1:length(fileDirs);
    imgDir = fullfile(baseDir, fileDirs{m}, '')
    outDir = fullfile(outputDir, fileDirs{m})
    
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
        if mod(i,15) == 0 | i == 1 | i == n% copy every 15th image
            fprintf('Status: %d%%\n', (i/n)*100)
            if(exist(fullfile(outDir, images{i}), 'file'))
                continue
            end
            copyFile(fullfile(imgDir, images{i}), fullfile(outDir, images{i}));
        end
    end
end

function [] = copyFile(inputFilePath,outputFilePath)
    imwrite(imread(inputFilePath), outputFilePath)
end

