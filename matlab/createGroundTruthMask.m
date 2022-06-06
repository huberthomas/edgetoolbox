close all;
clearvars;

addpath('..\dependencies\edgeval\')
addpath('..\dependencies\edgeval\mex\')
addpath('..\dependencies\edgeval\utils\')

baseDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/desk_neon_static/mask';
% baseDir = '/home/tom/University/repositories/projects/kinect-build/results/desk_neon_static_ae_on/mask';
srcExtension = 'png';

inputDirs = {
    'bdcn', ...
    'rcf', ...
    'hed', ...
    'sf', ...
    'canny'
    };

result = [];

% figure('Name', 'Results');

GT = containers.Map('KeyType','char','ValueType','any')

for j=1:length(inputDirs)
    images = dir(fullfile(baseDir, inputDirs{j}, strcat('*.', srcExtension)));
    images = { images.name };
    n = length(images);
    
    %     suptitle(inputDirs{j});
    
    for i=1:n, img = images{i}(1:end-4);
        fprintf('%d%% %s\n', floor(((i)/n)*100), fullfile(baseDir, inputDirs{j}, images{i}));
        
        E = imread(fullfile(baseDir, inputDirs{j}, images{i}));
        
        [w h c] = size(E);
        
        if(c > 1)
            fprintf('No binary image.');
            break;
        end
        
        ENms = ICG.nmsEdgeImage(E);
        EThin = ICG.edgeThinning(ENms);
        
        if GT.isKey(images{i})
            GT(images{i}) = GT(images{i}) + EThin;
        else
            GT(images{i}) = EThin;
        end
    end
end

n = length(GT);
keys = keys(GT);

optionsThin.P = 5;
optionsNms.t = 0.25;
optionsNms.m = 1.01;

figure('Name', 'Ground Truth'),
suptitle(''),
for i=1:1
    E = GT(keys{i});
    maxVal = max(max(E));
    E = E ./ maxVal;
    [counts, grayLevels] = imhist(E);
    ENms = ICG.nmsEdgeImage(E, optionsNms);    
    EThin = ICG.edgeThinning(ENms, optionsThin);
    
    subplot(2, 2, 1), imshow(E), title('Edge');
    subplot(2, 2, 2), bar(grayLevels, counts, 'EdgeColor', 'r', 'FaceColor', 'r', 'BarWidth', 0.95), title('Histogram');
    subplot(2, 2, 3), imshow(ENms), title('NMS');
    subplot(2, 2, 4), imshow(EThin), title('Thinned');
    pause(5/1000);    
end

