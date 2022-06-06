clear;
clearvars;
clear all; close all;
subdirs = {
%     'desk_daylight_static', ...
%     'desk_dimmed_static', ...
%     'desk_neon_static', ...
%     'desk_daylight_static_ae_off', ...
%     'desk_daylight_static_ae_on', ...
%     'desk_dimmed_static_ae_off', ...
%     'desk_dimmed_static_ae_on', ...
%     'desk_neon_static_ae_off', ...
%     'desk_neon_static_ae_on'
    };

% figOutputDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/eval/kinect_desk_daylight_dimmed_neon';
figOutputDir = 'd:\Downloads\'
addpath('..\dependencies\edgeval\')
addpath('..\dependencies\edgeval\mex\')
addpath('..\dependencies\edgeval\utils\')

for m=1:length(subdirs)
    
    figName = subdirs{m};
    
    if ispc
        baseDir = fullfile('Z:/Master/datasets/', figName, 'mask');
    else
        baseDir = fullfile('/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/', figName, 'mask');
    end
    % baseDir = '/home/tom/University/repositories/projects/kinect-build/results/desk_neon_static_ae_on/mask';
    srcExtension = 'png';
    
    inputDirs = {
        'BDCN', ...
        'RCF', ...
        'HED', ...
        'DC',...
        'SE', ...
        'SN', ...
        'Canny'
        };
    
    result = [];
    
    displayResults = false;
    
    if displayResults
        figure('Name', 'Results', 'Renderer', 'painters', 'Position', [10 10 1600 600]);
    end
    
    for j=1:length(inputDirs)
        images = dir(fullfile(baseDir, inputDirs{j}, strcat('*.', srcExtension)));
        images = { images.name };
        n = length(images);
        
        %         suptitle(inputDirs{j});
        mergedE = zeros(480,640,1);
        for i=1:n, img = images{i}(1:end-4);
            fprintf('%d%% %s\n', floor(((i)/n)*100), fullfile(baseDir, inputDirs{j}, images{i}));
            
            E = imread(fullfile(baseDir, inputDirs{j}, images{i}));
            
            [w h c] = size(E);
            maxPixelNum = w * h;
            
            if(c > 1)
                fprintf('No binary image.');
                break;
            end
            
            if strcmp(inputDirs{j}, 'SN')
                indices = find(E < (0.7 * 255.0));
                E(indices) = 0;
                ENms = ICG.nmsEdgeImage(E);
                EThin = ICG.edgeThinning(ENms);
                ENms = ENms .* 255.0;
            else
                ENms = ICG.nmsEdgeImage(E);
                EThin = ICG.edgeThinning(ENms);
            end
            
            if displayResults
                subplot(1, 3, 1), imshow(E), title('Edge');
                subplot(1, 3, 2), imshow(ENms), title('NMS');
                subplot(1, 3, 3), imshow(EThin), title('Thinned');
                
                pause(5/1000);
            end
            mergedE=mergedE+EThin;
            numberOfTruePixels = sum(EThin(:));
            
            result(i, j) = (numberOfTruePixels/maxPixelNum)*100;
            %result(i, j) = numberOfTruePixels;
        end
%         figure(1), imshow(imcomplement(mergedE),[]), title(inputDirs{j});
%         figure(2), imshow(depthToColormap(mergedE),[]), title(inputDirs{j});
        mergedE=double(mergedE);
        mergedE=mergedE/max(max(mergedE));
%         imwrite(imcomplement(mergedE),fullfile(figOutputDir, strcat(figName, '_', inputDirs{j}, '.png')));
%         imwrite(depthToColormap(mergedE),fullfile(figOutputDir, strcat(figName, '_', inputDirs{j}, '_rgb.png')));
    end
    
    meanVal = mean(result, 1);
    stdVal = std(result, 1);
    
    plotTitle = 'Kinect Desk';
    
    h(1) = figure('Name', plotTitle, 'Renderer', 'painters', 'Position', [10 10 1600 600]);
    subplot(1,1, 1), plot(result);
    title('Found Edges', 'Interpreter', 'none');
    xlabel('Frames', 'Interpreter', 'none');
    ylabel('Counted Edges [%]', 'Interpreter', 'none');
    ylim([0 (max(max(result)) + 1)]);
    legend(inputDirs, 'Location','southeast');
    
%     subplot(1,2,2), errorbar(1:length(inputDirs), transpose(meanVal), transpose(stdVal), 'o','MarkerSize',3,...
%         'MarkerEdgeColor','red','MarkerFaceColor','red');
%     title('Mean/StdDev', 'Interpreter', 'none');
%     xlabel('Algorithm', 'Interpreter', 'none');
%     ylabel('Counted Edges [%]', 'Interpreter', 'none');
%     xlim([0 (length(inputDirs) + 1)]);
%     ylim([0 (max(max(result)) + 1)]);
%     set(gca,'xtick', [1:length(inputDirs)], 'xticklabel', inputDirs);
%     
    subt = suptitle(upper(strrep(figName,'_',' ')));
    subt.Interpreter = 'none';
    
%     saveas(h, fullfile(figOutputDir, strcat(figName, '.png')));
end