clear all;
close all;

subDirs={ ...
    'rgbd_dataset_freiburg1_rpy', ...
        'rgbd_dataset_freiburg1_teddy', ...
        'rgbd_dataset_freiburg2_desk', ...
        'rgbd_dataset_freiburg2_xyz', ...
        'rgbd_dataset_freiburg2_dishes', ...
        'rgbd_dataset_freiburg3_nostructure_texture_far', ...
        'rgbd_dataset_freiburg3_structure_notexture_near', ...
        'rgbd_dataset_freiburg3_structure_texture_far', ...
        'rgbd_dataset_freiburg3_long_office_household'
    }
outputDir='D:\Nextcloud\master\master_thesis\assets\chapter04\num_edges\'
for k=1:length(subDirs);
    
    
    baseDir=strcat('Z:\Master\datasets\all_thinned\', subDirs{k}, '\level0\');
    
    fileDirs={ ...
        'bdcn_sgd_singleScale_gpu_tum_30k_augplus', ...
        'bdcn', ...
        'canny'
        %     'stableEdgesFo2'
        }
    
    result = [];
    resMean = [];
    resStdDev = [];
    for m=1:length(fileDirs);
        
        imgDir = fullfile(baseDir, fileDirs{m}, '')
        outDir = fullfile(outputDir, strcat(subDirs{k}, fileDirs{m},'.csv'))
        
        
        if(~exist(imgDir, 'dir'))
            error('Input directory "%s" does not exist.', imgDir)
        end
        
        %     if(~exist(outDir, 'dir'))
        %         mkdir(outDir);
        %     end
        
        srcExtension = 'png';
        
        images = dir(fullfile(imgDir, strcat('*.', srcExtension)));
        images={images.name};
        n=length(images);
        
        
        
        for i=1:n, img = images{i}(1:end-4);
            %         if mod(i,15) == 0 | i == 1 | i == n% copy every 15th image
            fprintf('Status: %d%%\n', (i/n)*100)
            %             if(exist(fullfile(outDir, images{i}), 'file'))
            %                 continue
            %             end
            E = imread(fullfile(baseDir, fileDirs{m}, images{i}));
            if max(max(E(:)))>1;
                E = im2bw(E,0.5);
            end
            numberOfTruePixels=sum(E(:));
            [w h c] = size(E);
            maxPixelNum = w * h;
            %             copyFile(fullfile(imgDir, images{i}), fullfile(outDir, images{i}));
            result(i, m) = (numberOfTruePixels/maxPixelNum)*100;
            res{k}=result;
            
            
            %             fprintf(fileID,'%s;%s;%s\n','img','numberOfTruePixels','result(i, m)');
            %             fprintf(fileID,'%s;%6.2f;%12.2f\n',images{i},numberOfTruePixels,result(i,m));
            %         end
        end
        
        %         meanVal = mean(result(:,m));
        %         stdVal = std(result(:,m));
        
        
        %
        %     plotTitle = 'Kinect Desk';
        %
        %     h(1) = figure('Name', plotTitle, 'Renderer', 'painters', 'Position', [10 10 1600 600]);
        %     subplot(1,1, 1), plot(result);
        %     title('Found Edges', 'Interpreter', 'none');
        %     xlabel('Frames', 'Interpreter', 'none');
        %     ylabel('Counted Edges [%]', 'Interpreter', 'none');
        %     ylim([0 (max(max(result)) + 1)]);
        %     legend(inputDirs, 'Location','southeast');
        %
        % %     subplot(1,2,2), errorbar(1:length(inputDirs), transpose(meanVal), transpose(stdVal), 'o','MarkerSize',3,...
        % %         'MarkerEdgeColor','red','MarkerFaceColor','red');
        % %     title('Mean/StdDev', 'Interpreter', 'none');
        % %     xlabel('Algorithm', 'Interpreter', 'none');
        % %     ylabel('Counted Edges [%]', 'Interpreter', 'none');
        % %     xlim([0 (length(inputDirs) + 1)]);
        % %     ylim([0 (max(max(result)) + 1)]);
        % %     set(gca,'xtick', [1:length(inputDirs)], 'xticklabel', inputDirs);
        % %
        %     subt = suptitle(upper(strrep(figName,'_',' ')));
        %     subt.Interpreter = 'none';
        
        
    end
    
%     [w h c] = size(result);
%     if m==3;
% %         outDir = fullfile(outputDir, strcat(subDirs{k}, '.csv'))
% %         fileID = fopen(outDir,'w');
% %         fprintf(fileID,'%s;%s;%s\n','BDCN_stable','BDCN','Canny');
% %         [w h c] = size(result);
% %         for m=1:length(fileDirs);
% %             for i=1:w
% %                 fprintf(fileID,'%6.2f;%6.2f;%6.2f\n',result(i, 1) ,result(i, 2) ,result(i, 3) );
% %             end
% %         end
% %         fclose(fileID);
%         outDir = fullfile(outputDir, strcat(subDirs{k}, '_',num2str(w),'_files.csv'))
%         fileID = fopen(outDir,'w');
%         fclose(fileID);
% 
%         outDir = fullfile(outputDir, strcat(subDirs{k}, '_mean_stddev_new.csv'))
%         fileID = fopen(outDir,'w');
%         fprintf(fileID,'%s;%s;%s\n','BDCN_stable','BDCN','Canny');
%         fprintf(fileID,'%6.6f;%6.6f;%6.6f\n',mean(result(:, 1)) ,mean(result(:, 2)) ,mean(result(:, 3) ));
%         fprintf(fileID,'%6.6f;%6.6f;%6.6f\n',std(result(:, 1)) ,std(result(:, 2)) ,std(result(:, 3) ));
%         fclose(fileID);
%     end
end


legendTitles={ ...
    'fr1/rpy', ...
        'fr1/teddy', ...
        'fr2/desk', ...
        'fr2/xyz', ...
        'fr2/dishes', ...
        'fr3/no structure no texture far', ...
        'fr3/structure no texture near', ...
        'fr3/structure texture far', ...
        'fr3/long office household'
    }

fileDirs={ ...
        'BDCN_{stable}', ...
        'BDCN', ...
        'Canny'
        %     'stableEdgesFo2'
        }

figure(1),hold on,
subplot(1,1,1), errorbar((3*0)+(1:length(fileDirs)), transpose(mean(res{1})), transpose(std(res{1})), 'o','MarkerSize',3,'MarkerEdgeColor','red','MarkerFaceColor','red'), whitebg('w');;
subplot(1,1,1), errorbar((3*1)+(1:length(fileDirs)), transpose(mean(res{2})), transpose(std(res{2})), 'o','MarkerSize',3,'MarkerEdgeColor','red','MarkerFaceColor','red'), whitebg('w');;
subplot(1,1,1), errorbar((3*2)+(1:length(fileDirs)), transpose(mean(res{3})), transpose(std(res{3})), 'o','MarkerSize',3,'MarkerEdgeColor','red','MarkerFaceColor','red'), whitebg('w');;
subplot(1,1,1), errorbar((3*3)+(1:length(fileDirs)), transpose(mean(res{4})), transpose(std(res{4})), 'o','MarkerSize',3,'MarkerEdgeColor','red','MarkerFaceColor','red'), whitebg('w');;
subplot(1,1,1), errorbar((3*4)+(1:length(fileDirs)), transpose(mean(res{5})), transpose(std(res{5})), 'o','MarkerSize',3,'MarkerEdgeColor','red','MarkerFaceColor','red'), whitebg('w');;
subplot(1,1,1), errorbar((3*5)+(1:length(fileDirs)), transpose(mean(res{6})), transpose(std(res{6})), 'o','MarkerSize',3,'MarkerEdgeColor','red','MarkerFaceColor','red'), whitebg('w');;
subplot(1,1,1), errorbar((3*6)+(1:length(fileDirs)), transpose(mean(res{7})), transpose(std(res{7})), 'o','MarkerSize',3,'MarkerEdgeColor','red','MarkerFaceColor','red'), whitebg('w');;
subplot(1,1,1), errorbar((3*7)+(1:length(fileDirs)), transpose(mean(res{8})), transpose(std(res{8})), 'o','MarkerSize',3,'MarkerEdgeColor','red','MarkerFaceColor','red'), whitebg('w');;
subplot(1,1,1), errorbar((3*8)+(1:length(fileDirs)), transpose(mean(res{9})), transpose(std(res{9})), 'o','MarkerSize',3,'MarkerEdgeColor','red','MarkerFaceColor','red'), whitebg('w');;

% title('Mean/StdDev', 'Interpreter', 'none');
% xlabel('Algorithm', 'Interpreter', 'none');
ylabel('Number of Edges [%]', 'Interpreter', 'none');
xlim([0 (27+1)]);
ylim([0 (max(max(res{1})) + 1)]);
set(gca,'xtick', [1:27], 'xticklabel', fileDirs, 'XTickLabelRotation',90);

legend(legendTitles, 'Location','southeast');
