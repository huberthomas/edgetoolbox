clear all;
clc;
inputDir = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/'
subDir = {
    'rgbd_dataset_freiburg1_360',...
    'rgbd_dataset_freiburg1_desk',...
    'rgbd_dataset_freiburg1_desk2',...
    'rgbd_dataset_freiburg1_floor',...
    'rgbd_dataset_freiburg1_plant',...
    'rgbd_dataset_freiburg1_room',...
    'rgbd_dataset_freiburg1_rpy',...
    'rgbd_dataset_freiburg1_teddy',...
    'rgbd_dataset_freiburg1_xyz',...
    'rgbd_dataset_freiburg2_360_hemisphere',...
    'rgbd_dataset_freiburg2_coke',...
    'rgbd_dataset_freiburg2_desk',...
    'rgbd_dataset_freiburg2_desk_with_person',...
    'rgbd_dataset_freiburg2_dishes',...
    'rgbd_dataset_freiburg2_flowerbouquet',...
    'rgbd_dataset_freiburg2_flowerbouquet_brownbackground',...
    'rgbd_dataset_freiburg2_large_no_loop',...
    'rgbd_dataset_freiburg2_metallic_sphere',...
    'rgbd_dataset_freiburg2_metallic_sphere2',...
    'rgbd_dataset_freiburg2_pioneer_360',...
    'rgbd_dataset_freiburg2_pioneer_slam',...
    'rgbd_dataset_freiburg2_xyz',...
    'rgbd_dataset_freiburg3_cabinet',...
    'rgbd_dataset_freiburg3_large_cabinet',...
    'rgbd_dataset_freiburg3_long_office_household',...
    'rgbd_dataset_freiburg3_nostructure_texture_far',...
    'rgbd_dataset_freiburg3_nostructure_texture_near_withloop',...
    'rgbd_dataset_freiburg3_sitting_static',...
    'rgbd_dataset_freiburg3_structure_notexture_far',...
    'rgbd_dataset_freiburg3_structure_notexture_near',...
    'rgbd_dataset_freiburg3_structure_texture_far',...
    'rgbd_dataset_freiburg3_structure_texture_near',...
    'rgbd_dataset_freiburg3_teddy',...
    'rgbd_dataset_freiburg3_walking_xyz'
    };

D = parallel.pool.DataQueue;
afterEach(D, @disp);
for i=1:length(subDir)
    subDir
    %% extract image pairs
    fid = fopen(fullfile(inputDir, subDir{i}, 'groundtruth_associated.txt'),'r');
    tline = fgetl(fid);
    
    recOutputDir = fullfile(inputDir, subDir{i}, 'reconstructed_depth');
    
    if(~ICG.existsOrCreate(recOutputDir, true))
        error('Invalid output directory: "%s".', recOutputDir);
    end
    
    rgbFiles = {};
    depthFiles = {};
    while ischar(tline);
        if ~contains(tline, '#');
            data = strsplit(tline, ' ');
            if ~exist(fullfile(inputDir, subDir{i}, data{10}), 'file')
                rgbFiles{end+1} = data{9};
                depthFiles{end+1} = data{10};
            end
        end
        %%
        tline = fgetl(fid);
    end
    
    
    total = length(rgbFiles)
    parfor j=1:total
        reconstructDepthImage(fullfile(inputDir, subDir{i}), rgbFiles{j}, depthFiles{j}, recOutputDir)
        send(D, fprintf('%d%% %s\n',(floor(j/total))*100, subDir{i}));
    end
end

function [] = reconstructDepthImage(baseDir, rgbFilename, depthFilename, outputDir)
%% read images
rgbImg = imread(fullfile(baseDir, 'rgb', rgbFilename));
depthImg = double(imread(fullfile(baseDir, 'depth', depthFilename)));

%% reconstruct undefined areas and write output
fdColorDepth = fill_depth_colorization(double(rgbImg)/255., depthImg);
imwrite(fdColorDepth, fullfile(outputDir, depthFilename));

% fdCrossDepth = fill_depth_cross_bf(rgbImg, depthImg);
% imwrite(fdCrossDepth, fullfile(outputDirI, depthFilename));
% figure(1),
% subplot(1,2,1), imshow(fdColorDepth,[]),
% subplot(1,2,2), imshow(fdCrossDepth,[]),
end