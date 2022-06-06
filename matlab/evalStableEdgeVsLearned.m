clear all;

% createMat();
copyProcessedFiles(1000);
benchmark();

%%
function [] = benchmark()
addpath('..\dependencies\edgeval\')
addpath('..\dependencies\edgeval\mex\')
addpath('..\dependencies\edgeval\utils\')

gtDir='Z:\Master\eval\frameworkVsLearned\gt_framework';
dirs = {'Z:\Master\eval\frameworkVsLearned\test_learned'};
names = {'Trained'};
benchmarkBoundary(dirs{1}, gtDir, names{1});

nmsDirs = {}
for i=1:length(dirs)
    nmsDirs{i}=strcat(dirs{i},'-nms');
end

edgesEvalPlot(nmsDirs, names)
end

%%
function [] = copyProcessedFiles(numOfCandidates)
data=load('Z:\Master\train\stableEdges2\gt\allFiles.mat');
dataLength=length(data.files);
d=randperm(dataLength,numOfCandidates);

evalDir='Z:\Master\eval\frameworkVsLearned\';
gtDir = 'Z:\Master\train\stableEdges2\gt\';
learnedDir='Z:\Master\datasets\all_thinned\';
save(fullfile(evalDir,'randNum'),'d');

if(~exist(fullfile(evalDir,'gt_framework'), 'dir'))
        mkdir(fullfile(evalDir,'gt_framework'));
end
if(~exist(fullfile(evalDir,'test_learned'), 'dir'))
        mkdir(fullfile(evalDir,'test_learned'));
end

for i=1:length(d)
    cellData=data.files(i);
    cellData=cellData{1};
    copyfile(fullfile(gtDir,cellData{1},strcat(cellData{2},'.png')),fullfile(evalDir,'gt_framework',strcat(cellData{2},'.png')));
    copyfile(fullfile(gtDir,cellData{1},strcat(cellData{2},'.mat')),fullfile(evalDir,'gt_framework',strcat(cellData{2},'.mat')));
    copyfile(fullfile(learnedDir,cellData{1},'level0','bdcn_sgd_singleScale_gpu_tum_30k_augplus',strcat(cellData{2},'.png')),fullfile(evalDir,'test_learned',strcat(cellData{2},'.png')));
end
end

%%
function [] = createMat()
% ground truth
baseDir = 'Z:\Master\train\stableEdges2\gt\';
outputDir='Z:\Master\train\stableEdges2\gt\'

fileDirs = { ...
    'rgbd_dataset_freiburg1_360', ...
    'rgbd_dataset_freiburg1_desk', ...
    'rgbd_dataset_freiburg1_desk2', ...
    'rgbd_dataset_freiburg1_floor', ...
    'rgbd_dataset_freiburg1_plant', ...
    'rgbd_dataset_freiburg1_room', ...
    'rgbd_dataset_freiburg1_rpy', ...
    'rgbd_dataset_freiburg1_teddy', ...
    'rgbd_dataset_freiburg1_xyz', ...
    'rgbd_dataset_freiburg2_360_hemisphere', ...
    'rgbd_dataset_freiburg2_coke', ...
    'rgbd_dataset_freiburg2_desk', ...
    'rgbd_dataset_freiburg2_desk_with_person', ...
    'rgbd_dataset_freiburg2_dishes', ...
    'rgbd_dataset_freiburg2_flowerbouquet', ...
    'rgbd_dataset_freiburg2_flowerbouquet_brownbackground', ...
    'rgbd_dataset_freiburg2_large_no_loop', ...
    'rgbd_dataset_freiburg2_metallic_sphere', ...
    'rgbd_dataset_freiburg2_metallic_sphere2', ...
    'rgbd_dataset_freiburg2_pioneer_360', ...
    'rgbd_dataset_freiburg2_pioneer_slam', ...
    'rgbd_dataset_freiburg2_xyz', ...
    'rgbd_dataset_freiburg3_cabinet', ...
    'rgbd_dataset_freiburg3_large_cabinet', ...
    'rgbd_dataset_freiburg3_long_office_household', ...
    'rgbd_dataset_freiburg3_nostructure_texture_far', ...
    'rgbd_dataset_freiburg3_nostructure_texture_near_withloop', ...
    'rgbd_dataset_freiburg3_sitting_static', ...
    'rgbd_dataset_freiburg3_structure_notexture_far', ...
    'rgbd_dataset_freiburg3_structure_notexture_near', ...
    'rgbd_dataset_freiburg3_structure_texture_far', ...
    'rgbd_dataset_freiburg3_structure_texture_near', ...
    'rgbd_dataset_freiburg3_teddy', ...
    'rgbd_dataset_freiburg3_walking_xyz'
    }

files = [];
count = 1;
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
        fprintf('Status: %d%%\n', (i/n)*100)
        %         if(exist(fullfile(outDir, strcat(img,'.mat')), 'file'))
        %             continue
        %         end
        files{count}={fileDirs{m}, img}; count=count+1;
        %% create GT mat file
        createMatFile(fullfile(imgDir, images{i}), fullfile(outDir, strcat(img,'.mat')));
    end
end

save(fullfile(baseDir,'\allFiles.mat'),'files')

end

%%
function [] = createMatFile(inputFilePath, outputFilePath)
img=logical(imread(inputFilePath));
[h,w]=size(img);
cellData=struct('Segmentation',uint16(zeros(h,w)),'Boundaries',img);
groundTruth={cellData};
save(outputFilePath,'groundTruth');
end
