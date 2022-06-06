clear all;
close all;
baseDir='Z:/Master/datasets/'

fileDirs={ ...
%     'desk_daylight_static_ae_off/rgb', ...
    %     'desk_daylight_static_ae_on/rgb', ...
    %     'desk_dimmed_static_ae_off/rgb', ...
    %     'desk_dimmed_static_ae_on/rgb', ...
        'desk_neon_static_ae_off/rgb', ...
%         'desk_neon_static_ae_on/rgb'
    }
outputDir='D:/Nextcloud/master/master_thesis/datasets/kinect/'
outputDir='D:/Nextcloud/master/master_thesis/assets/chapter03/kinect/'
% outputDir=fullfile(baseDir,'out');
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
    [h,w,c]=size(imread(fullfile(imgDir, images{1})));
    total = zeros(h,w,n);
    totalMean=zeros(h,w,1);
    totalStd=zeros(h,w,1);
    for i=1:n, img = images{i}(1:end-4);
        %         if mod(i,50) == 0 | i == 1 | i == n% copy every 15th image
        fprintf('Status: %d%%\n', (i/n)*100)
        %             if(exist(fullfile(outDir, images{i}), 'file'))
        %                 continue
        %             end
        img = double(rgb2gray(imread(fullfile(imgDir, images{i}))));
        total(:,:,i) = img;
        totalMean=totalMean+img;
        %             figure(2), imshow(img,[]);
        %             imwrite(fullfile(outDir, images{i}));
        %         end
    end
    totalMean=totalMean/n;
    for i=1:n
%         stdDev=(total(:,:,i)-totalMean).^2;
        stdDev=abs(total(:,:,i)-totalMean);
        totalStd=totalStd+stdDev;
    end
%     totalStd=sqrt(totalStd/n);
    totalStd=(totalStd/n);
    maxStdDevVal=max(max(totalStd))
    minStdDevVal=min(min(totalStd))
    meanVal=mean(totalStd(:))
    medianVal=median(totalStd(:))
    invTotalStd=(imcomplement(totalStd));
    
    
%     totalStd8=int8((totalStd));
    counts = hist(totalStd(:),int8(maxStdDevVal));
    counts = counts*100/(w*h);
    figure(1),bar((1:int8(maxStdDevVal)),counts);
    
%     [counts,binLocations] = hist(totalStd);
%     binLocations(255)=0;
%     binLocations=(binLocations*255);
%     counts(256)=0;
%     figure(1), subplot(1,2,1), imshow(totalStd), subplot(1,2,2), plot(binLocations,counts);
    %     figure(1),imshow(depthToColormap(totalStd),[]);
    %     imwrite(invTotalStd, fullfile(outDir, strcat('all_100_stddev_', num2str(minStdDevVal), '-', num2str(maxStdDevVal), '.png')));
    %     imwrite(imcomplement(invTotalStd), fullfile(outDir, strcat('all_100_inv_stddev_', num2str(minStdDevVal), '-', num2str(maxStdDevVal), '.png')));
end

