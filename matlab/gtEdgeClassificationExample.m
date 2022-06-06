clear all
close all
addpath('..\dependencies\edgeval\')
addpath('..\dependencies\edgeval\mex\')
addpath('..\dependencies\edgeval\utils\')

inputDir = 'D:\Nextcloud\master\master_thesis\assets\chapter02\bsds500\'
images = dir(fullfile(inputDir,'gt','*.png'));
images={images.name};
n=length(images);

for i=1:n, img = images{i};
    fprintf('%d%% %s\n', floor(((i)/n)*100), fullfile(inputDir, img));
    
    if false
        k = imread(fullfile(inputDir,'gt',img));
        
        
        
        return
    end
    
    if true
        gtDir = 'Z:\Master\datasets\bsr_bsds500\BSR\BSDS500\data\groundTruth\test\'
        imwrite(imcomplement(imread(fullfile(inputDir,'edge',img))),fullfile(inputDir,'edge',strcat('inv_',img)));
        E = fullfile(inputDir,'edge',strcat('inv_',img));
        G = fullfile(gtDir,strcat(img(1:end-4),'.mat'));
        options.out =''; %      - [''] optional output file for writing results
        options.thrs=99;  %     - [99] number or vector of thresholds for evaluation
        options.maxDist=0.0075; %   - [.0075] maximum tolerance for edge match
        options.thin = 1;      % - [1] if true thin boundary maps
        [thrs,cntR,sumR,cntP,sumP,V] = edgesEvalImg( E, G, options);
        maxCntRIndex = find(cntR==max(cntR(:)));
        maxCntPIndex = find(cntP==max(cntP(:)));
        maxSumRIndex = find(sumP==max(sumP(:)));
        maxSumPIndex = find(sumR==max(sumR(:)));
        
        minCntRIndex = find(cntR==min(cntR(:)));
        minCntPIndex = find(cntP==min(cntP(:)));
        minSumRIndex = find(sumP==min(sumP(:)));
        minSumPIndex = find(sumR==min(sumR(:)));
        
        figure(1),
        subplot(2,2,1), imshow(V(:,:,:,max(maxCntRIndex)));
        subplot(2,2,2), imshow(V(:,:,:,max(maxCntPIndex)));
        subplot(2,2,3), imshow(V(:,:,:,max(maxSumRIndex)));
        subplot(2,2,4), imshow(V(:,:,:,max(maxSumPIndex)));
        
        figure(2),
        subplot(2,2,1), imshow(V(:,:,:,min(minCntRIndex)));
        subplot(2,2,2), imshow(V(:,:,:,min(minCntPIndex)));
        subplot(2,2,3), imshow(V(:,:,:,min(minSumRIndex)));
        subplot(2,2,4), imshow(V(:,:,:,min(minSumPIndex)));
        
        P = cntP./sumP;
        R = cntR./sumR;
        absDiff=abs(P-R);
        EER = find(absDiff==min(absDiff(:)));
        figure(3), hold on, plot(R,P),plot(P(EER)),legend('precision','recall','EER')
        pause
    end
    
    
    if true
        % tp, fn, fp overlay
        GT = imread(fullfile(inputDir,'gt',img));
        rgbGT =  uint8(GT(:,:,[1 1 1]) * 255 );
        
        E = imread(fullfile(inputDir,'nms',img));
        
        GT = imcomplement(GT);
        %GT = im2bw(GT,0.5);
        %       GT = ICG.edgeThinning(GT);
        E = imcomplement(E);
        
        [height, width]=size(GT);
        
%         maxVal = max(GT(:));
%         normGT = (double(GT)/maxVal)*255;
        
        for x=1:width
            for y=1:height
                if and(GT(y,x)>0,E(y,x)>0) %hit true positive
                    rgbGT(y,x,1)=0;
                    rgbGT(y,x,2)=255; % green
                    rgbGT(y,x,3)=0;                    
                elseif and(GT(y,x)>0,E(y,x)==0) %missed false negative
                    rgbGT(y,x,1)=255;
                    rgbGT(y,x,2)=0; % red
                    rgbGT(y,x,3)=0;
                elseif and(GT(y,x)==0,E(y,x)>0) %wrong detected false positive
                    rgbGT(y,x,1)=0;
                    rgbGT(y,x,2)=0; % blue
                    rgbGT(y,x,3)=255;
                else %hit true negative
                    rgbGT(y,x,1)=255;
                    rgbGT(y,x,2)=255; %white
                    rgbGT(y,x,3)=255;
                end
            end
        end
        
%         figure(1),
%         subplot(1,3,1), imshow(GT,[]),
%         subplot(1,3,2), imshow(E,[]);
%         subplot(1,3,3), imshow(rgbGT,[]);

        imwrite(rgbGT,fullfile(inputDir,'statistics',strcat('rgba_',img)));
                pause
    end
end
