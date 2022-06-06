clear all
close all
inputDir = 'D:\Nextcloud\master\master_thesis\assets\chapter02\bsds500\edge\'
inputFile = '81095.png'
I = double(imcomplement(imread(fullfile(inputDir, inputFile))));
I=imgaussfilt(I,2);
% maxVal=max(max(I));
% I = I / maxVal;
% I = I * 100;
% n = size(unique(reshape(I,size(I,1)*size(I,2),size(I,3))),1)
% rgb = ind2rgb(gray2ind(I,255),jet(255));

rgb = label2rgb(gray2ind(I,255),jet(255));
figure(1),
imshow(rgb,[]); %,'Colormap',jet(64))

% DISPLAY 3D with underlying 2D image
% figure(1), 
% title(inputFile),
% surf(I,'FaceAlpha',0.75,'edgecolor','none','CDataMapping','scaled'),
% colormap jet,
% hold on,
% image(I,'CDataMapping','scaled');
% h = image(I); colormap(map)
% get(h,'CDataMapping')
% h = imagesc(I,[0 1]); colormap(map)
% get(h,'CDataMapping')