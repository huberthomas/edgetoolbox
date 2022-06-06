close all;
clear all;
folder = 'D:\Nextcloud\master\master_thesis\assets\chapter03\img_pyr\example\'
files = dir(strcat(folder,'*.png'));

pyrDown(folder, files)

function []=pyrDown(folder,files)
%% save annotations to different files
threshold=0.3;
for m = 1:length(files)
    fprintf('%s\n', files(m).name);
    filename = strcat(folder,files(m).name);
    I = rgb2gray(imread(filename));
    E = edge(I,'Canny',threshold);
    I1 = impyramid(I, 'reduce');
    E1 = edge(I1,'Canny',threshold);
    I2 = impyramid(I1, 'reduce');
    E2 = edge(I2,'Canny',threshold);
    I3 = impyramid(I2, 'reduce');
    E3 = edge(I3,'Canny',threshold);
    
    figure, imshow(I,[]);
    figure, imshow(I1,[]);
    figure, imshow(I2,[]);
    figure, imshow(I3,[]);
    figure, imshow(E,[]);
    figure, imshow(E1,[]);
    figure, imshow(E2,[]);
    figure, imshow(E3,[]);
%     img = imcomplement(f.groundTruth{n}.Boundaries);
%     imwrite(img, strcat(folder,files(m).name(1:end-4),'_',num2str(n),'.png'));
    
end
end