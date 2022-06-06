clear all
addpath('..\dependencies\edgeval\mex\')
addpath('..\dependencies\edgeval\utils\')
inputDir = 'D:\Nextcloud\master\master_thesis\assets\chapter02\bsds500\'
inputDir = 'D:\Nextcloud\master\master_thesis\assets\chapter02\tum_rgbd\fr2_xyz\'
inputDir = 'D:\Nextcloud\master\master_thesis\assets\chapter02\hdr\flicker\'
% outputDir = 'D:\Nextcloud\master\master_thesis\assets\chapter02\bsds500\dist_transform\'
% outputDir = 'D:\Nextcloud\master\master_thesis\assets\chapter02\bsds500\'

srcExtension='png'
images = dir(fullfile(inputDir, strcat('*.', srcExtension)));
images={images.name};
n=length(images);
 
for i=1:n, img = images{i};
    close all
    inputFile = img;
    
    I = rgb2gray(imread(fullfile(inputDir, inputFile)));
    [counts,x] = imhist(I,255);
    T = otsuthresh(counts)
    BW = imbinarize(I,T);
    
    counts = counts/max(counts(:));
    figure(1),
    subplot(1,3,1), imshow(I,[]),   
    subplot(1,3,3), imshow(BW,[]),
    
    h2=figure(2),
    hold on,
    set(gca,'xtick',[],'ytick',[]);
    xlabel('Intensities')
    ylabel('Pixel')
    stem(x,counts,'Marker','none','LineWidth',1),
    stem(255*T,1,'Marker','none','Color','red','LineStyle','--','LineWidth',1),
    hold off
    set(h2, 'units','normalized','outerposition',[0.25 0.25 0.25 0.25]);
    frame=getframe(h2);
    frame=frame.cdata;
    
    imwrite(I,fullfile(inputDir, strcat('gray_',inputFile)));
    imwrite(BW,fullfile(inputDir, strcat('otsu_',inputFile)));
    return
    pause
    clear counts x I BW;
end
