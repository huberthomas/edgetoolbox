clear all
close all
addpath('..\dependencies\edgeval\')
addpath('..\dependencies\edgeval\mex\')
addpath('..\dependencies\edgeval\utils\')
inputDir = 'D:\Nextcloud\master\master_thesis\assets\chapter02\bsds500\edge\'
inputFile = '81095.png'
I = double((imread(fullfile(inputDir, inputFile))));

if false
    % gradient and magnitude
    [Gx,Gy] = imgradientxy(I);
    [Gmag,Gdir] = imgradient(Gx,Gy);
    
    Gx = ICG.scaleLowHigh(Gx);
    Gy = ICG.scaleLowHigh(Gy);
    Gmag = ICG.scaleLowHigh(Gmag);
    Gdir = ICG.scaleLowHigh(Gdir);
    
%     imwrite(Gx, fullfile(inputDir, strcat('gx_',inputFile)));
%     imwrite(Gy, fullfile(inputDir, strcat('gy_',inputFile)));
%     imwrite(Gmag, fullfile(inputDir, strcat('gmag_',inputFile)));
%     imwrite(Gdir, fullfile(inputDir, strcat('gdir_',inputFile)));
    
    figure(1),
    subplot(2,2,1),imshow(Gx,[]),
    subplot(2,2,2),imshow(Gy,[]),
    subplot(2,2,3),imshow(Gmag,[]);
    subplot(2,2,4),imshow(Gdir,[]);
end
if false
    %% triangle filter
    r=3
    sigma=sqrt(r*(r+2)/6)
    f = [1:r r+1 r:-1:1]/(r+1)^2;
    f = conv2(f,f');
    f = (double(f) - min(f(:))) ./ (max(f(:)) - min(f(:)));
    figure, imshow(f);
%     imwrite(f, fullfile(inputDir, strcat('tri_filter_',inputFile)));
end
if false
    %% triangle/gaussian filtering
    inputDir = 'D:\Nextcloud\master\master_thesis\assets\chapter02\bsds500\'
    images = dir(fullfile(inputDir,'edge','*.png'));
    images={images.name};
    n=length(images);
    
    for i=1:n, img = images{i};
        fprintf('%d%% %s\n', floor(((i)/n)*100), fullfile(inputDir, img));
        I = imread(fullfile(inputDir,'edge',img));
        I = 1-(double(I)/255); % white background -> invert
        E = single(I);
        E = E ./ max(E(:));
        r=2
        sigma=sqrt(r*(r+2)/6);
        smoothedE = convTri(E,r);
        coloredSmoothedE=depthToColormap(smoothedE);
        coloredGaussE=depthToColormap(imgaussfilt(E,sigma));
        figure(1),
        subplot(1,3,1),imshow(depthToColormap(E),[])
        subplot(1,3,2),imshow(coloredSmoothedE,[])
        subplot(1,3,3),imshow(coloredGaussE,[])
        imwrite(coloredGaussE,fullfile(inputDir,'edge_colorized_smoothed_gaussian',img))
        imwrite(coloredSmoothedE,fullfile(inputDir,'edge_colorized_smoothed_triangle',img))
    end
end
if false 
    %% plot signal
    figure(1),
    hold on,
    m=80
    plot(E(m,:), 'color', 'blue', 'linewidth', '2'),
    plot(smoothedE(m,:), 'color', 'red'),
end
if false
    i=1
    for x=-7:0.1:7
%       f(i) = 0.5*(exp(x)-1)/(exp(x)+1);
        f(i) = sin(x)
%         f(i) = 2*exp(x)/((exp(x)+1)*(exp(x)+1))
        i=i+1;
    end
%     f=f+(abs(min(f(:))));
    f=ICG.scaleLowHigh(f);    
    [Gx,Gy] = imgradientxy(f);
    [Gxx,Gy] = imgradientxy(Gx);
    f=(f/max(f(:)))*0.26;
    figure(1),
    lineWidth=2;
    hold on,
    plot(f,'color','black','DisplayName','f','LineWidth',lineWidth),
    plot(Gx,'color','blue','DisplayName',"f_{x}",'LineWidth',lineWidth),
    plot(Gxx,'color','red','DisplayName',"f_{xx}",'LineWidth',lineWidth),
    legend, 
end

if true
    %% non maximum suppression
    I = 1-(double(I)/255); % white background -> invert
    E = single(I);
    E = E ./ max(E(:));
    r=4
%     sigma=sqrt(r*(r+2)/6);
    smoothedE = convTri(E,r);
%     smoothedE = imgaussfilt(E,sigma);
    [Ox, Oy] = gradient2(smoothedE);
    [Oxx, ~] = gradient2(Ox);
    [Oxy, Oyy] = gradient2(Oy);
    O = mod(atan(Oyy .* sign(-Oxy) ./ (Oxx+1e-5)), pi);
%     [Gmag,Gdir] = imgradient(Oxx,Oyy);
%     EE = edgesNmsMex(E, Gdir, 1, 5, 1.01, 4);
% EE=edge(E,'Canny');    
E = edgesNmsMex(E, O, 1, 5, 1.01, 8);    
    EE = edgesNmsMex(E, O, 5, 5, 1.01, 8);    
    E = imcomplement(E);
    EE = imcomplement(EE);
    
    %     subplot(2,2,1),imshow(E,[])
    %     subplot(2,2,2),imshow(imcomplement(E),[])
    %     sigma = 4; rg = ceil(3*sigma)
    %     tic, J1=imgaussfilt(E,sigma); toc
    %     r=sqrt(6*sigma*sigma+1)-1
    %     tic, J2=convTri(E,r,1,1); toc
    %     figure(1); imshow(J1,[]); figure(2); imshow(J2,[]); figure(3); imshow(abs(J2-J1),[]);
    %
    
%         figure(1)
% %         subplot(4,2,1),imshow(Ox,[])
%         subplot(4,2,2),imshow(Oy,[])
%         subplot(4,2,3),imshow(Oxx,[])
%         subplot(4,2,4),imshow(Oyy,[])
%         subplot(2,3,1),imshow(I,[]);
%         subplot(2,3,2),imshow(O,[]);
%         subplot(2,3,3),imshow(Gdir,[]);
        figure(3)
        subplot(1,3,1),imshow(E,[]);
        subplot(1,3,2),imshow(EE,[]);
        subplot(1,3,3),imshow(EE-E,[]);
%         subplot(1,2,2),imshow(EE,[]);
        
%         imwrite(ICG.scaleLowHigh(O), fullfile('D:\Nextcloud\master\master_thesis\assets\chapter02\bsds500\edge_grad_mag', strcat('nms_orientation_', inputFile)));
         imwrite(ICG.scaleLowHigh(E), fullfile('D:\Nextcloud\master\master_thesis\assets\chapter02\bsds500\edge_grad_mag', strcat('nms_raw_', inputFile)));

%         figure(2),
%         hold on,
%         m=80,
%         plot(E(m,100:250)/max(E(:))*0.2,'color','red')
%         plot(Ox(m,100:250),'color','blue')
%         plot(Oxx(m,100:200),'color','green')
%         plot(O(m,100:200)/max(O(:))*0.2,'color','black')
    %%
    
    
end
% sigma=2
% r=sqrt(6*sigma*sigma+1)-1
% r = 4;
% sigma=sqrt(r*(r+2)/6);
% sampling=1
% f = [1:r r+1 r:-1:1]/(r+1)^2;
% J = padarray(I,[r r],'symmetric','both');
% J = convn(convn(J,f,'valid'),f','valid');
% % conv2(f,f'))
% if(sampling>1), t=floor(sampling/2)+1; J=J(t:sampling:end-sampling+t,t:sampling:end-sampling+t,:);end

% rgbImage = int8((J/max(max(J)))*255)
% rgbImage = ind2rgb(rgbImage, jet(256));

% I = imgaussfilt(I,sigma);
% rgbI = depthToColormap(I);
% rgbJ = depthToColormap(J);
%
% figure(1),
% subplot(2,2,1),imshow(I,[]),
% subplot(2,2,2),imshow(abs(I-J),[]),
% subplot(2,2,3),imshow(rgbI,[]),
% subplot(2,2,4),imshow(rgbJ,[]);
%
% figure(2),
% m=80;
% plot(I(m,:)/max(max(I)), 'color', 'blue'),
% hold on,
% plot(J(m,:)/max(max(J)), 'color', 'red')
% plot(img(m,:)/max(max(img)), 'color', 'green')

% %UNTITLED Summary of this function goes here
% %   Detailed explanation goes here
% if(~exist(inputDir, 'dir'))
%     error('Input directory "%s" does not exist.', inputDir)
% end
%
% if(~ICG.existsOrCreate(outputDir, true))
%     error('Invalid output directory: "%s".', outputDir);
%     return;
% end
%
% srcExtension = 'png';
% dstExtension = 'png';
%
% images = dir(fullfile(inputDir, strcat('*.', srcExtension)));
% images={images.name};
% n=length(images);
%
% for i=1:n, img = images{i}(1:end-4);
%     fprintf('%d%% %s\n', floor(((i)/n)*100), fullfile(outputDir, strcat(img, '.', dstExtension)));
%
%     E = imread(fullfile(inputDir, images{i}));
%     E = imgaussfilt(I,2);
%     if nargin > 2;
%         E = imcomplement(E);
%     end
%
%     E = depthToColormap(E);
%
%     imwrite(E, fullfile(outputDir, strcat(img, '.', dstExtension)));
% end