clear all
close all
addpath('..\dependencies\edgeval\')
addpath('..\dependencies\edgeval\mex\')
addpath('..\dependencies\edgeval\utils\')
% inputDir = 'D:\Nextcloud\master\master_thesis\assets\chapter02\bsds500\'
% outputDir = 'D:\Nextcloud\master\master_thesis\assets\chapter02\bsds500\dist_transform\'
% outputDir = 'D:\Nextcloud\master\master_thesis\assets\chapter02\bsds500\edge_grad_mag\'

inputDir = 'D:\Nextcloud\master\master_thesis\assets\chapter02\tum_rgbd\fr2_xyz\bdcn';
outputDir = 'D:\Nextcloud\master\master_thesis\assets\chapter02\tum_rgbd\fr2_xyz\bdcn\dt';

srcExtension='png'
images = dir(fullfile(inputDir, strcat('*.', srcExtension)));
images={images.name};
n=length(images);

for i=1:n, img = images{i};
    inputFile = img;
    
    E = (imread(fullfile(inputDir, inputFile)));
    if false
        sigma =1;
        [Gx,Gy] = imgradientxy(imgaussfilt(rgb2gray(E),sigma));
        [Gmag,Gdir] = imgradient(Gx,Gy);
        
        Gx = rescale(Gx);
        Gy = rescale(Gy);
        Gmag = rescale(Gmag);
        Gdir = rescale(Gdir);
        
        imwrite(Gx, fullfile(outputDir, strcat('gx_',inputFile)));
        imwrite(Gy, fullfile(outputDir, strcat('gy_',inputFile)));
        imwrite(imcomplement(Gmag), fullfile(outputDir, strcat('gmag_',inputFile)));
        imwrite(Gdir, fullfile(outputDir, strcat('gdir_',inputFile)));
    end
    %% distance transform
    if true
        D = bwdist(~E,'euclidean');
        numPoints = 30;
        mx=uint8(size(D,2)/numPoints);
        my=uint8(size(D,1)/numPoints);
        for x=2:size(D,2)-1
            for y=2:size(D,1)-1
                %%
                % Jacobian JD of the DT: The derivative in the image plane at a point p
                %%
                Gx(y,x)=0.5*(D(y,x-1)-D(y,x+1));
                Gy(y,x)=0.5*(D(y-1,x)-D(y+1,x));
                if mod(x,mx) == 0 || x == 2
                    if mod(y,my) == 0 || y == 2
                        X(y,x)=x;
                        Y(y,x)=y;
                        K(y,x)=7*Gx(y,x);
                        L(y,x)=7*Gy(y,x);
                    end
                end
            end
        end
    end
    %% draw contours
    if false
        h2=figure(2),
        set(h2, 'units','normalized','outerposition',[0 0 1 1]);
        imshow(repmat(ICG.scaleLowHigh(D), [1 1 3])),
        hold on,
        colormap jet
        imcontour(D),
        hold off
        pause(2)
        contourImg = getframe(h2);
        contourImg=contourImg.cdata;
        if size(D,2) > size(D,1)
            contourImg=imcrop(contourImg,[838 165 2138 1427]);
        else
            contourImg=imcrop(contourImg,[1379 122 1056 1583]);
        end
        imwrite(contourImg, fullfile(outputDir, strcat('dist_grad_contour_',inputFile)));
        clear Gx Gy X Y K L;
    end
    %% draw gradients
    if true
        rgbD=cat(3,D,D,D);
        maxD=max(rgbD(:));
        rgbE = reshape([maxD maxD maxD],[1,1,3]) .* ~E;
        fusedDE=rgbD+rgbE;
        fusedDE=uint8((fusedDE/max(fusedDE(:)))*255);
        h=figure(1),
        set(h, 'units','normalized','outerposition',[0 0 1 1]);
        % set(h, 'Position', [0 0 size(Gx,1) size(Gx,2)]);  % to define Figure Properties
        imshow(fusedDE,[]),
        hold on,
        quiver(X,Y,K,L,0,'Color','yellow','Marker','.','MarkerSize',20,'ShowArrowHead','off','LineWidth',2);
        % subplot(3,1,1), imshow(D,[]);
        % subplot(3,1,2), imshow(Gmag,[]);
        % subplot(3,1,3), imshow(Gdir,[]);
        hold off
        % fusedDEImg = getimage(h);
        pause(1)
        fusedDEImg = getframe(h);
        close(h)
        fusedDEImg=fusedDEImg.cdata;
        
        if size(D,2) > size(D,1)
            fusedDEImg=imcrop(fusedDEImg,[838 165 2138 1427]);
        else
            fusedDEImg=imcrop(fusedDEImg,[1379 122 1056 1583]);
        end
        imwrite(fusedDEImg, fullfile(outputDir, strcat('dist_grad_',num2str(numPoints),'_',inputFile)));
        clear Gx Gy X Y K L;
    end
    %%
    % D=uint8((D/max(D(:)))*255);
    % rgbE = reshape(uint8([255 255 255]),[1,1,3]) .* uint8(~E);
    % rgbD = uint8(depthToColormap(D)*255);% rgbD = reshape(uint8([1 1 1]),[1,1,3]) .* uint8(D);
    % fused=rgbD+rgbE;
    % figure(2);
    % imshow(fused,[]);
    % hold on,
    % quiver(X,Y,K,L,0,'Color','yellow','Marker','o','MarkerSize',2);
    % hold off
    
    % imwrite(D, fullfile(outputDir, strcat('dist_',inputFile)));
    % imwrite(fused, fullfile(outputDir, strcat('rgb_dist_',inputFile)));
    % imwrite(fusedDEImg, fullfile(outputDir, strcat('dist_grad_',num2str(numPoints),'_',inputFile)));
    % imwrite(ICG.scaleLowHigh(Gx), fullfile(outputDir, strcat('dist_gx_',inputFile)));
    % imwrite(ICG.scaleLowHigh(Gy), fullfile(outputDir, strcat('dist_gy_',inputFile)));
end