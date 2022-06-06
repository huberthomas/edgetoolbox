close all;
clear all;
% I = imread('/home/tom/University/datasets/rgbd_dataset_freiburg1_desk/rgb/1305031453.759694.png');
g = rgb2gray(imread('D:\Nextcloud\master\master_thesis\assets\chapter03\test_dataset\tum\rgbd_dataset_freiburg2_xyz\rgb\1311867277.948933.png'));
g = imgaussfilt(g,1);
[gx, gy] = imgradientxy(g, 'centraldifference');

[Gmag,Gdir] = imgradient(gx,gy);

figure(1),
subplot(2,3,1), imshow(g);
subplot(2,3,2), imshow(gx,[]);
subplot(2,3,3), imshow(gy,[]);
subplot(2,3,4), imshow(Gmag,[]);
figure(2), imshow(Gdir,[]);
Gdir = im2double(Gdir);
imwrite(Gdir/max(max(Gdir)), 'D:\Nextcloud\master\master_thesis\assets\chapter03\canny_steps\mag_1311867277.948933.png');