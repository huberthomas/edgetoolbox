% convert images
folder = '/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/datasets/bsr_bsds500/BSR/BSDS500/data/groundTruth/test/'
files = dir(strcat(folder,'*.mat'));

for m = 1:length(files)
   fprintf('%s\n', files(m).name);
   filename = strcat(folder,files(m).name);
   f = load(filename, '-mat');
   
   img = f.groundTruth{1}.Boundaries;
   for n = 2:length(f.groundTruth)
       img = img + f.groundTruth{n}.Boundaries; 
   end   
   img = img/length(f.groundTruth);
   img = imcomplement(img);
   
   imwrite(img, fullfile(folder, 'inv', strcat(files(m).name(1:end-4),'.png')));
end

% files = dir(strcat(folder,'*.jpg'));
% 
% for m = 1:length(files)
%    fprintf('%s\n', files(m).name);
%    filename = strcat(folder,files(m).name);
%    img = imread(filename);
%    imwrite(img, strcat(folder,files(m).name(1:end-4),'.png'));
% end