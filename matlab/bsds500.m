% convert images
% folder = 'D:\\Master-Thesis\\datasets\\BSR_bsds500\\BSR\\BSDS500\\data\\groundTruth\\test\\'
folder = 'D:\\Nextcloud\\master\\master_thesis\\assets\\chapter03\\bsds500\\single_annotations\\'
files = dir(strcat(folder,'*.mat'));

mat2img(folder, files, true)

function []=mat2img(folder, files, normalize)
%% save annotation to a single file
for m = 1:length(files)
    fprintf('%s\n', files(m).name);
    filename = strcat(folder,files(m).name);
    f = load(filename, '-mat');
    
    img = f.groundTruth{1}.Boundaries;
    for n = 2:length(f.groundTruth)
        img = img + f.groundTruth{n}.Boundaries;
    end
    
    if normalize
        img = img/max(max(img));
    end
    
    
    img = imcomplement(img);
    imwrite(img, strcat(folder,files(m).name(1:end-4),'_overlay.png'));
end
end

function []=mat2multiImg(folder,files)
%% save annotations to different files
for m = 1:length(files)
    fprintf('%s\n', files(m).name);
    filename = strcat(folder,files(m).name);
    f = load(filename, '-mat');
       
    for n = 1:length(f.groundTruth)
        img = imcomplement(f.groundTruth{n}.Boundaries);
        imwrite(img, strcat(folder,files(m).name(1:end-4),'_',num2str(n),'.png'));
    end
    
end
end