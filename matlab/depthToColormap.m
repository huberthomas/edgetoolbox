function [out] = depthToColormap(D)
D_jet = double(D)/double(max(D(:)));
D_jet = D_jet * 255.0;
J = jet(256);
% J = flipud(J); %invert colormap
% figure;
% imshow(reshape(J(uint8(D_jet(:)+1),:),[480,640,3]))
% imwrite(reshape(J(uint8(D_jet(:)+1),:),[480,640,3]),'depth_jet.png');
out = reshape(J(uint8(D_jet(:)+1),:),[size(D,1),size(D,2),3]);
end

