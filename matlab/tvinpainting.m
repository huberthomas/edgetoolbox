img = im2double(imread('/run/user/1000/gvfs/smb-share:server=192.168.0.253,share=data/Master/train/rgbd_dataset_freiburg2_xyz/depth/1311867170.450076.png'));
miss_data = [1,5,6,7,20,25,29,31,32,49,51,70,71,76,90]; %find(img == 0);%
img(miss_data,:) = 0;
f = img(:);
nMaxIterations = 1500;
[n_row,n_col] = size(img);
miss_img = ones(n_row,n_col);
miss_img(miss_data,:) = 0;
miss_img = miss_img(:);
N = n_row*n_col;
%generate nabla
nabla = make_nabla(n_row,n_col);
nabla_t = nabla';
lambda = 10;
u = f;
u_ = f;
p = zeros(2*N,1);
sigma = 1;
tau = 1/sigma;
sigma_p = sigma/2;
tau_u = tau/(4+lambda);

L      = sqrt(8);
tau_u    = 0.01;
sigma_p  = (1/L^2)/tau;
theta = 0.5;
for nProcessing = 1:nMaxIterations
    %update dual
    temp_p = p + sigma_p * nabla * u_;
    sqrt_p = sqrt(temp_p(1:N).^2 + temp_p(N+1:2*N));
    sqrt_p_ = [sqrt_p;sqrt_p];
    p = temp_p./max(1,abs(sqrt_p_));
    u_ = u;
    u1 = u - tau_u * nabla_t * p;
    idx1 = (miss_img == 1);
    u(idx1) = (u1(idx1) + tau_u*lambda*f(idx1)) / (1+tau_u*lambda);
    %missing lines
    idx2 = (miss_img == 0);
    u(idx2) = u1(idx2);
    %u = (u + tau_u*(nabla_t * p + lambda*f))/(1+tau*lambda);
    u_ =  u+theta*(u-u_);
    imshow(reshape(u,n_row,n_col),[]);
    drawnow
    Du = nabla*u;
    energy(nProcessing) = sum(sqrt(Du(1:N).^2+Du(N+1:2*N).^2)+lambda/2 * (u-f).^2);
    energy(nProcessing)
end

