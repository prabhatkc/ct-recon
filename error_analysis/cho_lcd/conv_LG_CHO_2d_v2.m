
%[auc,snr, chimg,tplimg,meanSP,meanSA,meanSig, k_ch, t_sp, t_sa,]=conv_LG_CHO_2d(trimg_sa, trimg_sp, testimg_sa, testimg_sp, ch_width, nch, b_conv, ch2)
%Filtered/convolutional Channels CHO, based on the paper:
%Diaz et al, IEEE-tmi-34(7), 2015, "Derivation of an observer model adapted
%to irregular signals based on covolution channels"
%Inputs
%   testimg_sa: the test set of signal-absent, a stack of 2D array;
%   testimg_sp: the test set of signal-present;
%   trimg_sa: the training set of signal-absent;
%   trimg_sp: the training set of signal-present;
%   ch_width: channel width parameter;
%   nch: number of channels to be used;
%   b_conv: 1 or 0 to indicate whether to apply a convolution of the signal
%   to the LG channels. Default is 1.
%   ch2: an optional additional LG channel, 2-element vector form [ch_width
%   nch] eg. for the spiculated mass, one may use a main channel of width
%   matching the signal size and use an additional channel with small width
%   for detecting the edge feature.
%Outputs
%   auc: the AUC values
%   snr: the detectibility SNR
%   t_sp: t-scores of SP cases
%   t_sa: t-scores of SA cases
%
%R Zeng, 6/2016, FDA/CDRH/OSEL/DIDSR

function [auc, snr,chimg,tplimg,meanSP,meanSA,meanSig, k_ch, t_sp, t_sa]=conv_LG_CHO_2d_v2(trimg_sa, trimg_sp,testimg_sa, testimg_sp,  ch_width, nch, b_conv, ch2)

if(nargin<7)
    b_conv=1;
end

[nx, ny, nte_sa]=size(testimg_sa);

%Ensure the images all having the same x,y sizes. 
[nx1, ny1, nte_sp]=size(testimg_sp);
if(nx1~=nx | ny1~=ny)
    error('Image size does not match! Exit.');
end
[nx1, ny1, ntr_sa]=size(trimg_sa);
if(nx1~=nx | ny1~=ny)
    error('Image size does not match! Exit.');
end
[nx1, ny1, ntr_sp]=size(trimg_sp);
if(nx1~=nx | ny1~=ny)
    error('Image size does not match! Exit.');
end

%LG channels
xi=[0:nx-1]-(nx-1)/2;
yi=[0:ny-1]-(ny-1)/2;
[xxi,yyi]=meshgrid(xi,yi);
r=sqrt(xxi.^2+yyi.^2);
u=laguerre_gaussian_2d(r,nch-1,ch_width);
ch=reshape(u,nx*ny,size(u,3)); %if not applying the following filtering to the channels

%if ch2
nch1 = nch;
if(exist('ch2')) %#ok<EXIST>
    
    u2=laguerre_gaussian_2d(r,ch2(2)-1,ch2(1));
    ch2=reshape(u2,nx*ny,size(u2,3));
    u(:,:,nch+1: nch+size(ch2,2))=u2; %Append the second channel to the main channels
    ch=reshape(u,nx*ny,size(u,3));
    nch=size(ch,2);
end

%Create signal convolved channels
sig_mean=mean(trimg_sp,3)-mean(trimg_sa,3);
if(b_conv)  
    for ich=1:nch
        ch_sig(:,:,ich) = (ifft2(abs(fft2(u(:,:,ich))).^2 .* fft2(sig_mean)))/nx/ny;
        ch_sig(:,:,ich) = ch_sig(:,:,ich)/sqrt(sum(sum(ch_sig(:,:,ich).^2))); %Normalize the energy of the channel function. but this does not affect the detectability at all.
    end
else
    ch_sig(:,:,1:nch1) = u(:,:,1:nch1); %the first large-width channel was kept the same.
    
    for ich=nch1+1:nch %apply the convolution to the smaller-width channelfunction.
        ch_sig(:,:,ich) = (ifft2(abs(fft2(u(:,:,ich))).^2 .* fft2(sig_mean)))/nx/ny;
        ch_sig(:,:,ich) = ch_sig(:,:,ich)/sqrt(sum(sum(ch_sig(:,:,ich).^2))); %Normalize the energy of the channel function. but this does not affect the detectability at all.
    end
end
ch=reshape(ch_sig, nx*ny, nch);


%Training MO
nxny=nx*ny;
tr_sa_ch = zeros(nch, ntr_sa);
tr_sp_ch = zeros(nch, ntr_sp);
for i=1:ntr_sa
    tr_sa_ch(:,i) = reshape(trimg_sa(:,:,i), 1,nxny)*ch;
end
for i=1:ntr_sp
    tr_sp_ch(:,i) = reshape(trimg_sp(:,:,i), 1,nxny)*ch;
end
s_ch = mean(tr_sp_ch,2) - mean(tr_sa_ch,2);
k_sa = cov(tr_sa_ch');
k_sp = cov(tr_sp_ch');
k = (k_sa+k_sp)/2;
w = s_ch(:)'*pinv(k); %this is the hotelling template

%detection (testing)
for i=1:nte_sa
    te_sa_ch(:,i) = reshape(testimg_sa(:,:,i), 1, nxny)*ch;
end
for i=1:nte_sp
    te_sp_ch(:,i) = reshape(testimg_sp(:,:,i), 1, nxny)*ch;
end
t_sa=w(:)'*te_sa_ch;
t_sp=w(:)'*te_sp_ch;

snr = (mean(t_sp)-mean(t_sa))/sqrt((std(t_sp)^2+std(t_sa)^2)/2);

nte = nte_sa + nte_sp;
data=zeros(nte,2);
data(1:nte_sp,1) = t_sp(:);
data(nte_sp+[1:nte_sa],1) = t_sa(:);
data(1:nte_sp,2)=1;
out = roc(data);
auc = out.AUC;

%Optional outputs
tplimg=(reshape(w*ch',nx,ny)); % MO template
chimg=reshape(ch,nx,ny,nch); %Channels
meanSP=mean(trimg_sp,3);
meanSA=mean(trimg_sa,3);
meanSig=sig_mean;
k_ch=k;