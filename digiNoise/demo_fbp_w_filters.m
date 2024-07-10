
load data/matfiles/I0.mat 
tic;
addpath('src/');
%% system setup 
down=1;
sg = sino_geom('fan', 'na',1024, 'nb', 986, 'ds', 0.95, ...
    'dsd', 1085.6, 'dso',595, ...
    'source_offset',0.0,'orbit',360, 'down', 1, 'strip_width', 'd');%0.909976 

ig = image_geom('nx', 512, 'ny', 512,'fov', 250,'offset_x', 0,'down', 1);
G  = Gtomo2_dscmex(sg, ig);

%% phantom setup
ell        = phantom_parameters('Shepp-Logan', ig);
xtrue      = ellipse_im(ig, ell, 'oversample', down);  % noise-free image unit: mm-1
sino       = G * xtrue;

%% other parameters 
MAX_FLUX = 2.25e5;
k_nd     = 0.65;
k_ld     = 0.25;
I02      = imresize(I0, size(sino)); 
I02      = I02*MAX_FLUX/max(I02(:));
sig      = sqrt(8); % standard deviation of electronic noise, a characteristic of CT scanner
% Ne     = electronic noise count is the variance of the electronic noise
mu_water = 0.1875/10.0; % mm-1
scalefac_k = 1024;
%% CT simulation
% perform low-dose simulation on the projection data
% lsino = pct_ldsino(I02, sino_n, k, sig);
hsino = pct_ldsino(I02, sino, k_nd, sig, 'T');
lsino = yuct_ldsino(I02, hsino, k_ld, sig*sig);
%%

% reconstruct image
% different filter types are in the file fbp2_window
tmp     = fbp2(sg, ig, 'type','std:mat');
fbpr_nd = fbp2(hsino, tmp, 'window', ''); 
fbpr    = fbp2(lsino, tmp, 'window', ''); % default is ramp filter
fbph0_6 = fbp2(lsino, tmp, 'window', 'hanning,0.6');
fbph85  = fbp2(lsino, tmp, 'window', 'hann85');
fbph150 = fbp2(lsino, tmp, 'window', 'hann150');
fbph175 = fbp2(lsino, tmp, 'window', 'hann175');
fbph205 = fbp2(lsino, tmp, 'window', 'hann205');
fbph250 = fbp2(lsino, tmp, 'window', 'hann250');
clear sgn G I0 tmp I02;

%im plc 2 2 
clim = [min(xtrue(:)) max(xtrue(:))];

xtrue   = (xtrue' - mu_water)*scalefac_k/mu_water + scalefac_k;
fbpr_nd = (fbpr_nd' - mu_water)*scalefac_k/mu_water + scalefac_k;
fbpr    = (fbpr' - mu_water)*scalefac_k/mu_water + scalefac_k;
fbph0_6 = (fbph0_6' - mu_water)*scalefac_k/mu_water + scalefac_k;
fbph85  = (fbph85' - mu_water)*scalefac_k/mu_water + scalefac_k;
fbph150 = (fbph150' - mu_water)*scalefac_k/mu_water + scalefac_k;
fbph175 = (fbph175' - mu_water)*scalefac_k/mu_water + scalefac_k;
fbph205 = (fbph205' - mu_water)*scalefac_k/mu_water + scalefac_k;
fbph250 = (fbph250' - mu_water)*scalefac_k/mu_water + scalefac_k;

fbpr_nd_crop = fbpr_nd;
fbpr_crop    = fbpr;
fbph0_6_crop = fbph0_6;
fbph85_crop  = fbph85;
fbph150_crop = fbph150;
fbph175_crop = fbph175;
fbph205_crop = fbph205;
fbph250_crop = fbph250;

norm_type = 'positive_scale';

nfbpr_nd_crop = fbpr_nd_crop;
nfbpr_crop    = fbpr_crop;
nfbph0_6_crop = fbph0_6_crop;
nfbph85_crop  = fbph85_crop;
nfbph150_crop = fbph150_crop;
nfbph175_crop = fbph175_crop;
nfbph205_crop = fbph205_crop;
nfbph250_crop = fbph250_crop;

if strcmp(norm_type, 'positive_scale')
	norm_type
	if (min(fbpr_nd_crop(:))<0)
		nfbpr_nd_crop = fbpr_nd_crop + (-min(fbpr_nd_crop(:)));
	end
	if (min(fbpr_crop(:))<0)
		nfbpr_crop = fbpr_crop + (-min(fbpr_crop(:)));
	end
	if (min(fbph0_6_crop(:))<0)
		nfbph0_6_crop = fbph0_6_crop + (-min(fbph0_6_crop(:)));
	end
	if (min(fbph85_crop(:))<0)
		nfbph85_crop = fbph85_crop + (-min(fbph85_crop(:)));
	end
	if (min(fbph150_crop(:))<0)
		nfbph150_crop = fbph150_crop + (-min(fbph150_crop(:)));
	end
	if (min(fbph175_crop(:))<0)
		nfbph175_crop = fbph175_crop + (-min(fbph175_crop(:)));
	end
	if (min(fbph205_crop(:))<0)
		nfbph205_crop = fbph205_crop + (-min(fbph205_crop(:)));
	end
	if (min(fbph250_crop(:))<0)
		nfbph250_crop = fbph250_crop + (-min(fbph250_crop(:)));
	end

end

if strcmp(norm_type, 'remove_negative')
	norm_type

    if (min(fbpr_nd_crop(:))<0)
		neg_ind_fbpr 			   = find(fbpr_nd_crop<0);
		nfbpr_nd_crop(neg_ind_fbpr)= 0.0;
	end
	if (min(fbpr_crop(:))<0)
		neg_ind_fbpr 			= find(fbpr_crop<0);
		nfbpr_crop(neg_ind_fbpr)= 0.0;
	end
	if (min(fbph0_6_crop(:))<0)
		neg_ind_fbph0_6 = find(fbph0_6_crop<0);
		nfbph0_6_crop(neg_ind_fbph0_6) =0.0;
	end
	if (min(fbph85_crop(:))<0)
		neg_ind_fbph85 = find(fbph85_crop<0);
		nfbph85_crop(neg_ind_fbph85) =0.0;
	end
	if (min(fbph150_crop(:))<0)
		neg_ind_fbph150 = find(fbph150_crop<0);
		nfbph150_crop(neg_ind_fbph150) =0.0;
	end
	if (min(fbph175_crop(:))<0)
		neg_ind_fbph175 = find(fbph175_crop<0);
		nfbph175_crop(neg_ind_fbph175)=0.0;
	end
	if (min(fbph205_crop(:))<0)
		neg_ind_fbph205 = find(fbph205_crop<0);
		nfbph205_crop(neg_ind_fbph205)=0.0;
	end
	if (min(fbph250_crop(:))<0)
		neg_ind_fbph205 = find(fbph205_crop<0);
		nfbph250_crop(neg_ind_fbph205)=0.0;
	end

end

rmser_nd  =sqrt(sum((xtrue(:)-nfbpr_nd_crop(:)).^2)/(512*512));
rmser_ld  =sqrt(sum((xtrue(:)-nfbpr_crop(:)).^2)/(512*512));
rmseh0_6  =sqrt(sum((xtrue(:)-nfbph0_6_crop(:)).^2)/(512*512));
rmseh85   =sqrt(sum((xtrue(:)-nfbph85_crop(:)).^2)/(512*512));
rmseh150  =sqrt(sum((xtrue(:)-nfbph150_crop(:)).^2)/(512*512));
rmseh175  =sqrt(sum((xtrue(:)-nfbph175_crop(:)).^2)/(512*512));
rmseh205  =sqrt(sum((xtrue(:)-nfbph205_crop(:)).^2)/(512*512));
rmseh250  =sqrt(sum((xtrue(:)-nfbph250_crop(:)).^2)/(512*512));

rmse_vec = [rmser_nd,rmser_ld, rmseh0_6, rmseh85, rmseh150, rmseh175, rmseh205, rmseh250];


g=figure('visible', 'on');
im plc 3 3
clim = [0 max(xtrue(:))];
im(1, xtrue', 'GT', clim), cbar;
im(2, nfbpr_nd_crop', sprintf(['ND rFBP=', num2str(rmser_nd)]), [min(nfbpr_nd_crop(:)) max(nfbpr_nd_crop(:))]), cbar
im(3, nfbpr_crop', sprintf(['LD rFBP=', num2str(rmser_ld)]), [min(nfbpr_crop(:)) max(nfbpr_crop(:))]), cbar
im(4, nfbph0_6_crop', sprintf(['LD FBPh0.6=', num2str(rmseh0_6)]), [min(nfbph0_6_crop(:)) max(nfbph0_6_crop(:))]), cbar
im(5, nfbph85_crop', sprintf(['LD FBPh85=', num2str(rmseh85)]), [min(nfbph85_crop(:)) max(fbph85_crop(:))]), cbar
im(6, nfbph150_crop', sprintf(['LD FBPh150=', num2str(rmseh150)]), [min(nfbph150_crop(:)) max(nfbph150_crop(:))]), cbar
im(7, nfbph175_crop', sprintf(['LD FBPh175=', num2str(rmseh175)]), [min(nfbph175_crop(:)) max(nfbph175_crop(:))]), cbar
im(8, nfbph205_crop', sprintf(['LD FBPh205=', num2str(rmseh205)]), [min(nfbph205_crop(:)) max(nfbph205_crop(:))]), cbar
im(9, nfbph250_crop', sprintf(['LD FBPh250=', num2str(rmseh250)]), [min(nfbph250_crop(:)) max(nfbph250_crop(:))]), cbar
colormap gray;
hold off;
saveas(g, 'results/fbp_nd_ldfilters_imgplots.png')


