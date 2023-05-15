function []= mita_lcd_4r_mayo(data_folder, proc_data_folder, chkpt_string, idx_insert, output_fname) 
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

all_recon_type= {'fbp_sharp'};% 'blf_fbp_sharp'}; 
dose_strings  = {'qd', 'hfd', 'tfd', 'nd'};

%I0_vector = 3e5*[30 55 70 85 100]/100;
%I0 = I0_vector(1);%6e5*0.6;;
n_spfile = 200;
n_safile = 100;
n_reader = 10;
n_train = 100;
n_I0 = length(dose_strings);

%%inserts info
nx = 512;%256;
dx = 0.48828; %PixelSpacing
fov = dx*nx;  
d = 40;     % mm
% insert_info [x_center, y_center, r, HU]
insert_info = [...
    d*cosd(45)  d*sind(45)    3/2  14;      % 3mm, 14 HU
    -d*cosd(45)  d*sind(45)   5/2   7;      % 5 mm, 7 HU
    -d*cosd(45) -d*sind(45)   7/2   5;      % 7 mm, 5 HU
    d*cosd(45) -d*sind(45)   10/2   3;      % 10 mm, 3 HU
    ];
num_inserts = size(insert_info, 1);

% convert roi locations from mm to pixels
insert_centers = round(insert_info(:,1:2) * (nx/fov) + (nx+1)/2);
insert_radii = insert_info(:,3) * (nx/fov);
% select insert
center_x = insert_centers(idx_insert, 1); % 203
center_y = nx-insert_centers(idx_insert, 2); %117
insert_r = insert_radii(idx_insert); %7.53 %due to matlab coordinate system, 
crop_r = ceil(3*max(insert_radii)); %23
% get roi
sp_crop_xfov = center_x + [-crop_r:crop_r]; %180:226
sp_crop_yfov = center_y + [-crop_r:crop_r]; % 94:140
roi_nx = 2*crop_r + 1; %47

nroi = 5;
sa_crop_xfov = zeros(nroi, roi_nx);
sa_crop_yfov = sa_crop_xfov;
sa_center_x = [center_x center_x center_x center_x-crop_r center_x+crop_r];
sa_center_y = [center_y-crop_r center_y center_y+crop_r center_y center_y];
for i=1:nroi
    sa_crop_xfov(i,:) = sa_center_x(i) + [-crop_r:crop_r];
    sa_crop_yfov(i,:) = sa_center_y(i) + [-crop_r:crop_r];
end
%check roi
fid = fopen('../irt/digiNoise/data/true/mita_512.raw');
xtrue = fread(fid,[nx nx],'int16');
fclose(fid);

%plot ROI of the idx used
figure,imagesc(xtrue(sp_crop_xfov, sp_crop_yfov)); colorbar;% sp roi in the loaded images
title(['r=' num2str(insert_r*dx) ' (mm)']);
print('-depsc', [output_fname '/_idx_' num2str(idx_insert) '.pdf']);
%close;

actual_insert_HU = xtrue(center_x, center_y); % 3mm of 14 H
if(actual_insert_HU ~= insert_info(idx_insert, 4))
    disp('Warning: geometric mismatch! Quit.')
    return;
end

n_recon_option = length(all_recon_type);
n_sp           = n_spfile; % 200
n_sa           = n_safile*nroi; %500
auc_all        = zeros(n_reader, n_recon_option, n_I0); %[10, 6, 5]
snr_all        = zeros(n_reader, n_recon_option, n_I0); 
ndose          = length(dose_strings);

for iI = 1:ndose %-> incident flux option
  iI
  each_dstring = dose_strings{iI};
  
  for k=1:n_recon_option
    %% here we have 2 recon options
    % first is fbp and second is cnn
    % so for each dose level we have [n_reader, 2] snr, auc vals
    recon_option = all_recon_type{k}

    folder_sp = [data_folder '/disk/' each_dstring '/' recon_option]
    folder_sa = [data_folder '/bkg/'  each_dstring '/' recon_option ]
    
    if (strfind(recon_option, 'blf'))
        folder_sp = [proc_data_folder '/disk/' each_dstring '_' chkpt_string '/' recon_option]
        folder_sa = [proc_data_folder '/bkg/'  each_dstring '_' chkpt_string '/' recon_option ]
    end

    sp_img = zeros(nx, nx, n_spfile); %[320, 320, 200]
    sp_roi = zeros(roi_nx,roi_nx, n_sp); %[47 47 200]
    for i=1:n_spfile
        %filenum = i;
        filenum_string = num2str(i);
        filename = [folder_sp  '_' filenum_string '.raw']; 
        fid = fopen(filename);
        im_current = fread(fid, [nx, nx], 'int16');
        img = im_current;
        fclose(fid);
        sp_img(:,:,i) = img;
        img_crop = img(sp_crop_xfov, sp_crop_yfov);
        sp_roi(:,:,i) = img_crop - mean(img_crop(:));
    end

    sa_img = zeros(nx,nx,n_safile);
    sa_roi = zeros(roi_nx,roi_nx, n_sa);
    for i=1:n_safile
        %filenum = i;
        filenum_string = num2str(i);
        filename = [folder_sa  '_' filenum_string '.raw']; 

        fid = fopen(filename);
        im_current = fread(fid, [nx, nx], 'int16');
        img = im_current;
        fclose(fid);
        sa_img(:,:,i) = img;    
        for j=1:5
            img_crop = img(sa_crop_xfov(j,:), sa_crop_yfov(j,:));
            sa_roi(:,:,(i-1)*5+j) = img_crop - mean(img_crop(:));
        end
    end
    %
    auc=zeros(1, n_reader);
    snr=zeros(1, n_reader);
    for i=1:n_reader
        % shuffle training data

        %idx_sa1 = randperm(n_preset_sa);
        %idx_sp1 = randperm(n_preset_sp);
        idx_sa1 = randperm(n_sa);
        idx_sp1 = randperm(n_sp);

        idx_sa_tr = idx_sa1(1:n_train);
        idx_sp_tr = idx_sp1(1:n_train);
        idx_sa_test = idx_sa1(n_train+1:end);
        idx_sp_test = idx_sp1(n_train+1:end);

        % run LG CHO
        %auc, snr,chimg,tplimg,meanSP,meanSA,meanSig, k_ch, t_sp, t_sa]
        [auc(i), snr(i), chimg, tplimg, meanSP, meanSA, meanSig, kch, t_sp, t_sa] = ...
            conv_LG_CHO_2d_v2(sa_roi(:, :, idx_sa_tr), sp_roi(:, :, idx_sp_tr), ...
            sa_roi(:, :, idx_sa_test), sp_roi(:, :, idx_sp_test), insert_r*dx, 5, 0);%
    end
    auc_all(:,k,iI) = auc;
    snr_all(:,k,iI) = snr;

  end
  
end

matfname = [output_fname '/_idx_' num2str(idx_insert) '.mat'];
if isOctave==0
    save(matfname, 'auc_all', 'snr_all');
else
    save('-v7', matfname, 'auc_all', 'snr_all');
end


