%% ------------------------------------------------------------------------
% demo_modified_xcat_realizations
%% ------------------------------------------------------------------------
% This file creates load an xcat, removes 
% its bony regions, adds 
% noise relative to the normal dose (ND) and 
% quater dose (QD) acquistions.
% Then it shows that the added ND and QD noise align 
% with those exibited by ND and QD CT acquisition
% provided in the Mayo Grand Challenge dataset.
%
clc; clear all;
addpath("src/")

create_noisy_data   = 'T';
acquire_nd          = 'T';
acquire_ld          = 'T';

show_nps_validation  = 'T';
load_acr_nps_mat_file= 'T';
% -----------------------------
% load xcat file and remove its
% bony regions for calculating
% nps using only uniform regions
% -----------------------------
xtrue      = read_raw_img('./data/true/xcat_512_hu.raw', [512 512], 'uint16');
ind        = find(xtrue>274);
xtrue(ind) = 1024;
xtrue      = repmat(xtrue, [1, 1, 30]);
% plotLayers(xtrue)

% -----------------------------
% insert noise to digital object
    % Common acquisition parameters
    % as compared to acr acquisition following
    % parameters are changed for xcat acquisition
    % na, k_nd. Also norm type has to be set to
    % positive scale
% -----------------------------
sys_info.nb             = 986;
sys_info.na             = 1024;
sys_info.ds             = 0.95;
sys_info.max_flux       = 2.25e5;

obj_info.fov            = 250;
misc_info.k_nd          = 0.65;
misc_info.intercept_k   = 1024; 
misc_info.filter_string ='hann200';
misc_info.norm_type     = 'positive_scale';
misc_info.out_dtype     = 'uint16';

if strcmp(create_noisy_data, 'T')

    if strcmp(acquire_nd, 'T')
        misc_info.output_folder ='./results/mod_xcat/k_0.65/nd';
        insert_noise_2_obj_model(xtrue, obj_info, sys_info, misc_info);
    end 
    
    if strcmp(acquire_ld, 'T')
        misc_info.k_ld          = 0.25; %qd
        misc_info.output_folder ='./results/mod_xcat/k_0.65/qd/';
        insert_noise_2_obj_model(xtrue, obj_info, sys_info, misc_info);
    end 
end

if strcmp(show_nps_validation, 'T')
    % -----------------------------------------------------
    % Calculate NPS from a real acquisition system 
    % -----------------------------------------------------
    % NPS calculation of NPS from Mayo's physical water phantom acquisition for the
    % baseline comparision. This data is publicly available through the following 
    % TCIA website: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026
    % We make use of 17 slices from the repository's 3mm D45 (sharp)
    % normal dose (ND) & quater dose (QD) acquisitions. Following is the starting and ending
    % slices number             
    % QD:
    % R_ACR_GRANDCHALLENGE.CT.ABDOMEN_ABDOMENROUTINE_(ADULT).0011.0070.2016.02.18.10.56.16.495590.155341904.IMA
    % R_ACR_GRANDCHALLENGE.CT.ABDOMEN_ABDOMENROUTINE_(ADULT).0011.0085.2016.02.18.10.56.16.495590.155342309.IMA
    % ND:
    % R_ACR_GRANDCHALLENGE.CT.ABDOMEN_ABDOMENROUTINE_(ADULT).0008.0070.2016.02.18.10.56.16.495590.155315989.IMA
    % R_ACR_GRANDCHALLENGE.CT.ABDOMEN_ABDOMENROUTINE_(ADULT).0008.0085.2016.02.18.10.56.16.495590.155316394.IMA
    
    if strcmp(load_acr_nps_mat_file, 'T')
        load data/matfiles/acr_nd_ld_nps_3mm_sharp.mat;
    else
      mayo_nd_path = '/gpfs_projects/prabhat.kc/lowdosect/data/mayo_acr_data/uniform/full_3mm_sharp/';
      mayo_ld_path = '/gpfs_projects/prabhat.kc/lowdosect/data/mayo_acr_data/uniform/quarter_3mm_sharp/';
      [fr, nd_nps1d, nd_nps2d] = nps_of_catphanReal_4r_realization_path(mayo_nd_path, 'dicom', 'acr', []);
      [fr, ld_nps1d, ld_nps2d] = nps_of_catphanReal_4r_realization_path(mayo_ld_path, 'dicom', 'acr', []);
    end

    % -----------------------------------------------------
    % Validate NPS from our noise insertion model
    % -----------------------------------------------------
    sp_str        = ['fov_', num2str(obj_info.fov),'_na_', num2str(sys_info.na), '_ds_', ...
                    num2str(sys_info.ds), '_', misc_info.filter_string,'_Io_', num2str(sys_info.max_flux*misc_info.k_nd)]; % best para for aapm
    title_str     = strrep(sp_str, '_', '-');
    nps_pic_fld   = './results/mod_xcat/nps_pics/';
    nps_pic_fn    = [nps_pic_fld, 'ld-nd-k-0.65-xcat-nps.eps'];
    if ~exist(nps_pic_fld, 'dir')
        mkdir(nps_pic_fld)
    end
    
    % path of CT-images from the normal dose and low dose acquisitions of
    % the digital object
    misc.dx        = 0.488; % from mayo water dicom images pixelspacing;  
    misc.size      = [512, 512];
    nd_input_path  = ['results/mod_xcat/k_0.65/nd/'];%full
    ld_input_path  = ['results/mod_xcat/k_0.65/qd/'];%.25
    
    % -----------------------------------------------------
    % NPS of LD and ND digital water phantom
    % -----------------------------------------------------
    misc.obj                ='xcat';
    misc.remove_inflection  = 'T';
  % misc.zero_inflection    = 'T';
    misc.dtype              = 'uint16';
    [fr, nd_xcat_nps1d, nd_xcat_nps2d] = nps_of_catphanSim_4r_realization_path(nd_input_path, 'raw', misc);
    
    misc.obj                  ='xcat';
    misc.remove_inflection    = 'T';
  % misc_info.zero_inflection = 'T';
    misc.dtype                = 'uint16';
    [fr, ld_xcat_nps1d, ld_xcat_nps2d] = nps_of_catphanSim_4r_realization_path(ld_input_path, 'raw', misc);

    
    figure;
    plot(fr, nd_nps1d, 'k');
    hold on;
    plot(fr, ld_nps1d, 'b');
    plot(fr, nd_xcat_nps1d, 'o-k');
    plot(fr, ld_xcat_nps1d, 'o-b');
    %plot(fr, tf_w_nps1d, '.-g');
    %plot(fr, hf_w_nps1d, 'o-g');
    %ylim([0.0 2200]);
    xlim([0.0 1.01]);
    hold off;
    % legend('NPS-1d-LDGC-ND', 'NPS-1d-LDGC-QD', 'NPS-1d-ourSim-ND', 'NPS-1d-ourSim-QD', 'nps1d-ourSim-tfD', 'nps1d-ourSim-hfD');
    legend('NPS-1d-LDGC-ND', 'NPS-1d-LDGC-QD', 'NPS-1d-ourXcat-ND', 'NPS-1d-ourXcat-QD');
    %legend('xcat-nps1d-nd', 'xcat-nps1d--ld');
    grid on;
    set(gca, 'Fontsize', 14);
    h=xlabel('Spatial Frequency (mm^{-1})');
    set(h, 'Fontsize', 14);
    h=ylabel('Noise Power (HU^{2}mm^{2})');
    title(title_str);
    print('-depsc', nps_pic_fn);
end
