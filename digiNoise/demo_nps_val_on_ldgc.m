%% ------------------------------------------------------------------------
% (demo_nps_val_on_ldgc.m)
%% ------------------------------------------------------------------------
% This file creates digital water phantom. Adds 
% noise relative to the normal dose (ND) and 
% quater dose (QD) acquistions.
% Then it shows that the added ND and QD noise align 
% with those exibited by ND and QD CT acquisition
% provided in the Mayo Grand Challenge dataset
%
% ------------------------------------------------------------------------
% References
% ------------------------------------------------------------------------
% [1] https://github.com/JeffFessler/mirt
% [2] Zeng, D., Huang, J., Bian, Z., Niu, S., Zhang, H., Feng, Q., Liang, 
%     Z. and Ma, J., 2015. A simple low-dose x-ray CT simulation from 
%     high-dose scan. IEEE transactions on nuclear science, 62(5), 
%     pp.2226-2233
% [3] Yu, L., Shiung, M., Jondal, D. and McCollough, C.H., 2012. 
%     Development and validation of a practical lower-dose-simulation tool 
%     for optimizing computed tomography scan protocols. Journal of 
%     computer assisted tomography, 36(4), pp.477-487. 


clc; clear all;
addpath("src/");
addpath("data/matfiles/")

% change this bool option to "F" if 
% ND and QD have already
% been previously simulated
create_noisy_data = 'F';

% bool options for ND and QD acquisitions
acquire_nd = 'F';
acquire_ld = 'F';

% bool option to show 1D nps plot
show_nps_validation = 'T';

% -------------------------------------------------------------------------
% create or load object model
% -------------------------------------------------------------------------
xtrue = makecircle(512, 200, 0.0, -1000);
xtrue = repmat(xtrue, [1, 1, 17]);
% plotLayers(xtrue)

% -------------------------------------------------------------------------
% insert noise to digital object
% -------------------------------------------------------------------------
if strcmp(create_noisy_data, 'T')
    if strcmp(acquire_nd, 'T')
        sys_info.nb             = 986;
        sys_info.na             = 2880;
        sys_info.ds             = 0.95;
        sys_info.max_flux       = 2.25e5;
        
        obj_info.fov            = 250;
        misc_info.k_nd          = 0.85;
        misc_info.filter_string ='hann200';
        misc_info.norm_type     = 'positive_scale';
        misc_info.output_folder ='./results/digital_water/nd/';
        insert_noise_2_obj_model(xtrue, obj_info, sys_info, misc_info);
    end 
    
    if strcmp(acquire_ld, 'T')
        sys_info.nb             = 986;
        sys_info.na             = 2880;
        sys_info.ds             = 0.95;
        sys_info.max_flux       = 2.25e5;
        
        obj_info.fov            = 250;
        misc_info.k_nd          = 0.85;
        dose_levels             = [0.75, 0.5, 0.25];% 75%, 50%, 25% Dose
        output_fld_tag          = {'tfd', 'hfd', 'qd'};
        misc_info.filter_string ='hann200';
        misc_info.norm_type     ='';

        for i=1:length(dose_levels)
            misc_info.k_ld=dose_levels(i); 
            misc_info.output_folder= ['./results/digital_water/', output_fld_tag{i}];
            insert_noise_2_obj_model(xtrue, obj_info, sys_info, misc_info);
        end
    end 
end

if strcmp(show_nps_validation, 'T')
    % ---------------------------------------------------------------------
    % Calculate NPS from a real acquisition system 
    % ---------------------------------------------------------------------
    % Loading 1D NPS calculated using uniform phantom acquisition for the
    % comparision. This data is publicly available through the following 
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
    
    % -----------------------------------------------------
    % Validate NPS from our noise insertion model
    % -----------------------------------------------------
    sp_str        = 'ldgc_vs_ourSim_1d_nps'; % best para
    title_str     = strreps(sp_str, '_', '-');
    nps_pic_fld   = './results/digital_water/nps_pics/';
    nps_pic_fn    = [nps_pic_fld, sp_str, '.png'];
    if ~exist(nps_pic_fld, 'dir')
        mkdir(nps_pic_fld)
    end
    
    % path of CT-images from the normal dose and low dose 
    % acquisitions of the uniform object
    misc.dx     = 0.488; 
    misc.size   = [512, 512];
    nd_input_path  = ['./results/digital_water/nd/'];% normal dose
    ld_input_path  = ['./results/digital_water/qd/'];%.25
    hf_input_path  = ['./results/digital_water/hfd/']; % 0.5 D
    tf_input_path  = ['./results/digital_water/tfd/'];
    
    % -----------------------------------------------------
    % NPS of LD and ND digital water phantom
    % -----------------------------------------------------
    [fr, ld_w_nps1d, ld_w_nps2d] = nps_of_catphanSim_4r_realization_path(ld_input_path, 'raw', misc);
    [fr, hf_w_nps1d, hf_w_nps2d] = nps_of_catphanSim_4r_realization_path(hf_input_path, 'raw', misc);
    [fr, tf_w_nps1d, tf_w_nps2d] = nps_of_catphanSim_4r_realization_path(tf_input_path, 'raw', misc);
    misc.remove_inflection = 'T';
    [fr, hd_w_nps1d, hd_w_nps2d] = nps_of_catphanSim_4r_realization_path(nd_input_path, 'raw', misc);
    
    disp('Loading pre-calculated 1D NPS estimated using  3mm D45 (sharp)');
    disp('CT scans of ACR uniform phantom in the LDGC dataset.');
    disp('For more info look inside the demo_nps_val_on_ldgc.m file.');

    load ldgc_nps.mat;
    figure;
    plot(fr, nd_nps1d, 'k');
    hold on;
    plot(fr, ld_nps1d, 'b');
    plot(fr, hd_w_nps1d, 'o-k');
    plot(fr, ld_w_nps1d, 'o-b');
    plot(fr, tf_w_nps1d, '.-g');
    plot(fr, hf_w_nps1d, 'o-g');
    ylim([0.0 2200]);
    xlim([0.0 1.01]);
    hold off;
    legend('NPS-1d-LDGC-ND', 'NPS-1d-LDGC-QD', 'nps-1d-ourSim-ND', ...
        'nps-1d-ourSim-QD', 'nps-1d-ourSim-tfD', 'nps-1d-ourSim-hfD');
    grid on;
    grid minor;
    set(gca, 'Fontsize', 12);
    h=xlabel('Spatial Frequency (mm^{-1})');
    set(h, 'Fontsize', 12);
    h=ylabel('Noise Power (HU^{2}mm^{2})');
    print('-dpng', nps_pic_fn);
end
