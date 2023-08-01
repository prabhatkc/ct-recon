clc; clear all;
addpath("src/")

acquire_nd = 'T';
acquire_ld = 'T';

create_noisy_data = 'T'; % turn this option to 'F' for noiseless simulation
% -----------------------------
% create or load object model
% -----------------------------
xtrue = read_raw_img('data/true/acr_m1_512.raw', [512 512], 'int16');
figure, imshow(xtrue, []); colorbar; title('ACR module 1 object model');
xtrue = repmat(xtrue, [1, 1, 1]);

if strcmp(create_noisy_data, 'T')
    if strcmp(acquire_nd, 'T')
        sys_info.nb             = 986;
        sys_info.na             = 2880;
        sys_info.ds             = 0.95;
        sys_info.max_flux       = 2.25e5;
        
        obj_info.fov            = 250;
        misc_info.k_nd          = 0.85;
        misc_info.filter_string ='hann200';
        misc_info.norm_type     ='';
        misc_info.scalefac_k    = 1000.0;
        misc_info.plot_figures  ='T';
        misc_info.output_folder =[];%'./results/acr_m1/digi_nd/fov_250_na_2880_ds_0.95_hann200_k_0.85_I_2.25e5_ps_scalek1000/';
        
        insert_noise_2_obj_model(xtrue, obj_info, sys_info, misc_info);
    end 
    
    if strcmp(acquire_ld, 'T')
        sys_info.nb             = 986;
        sys_info.na             = 2880;
        sys_info.ds             = 0.95;
        sys_info.max_flux       = 2.25e5;
        
        obj_info.fov            = 250;
        misc_info.k_nd          = 0.85;
        misc_info.k_ld          = 0.25;
        misc_info.filter_string ='hann200';
        misc_info.norm_type     = '';
        misc_info.scalefac_k    = 1000.0;
        misc_info.plot_figures  ='T';
        misc_info.output_folder =[];%'./results/acr_m1/digi_qd/fov_250_na_2880_ds_0.95_hann200_k_0.85_I_2.25e5_scalek1000/';
        insert_noise_2_obj_model(xtrue, obj_info, sys_info, misc_info);
    end 
else
    % noiseless reconstruction 
    sys_info.nb             = 986;
    sys_info.na             = 2880;
    sys_info.ds             = 0.95;
    sys_info.max_flux       = 2.25e5;
        
    obj_info.fov            = 250;
    misc_info.k_nd          = 0.85;
    misc_info.filter_string ='hann200';
    misc_info.norm_type     ='';
    misc_info.scalefac_k    = 1000.0;
    misc_info.plot_figures  ='T';
    misc_info.noiseless     ='True';
    misc_info.output_folder =[];%'./results/acr_m1/digi_nd/fov_250_na_2880_ds_0.95_hann200_k_0.85_I_2.25e5_ps_scalek1000/';
    
    insert_noise_2_obj_model(xtrue, obj_info, sys_info, misc_info);
end 