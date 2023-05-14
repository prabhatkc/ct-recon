
clc; clear all;
addpath("src/");

% -------------------------------------------------
% declaring dose levels
% 1 indicates normal dose (ND); 
% 0.75, 0.5 and 0.25 indicate 75%, 50%, 25% of ND
% --------------------------------------------------
dose_levels    = [1.0, 0.75, 0.5, 0.25];
output_fld_tag = {'nd', 'tfd', 'hfd', 'qd'};
obj_models     = {'bkg', 'disk'};


% -----------------------------
% Set acquisition parameters
% -----------------------------
sys_info.nb             = 986;
sys_info.na             = 2880;
sys_info.ds             = 0.95;
sys_info.max_flux       = 2.25e5;

obj_info.fov            = 250;
misc_info.k_nd          = 0.85;
misc_info.filter_string ='hann200';
misc_info.norm_type     = '';

for ii=length(obj_models)
    obj=obj_models{ii};
    % ------------------------------------------------------------------
    % create or load object model
    % ------------------------------------------------------------------
    if strcmp(obj, 'disk')
        % simulated noisy realizations of CCT-189 phantom
        xtrue = read_raw_img('./data/true/mita_512.raw', [512 512], 'int16');
    else 
        % simulated noisy realizations of uniform phantom
        xtrue = makecircle(512, 200, 0.0, -1024);
    end
    
    % ------------------------------------------------------------------
    % CT acquisitions w.r.t different dose levels
    % ------------------------------------------------------------------
    xtrue = repmat(xtrue, [1, 1, 100]);
    for i=1:length(dose_levels)
        if (i~=1.0)
            misc_info.k_ld=dose_levels(i); 
        end 
        misc_info.output_folder= ['./results/mita/', obj,'/', output_fld_tag{i}];
        insert_noise_2_obj_model(xtrue, obj_info, sys_info, misc_info);   
    end

end

