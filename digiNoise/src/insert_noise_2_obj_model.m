function [] = insert_noise_2_obj_model(obj, obj_info, sys_info, misc_info)

%% this function adds noise to a True object relative to  CT acquisition 
%  parameters specified through system models.This function makese use of 
%  IRT package and the system geometry used is fan beam. For IRT
%  installation look into ref [1] below
%
% ------
% INPUT:
% ------
% 
% obj_model
%   Nx, Ny  : dimension of the object
%   fov     : field of view
%   down    : any downsampling of the object before acquisition
%
% sys_model
%   NA      : number of projections along the 360 orbit
%   NB      : number of detector elements
%   DS      : horizontal sample spacing
%   DSD     : distance - source to dectector
%   DSO     : distance - source to object
%   MAX_FLUX: maximum value of input flux
%
% misc_info
%   plot_fig     : bool for plot figures. 'T' or 'F'
%   output_folder: path to save reconstructed 'raw' images
%   filter_str   : filter string name such as ramp, hann. For more info
%                  look in the file ../fbp/fbp2_window.m
%   k_ld         : % of normal dose. this optional argument if set yields 
%                  low_dose acquistion relative to normal dose. typical 
%                  values 0.25  
%                  for quater, 0.5 for half dose and so on. For more info
%                  look into ref 3 below.
%   k_nd         : parameter for noise in normal dose. Has linear relation
%                  compared to mAs level. Based on Heusterics and ref [2]
%                  set this value from the range [0.85, 1.0]
%   norm_type    : options include 'positive_scale' or 'non_neg'
%                  This normalization is applied to densitiy values after 
%                  reconstruction (and before transfering the reconstruction ...
%                  to CT values).
%  intercept_k   : set this option to 1024 for patient images when you use 
%                  pre-intercepted xtrue or normal dose images with 1024.
%                  Else set it to 0 where your inputs is not intercepted
%                  and the CT value of air is -1024. 
%  noiseless     : reconstruction without adding noise to input xobject.
%                  Default is False.
%  out_dtype     : default output dtype is int16.
% -------
% OUTPUT:                   
% -------
%  * Either normal dose acquisition of true object or low dose acquistion
%  if k_ld is set
%
% ------------
% References
% -------------
% [1] https://github.com/JeffFessler/mirt
% [2] Zeng, D., Huang, J., Bian, Z., Niu, S., Zhang, H., Feng, Q., Liang, 
%     Z. and Ma, J., 2015. A simple low-dose x-ray CT simulation from 
%     high-dose scan. IEEE transactions on nuclear science, 62(5), 
%     pp.2226-2233
% [3] Yu, L., Shiung, M., Jondal, D. and McCollough, C.H., 2012. 
%     Development and validation of a practical lower-dose-simulation tool 
%     for optimizing computed tomography scan protocols. Journal of 
%     computer assisted tomography, 36(4), pp.477-487. 

%%
    % ----------------------
    % Default paramaters
    % --------------------
    if ~isfield(misc_info, 'plot_figures'); plot_figures='F'; 
    else, plot_figures=misc_info.plot_figures; end

    if ~isfield(sys_info, 'nb'); NB=986; 
    else, NB=sys_info.nb; end

    if ~isfield(sys_info, 'na'); NA=2880; 
    else, NA=sys_info.na; end

    if ~isfield(sys_info, 'ds'); DS=0.95;
    else, DS= sys_info.ds; end

    if ~isfield(obj_info, 'down'); DOWN=1;
    else, DOWN= obj_info.down; end

    if ~isfield(obj_info, 'fov'); FOV=250;
    else, FOV= obj_info.fov; end

    if ~isfield(sys_info, 'max_flux'); MAX_FLUX = 2.25e5;
    else, MAX_FLUX=sys_info.max_flux; end    

    if ~isfield(misc_info, 'filter_string'); filter_string = '';
    else, filter_string=misc_info.filter_string; end 

    % the default scale factor for HU is 1024;
    % else set this equal to HU of air
    if ~isfield(misc_info, 'scalefac_k'); scalefac_k = 1024.0;
    else, scalefac_k=misc_info.scalefac_k; end 

    if ~isfield(misc_info, 'intercept_k'); intercept_k = 0.0;
    else, intercept_k=misc_info.intercept_k; end 

    % This parameter is imputed from ref [2]. For ND acquisition
    % around 83 mAs, equivalent k factor is 0.85. 
    if ~isfield(misc_info, 'k_nd'); k_nd = 0.85;
    else, k_nd=misc_info.k_nd; end 

    if ~isfield (misc_info, 'noiseless'); noiseless = 'False';
    else, noiseless = misc_info.noiseless; end 

    %
    if ~isfield(misc_info, 'out_dtype'); out_dtype = 'int16';
    else, out_dtype = misc_info.out_dtype; end

    if ~exist(misc_info.output_folder, 'dir')
        mkdir(misc_info.output_folder)
    end
    obj_info
    sys_info
    misc_info
    
    %---------------------------
    % system matrix formulation
    %---------------------------
    [Nx, Ny, Nz] = size(obj);
    ig   = image_geom('nx',Nx, 'ny', Ny,'fov',FOV,'offset_x',0,'down', DOWN);
    sgn  = sino_geom('fan','na',NA, 'nb', NB, 'ds', DS, ...
            'dsd', 1085.6, 'dso',595, ...
            'source_offset',0.0,'orbit',360, 'down', DOWN, 'strip_width', 'd');%0.909976; na 2304 
    G    = Gtomo2_dscmex(sgn, ig);
    load I0;
    I02 = imresize(I0, sgn.dim);
    I02 = I02*MAX_FLUX/max(I02(:));
    sig = sqrt(8); % standard variance of electronic noise, a characteristic of CT scanner
    clear I0;

    %---------------------------
    % setup recon template
    %---------------------------
    tmp         = fbp2(sgn, ig, 'type','std:mat');
    mu_water    = 0.21/10.0; % mm-1
    % intercept_k = 0; % as the digital phantom is already in accurate CT number with -ve intercept slope
    % scalefac_k  = 1024;
    figure (1); 
    for i=1:Nz
        xtrue = obj(:, :, i);
        xtrue = xtrue - intercept_k;
        xtrue = xtrue'*mu_water/scalefac_k+ mu_water;
        if (min(xtrue(:))<0)
            xtrue = xtrue + (-min(xtrue(:)));
        end
    
        sino_n = G*xtrue;
        
        if strcmp(noiseless, 'True')
            % setting projection to be noiseless
            ldsino = sino_n;
        else 
            % perform low-dose simulation on the projection data
            ndsino  = pct_ldsino(I02, sino_n, k_nd, sig, 'T');
            if ~isfield(misc_info, 'k_ld')
                % i.e. if LD acquisition factor relative to ND is not specified
                % then simply perform ND acquisition
                ldsino  = ndsino;
            else
                k_ld    = misc_info.k_ld;
                ldsino  = yuct_ldsino(I02, ndsino, k_ld, sig*sig);
            end
        end 

        % reconstruct image
        fbp = fbp2(ldsino, tmp, 'window', filter_string); %'hanning,0.6');%hanning,0.6');

        if strcmp(misc_info.norm_type, 'positive_scale')
            if (min(fbp(:))<0)
                fbp = fbp + (-min(fbp(:)));
            end
        elseif strcmp(misc_info.norm_type, 'remove_negative')
            if (min(fbp(:))<0)
                neg_ind = find(fbp<0);
                fbp(neg_ind) =0.0;
            end
        else 
            fbp;
        end
        xtrue = (xtrue' - mu_water)*scalefac_k/mu_water + intercept_k;
        fbp   = (fbp' - mu_water)*scalefac_k/mu_water + intercept_k;
        % imshow(fbp, []); colorbar;
        
        if strcmp(plot_figures, 'T')
            if strcmp(noiseless, 'True')
                im plc 1 3
                im(1, xtrue, 'ground truth'), cbar;
                im(2, sino_n, 'noiseless sino'), cbar;
                im(3, fbp, 'noiseless FBP'), cbar;
                colormap gray;
            else
                if ~isfield(misc_info, 'k_ld')
                    im plc 1 3
                    im(1, xtrue, 'ground truth'), cbar;
                    im(2, ldsino, 'normal-dose sino'), cbar;
                    im(3, fbp, 'normal-dose FBP'), cbar;
                    colormap gray;
                else
                    im plc 1 3
                    im(1, xtrue, 'ground truth'), cbar;
                    im(2, ldsino, 'low-dose sino'), cbar;
                    im(3, fbp, 'low-dose FBP'), cbar;
                    colormap gray;
                end 
            end 
        end 

         

        if ~isempty(misc_info.output_folder)
            % default dtype of output is int16
            if strcmp(out_dtype, 'uint16')
                write_raw_img([misc_info.output_folder,'/fbp_sharp_', num2str(i), '.raw'], uint16(fbp), 'uint16');
            else
                write_raw_img([misc_info.output_folder,'/fbp_sharp_', num2str(i), '.raw'], int16(fbp), 'int16');
            end
        end 
    end
    
end 