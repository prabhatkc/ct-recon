%% 
% the object model for line pattern comes from average of scans from the
% path (averaged by using imageJ):
% /gpfs_projects/prabhat.kc/lowdosect/transfers/transfers_4_spie/exps/quant_analysis/mtflp/data/3mm_sharp/QD
% Here we rather use 3mm QD path than ND. Primarily because from the
% createbp/mtflp_of_avg_3mm_sharp analysis, it appears that QD mtf for different
% line patterns is higher than that corresponding to ND
%%
% clc; clear all;
% addpath('/home/prabhat.kc/Implementations/matlab/global_files');
function [] = z_average_threshold_of_qd_slices(hu_contrast, save_output)
    creation_type = 'obj_lp_avg_of_qd_slices';
    init_img      = imread('createbp/data/QD/AVG_3mm_QD.tif');
    init_img      = single(init_img)-1024;
    LL            = 1000;
    ww            = 100;
    output_folder = ['./data/', creation_type, '/', num2str(hu_contrast)];
    
    if strcmp(save_output, "True")
        if not(isfolder(output_folder))
           mkdir(output_folder)
        end
    end
    ind_high       = find(init_img>=(LL+ (ww*.5)));
    ing_low        = find(init_img<=(LL- (ww*.5)));
    Xobj           = init_img;
    Xobj(ind_high) = LL+ (ww*.5);
    Xobj(ing_low)  = LL- (ww*.5);
    
    %% 
    % Based on binarize thresholding for different values between 950 to 
    % 955, it seems that binalizing the object at 950 seems good enough
    %{
    load("data/bone_improfile.mat");
    %figure, hold on;
    for i=1:5
        bw = imbinarize(Xobj, 949+i);
        %figure, imshow(bw, [])
        c = improfile(bw, lp_th.cx{2}', lp_th.cy{2}');
        figure(i), plot(1:length(c), c);
    end
    %hold off
    %}
    %% 
    bw_Xobj     = imbinarize(Xobj, 950); 
    water_area  = makecircle(512, 200, 0.0, -1024.0);
    
    bw_Xobj = bw_Xobj*hu_contrast + water_area; 
    figure, imshow(bw_Xobj, []); title(['Line Pair with ', num2str(hu_contrast), ' HU']);
    %{
    bw_Xobj_340 = bw_Xobj*340 + water_area; 
    figure, imshow(bw_Xobj_340, []); title("Line Pair 340 HU");
    
    bw_Xobj_120 = bw_Xobj*120 + water_area; 
    figure, imshow(bw_Xobj_120, []); title("Line Pair 120 HU");
    
    bw_Xobj_n35 = bw_Xobj*(-35.0) + water_area; 
    figure, imshow(bw_Xobj_n35, []); title("Line Pair -35 HU");
    %}
    if strcmp(save_output, "True")
        write_raw_img([output_folder, '/lp_', num2str(hu_contrast), '.raw'], bw_Xobj, 'int16');
        % write_raw_img('data/obj_lp_avg_of_qd_slices/barP_340.raw', bw_Xobj_340, 'int16');
        % write_raw_img('data/obj_lp_avg_of_qd_slices/barP_120.raw', bw_Xobj_120, 'int16');
        % write_raw_img('data/obj_lp_avg_of_qd_slices/barP_n35.raw', bw_Xobj_n35, 'int16');
    end
    disp('only 7, 8, 9, 11, 13 lp patterns are appropriate for mtf analysis')
end 

