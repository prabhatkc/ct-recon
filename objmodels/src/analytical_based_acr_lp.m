% notes
% Resizing after rotation is incorrect. 
% Rotation does not change size of object. It simply yields 0 padded
% output which results in the increase in size.
% based on internal tuning
% nearest neighbour yields best outputs for imrotate as well as imresize.
% Also "seq(i+1)-1" yields pixel size of 5 if seq=1:5:N
% clc; clear all;

function [] = analytical_based_acr_lp(hu_contrast)
    % reading the physical CT scan to get (x,y) location of each of the lp in
    % the [512 x 512] img
    avg_img       = imread('./createbp/data/QD/AVG_3mm_QD.tif');
    creation_type = 'obj_lp_ana_based';
    output_folder = ['./data/', creation_type, '/', num2str(hu_contrast)];
    
    if not(isfolder(output_folder))
       mkdir(output_folder)
    end
    % -------------------------------------------------------------------------
    % central (x,y) location of each of the lp pattern
    % -------------------------------------------------------------------------
    %       % 4lp     % 5lp     % 6lp    % 7lp    % 8lp     %10 lp       % 9lp      %11
    irows = [159.1391  116.8354 159.1391 261.1659  364.4368  364.4368    404.2521   258.6774];
    icols = [155.7175  258.9885 359.1488 397.7199  359.1488  155.0954    258.9885   111.5474];
    
    % -------------------------------------------------------------------------
    % 4 lp/cm (2x)
    % -------------------------------------------------------------------------
    seq           = 1:5:70;
    n             = (length(seq));
    four_lp_scale = 2;
    four_lp       = zeros(62, 70);
    for i=1:2:n
        four_lp(:, seq(i):(seq(i+1)-1))=1.0;
    end
    [aNy, aNx]    = size(four_lp);
    paddim        = [round((aNx-aNy)/2) + round(aNx/2), round(aNx/2)];
    four_lp       = padarray(four_lp, paddim, 0, 'both');
    four_lp       = imrotate(four_lp, 45, "nearest");
    four_lp       = imresize(four_lp, 1/four_lp_scale, "nearest");
    ffour_lp      = pad_n_translate(four_lp, avg_img, irows(1), icols(1));
    ffour_lp      = imbinarize(ffour_lp, 0.95);
    figure(1), imshow(ffour_lp, [])
    
    % -------------------------------------------------------------------------
    % 5 lp/cm (2x)
    % -------------------------------------------------------------------------
    seq           = 1:4:64;
    n             = (length(seq));
    five_lp_scale = 2;
    five_lp       = zeros(62, 64);
    
    for i=1:2:n
        % seq substraction by 1 yields the best GT MTF
        five_lp(:, seq(i):(seq(i+1)-1))=1.0;
        %five_lp(:, seq(i):(seq(i+1)-zero_one_rep(jj)))=1.0;
        %jj = jj + 1;
    end
    [bNy, bNx]    = size(five_lp);
    paddim        = [round((bNx-bNy)/2) + round(bNx/2), round(bNx/2)];
    five_lp       = padarray(five_lp, paddim, 0, 'both');
    five_lp       = imrotate(five_lp, 45, "nearest");
    five_lp       = imresize(five_lp, 1/five_lp_scale, "nearest");
    ffive_lp      = pad_n_translate(five_lp, avg_img, irows(2), icols(2));
    ffive_lp      = imbinarize(ffive_lp, 0.95);
    figure(1),      imshow(ffive_lp, [])
    
    
    % -------------------------------------------------------------------------
    % 6 lp/cm (4x)
    % -------------------------------------------------------------------------
    seq           = 1:7:140;
    n             = (length(seq));
    six_lp_scale  = 4;
    six_lp        = zeros(124, 140);
    zero_one_rep  = repmat([0,1], [1, round(n/4)]);
    jj = 1;
    for i=1:2:n
        six_lp(:, seq(i):(seq(i+1)-zero_one_rep(jj)))=1.0;
        jj=jj+1;
    end
    [cNy, cNx]    = size(six_lp);
    paddim        = [round((cNx-cNy)/2) + round(cNx/2), round(cNx/2)];
    six_lp        = padarray(six_lp, paddim, 0, 'both');
    six_lp        = imrotate(six_lp,45, "nearest");
    six_lp        = imresize(six_lp, 1/six_lp_scale, "nearest");
    fsix_lp       = pad_n_translate(six_lp, avg_img, irows(3), icols(3));
    fsix_lp       = imbinarize(fsix_lp, 0.95);
    figure(1),      imshow(fsix_lp, [])
    
    % -------------------------------------------------------------------------
    % 7 lp/cm (2x)
    % -------------------------------------------------------------------------
    seq = 1:3:66;
    n   = (length(seq));
    seven_lp_scale  = 2;
    seven_lp        = zeros(62, 66);
    
    for i=1:2:n
        seven_lp(:, seq(i):(seq(i+1)-1))=1.0;
        %jj=jj+1;
    end
    [dNy, dNx]      = size(seven_lp);
    paddim          = [round((dNx-dNy)/2) + round(dNx/2), round(dNx/2)];
    seven_lp        = padarray(seven_lp, paddim, 0, 'both');
    seven_lp_p_dim  = size(seven_lp);
    seven_lp        = imrotate(seven_lp, 45, "nearest");
    seven_lp        = imresize(seven_lp, 1/seven_lp_scale, "nearest");
    fseven_lp       = pad_n_translate(seven_lp, avg_img, irows(4), icols(4));
    fseven_lp       = imbinarize(fseven_lp, 0.95);
    figure(1), imshow(fseven_lp, [])
    
    % -------------------------------------------------------------------------
    % 8 lp/cm (4x)
    % -------------------------------------------------------------------------
    seq             = 1:5:130;
    n               = (length(seq));
    eight_lp_scale  = 4;
    zero_one_rep    = repmat([0,1], [1, round(n/4)]);
    jj = 1;
    eight_lp = zeros(124, 130);
    for i=1:2:n
        %eight_lp(:, seq(i):(seq(i+1)))=1.0; % i.e. no minus yields poor
        %seperation
        eight_lp(:, seq(i):(seq(i+1)-1))=1.0;
        %eight_lp(:, seq(i):(seq(i+1)-zero_one_rep(jj)))=1.0;% minus zero
        %one pre. yields poor seperation between lp
        jj=jj+1;
    end
    [eNy, eNx]      = size(eight_lp);
    paddim          = [round((eNx-eNy)/2) + round(eNx/2), round(eNx/2)];
    eight_lp        = padarray(eight_lp, paddim, 0, 'both');
    eight_lp        = imrotate(eight_lp, 45, "nearest");
    eight_lp        = imresize(eight_lp, 1/eight_lp_scale, "nearest");
    feight_lp       = pad_n_translate(eight_lp, avg_img, irows(5), icols(5));
    feight_lp       = imbinarize(feight_lp, 0.95);
    figure(1),         imshow(feight_lp, [])
    
    % -------------------------------------------------------------------------
    % getting rest of the non-visible lp i.e. > 8lp/com
    % -------------------------------------------------------------------------
    [fNr, fNc] = size(avg_img);
    init_img   = single(avg_img)-1024;
    LL         = 1000;
    ww         = 100;
    
    ind_high       = find(init_img>=(LL+ (ww*.5)));
    ing_low        = find(init_img<=(LL- (ww*.5)));
    Xobj           = init_img;
    Xobj(ind_high) = LL+ (ww*.5);
    Xobj(ing_low)  = LL- (ww*.5);
    
    p1_Xobj        = imbinarize(Xobj, 950); 
    %%
    % figure, imagesc(p1_Xobj); axis off;
    % ax=gca;
    % [ix, iy]=getpts(ax)
    % 222.7186  222.7186  443.3536  444.4093
    % 80.7840  284.7761  285.7860   81.7939
    %%
    non_viz_lp_Xobj                  = zeros(fNr);
    non_viz_lp_Xobj(80:285, 222:444) = p1_Xobj(80:285, 222:444);
    lp_Xobj = (ffour_lp + ffive_lp + fsix_lp + fseven_lp + feight_lp + non_viz_lp_Xobj).*hu_contrast;
    
    % ----------------------------------------------------------------
    %  making & saving the object model 
    % ----------------------------------------------------------------
    water_area   = makecircle(fNr, 200, 0.0, -1024.0);
    bw_lp_Xobj   = lp_Xobj + water_area; 
    figure(2),     imshow(bw_lp_Xobj, []); title(['Line Pair', num2str(hu_contrast), 'HU']);
    
    write_raw_img([output_folder, '/lp_', num2str(hu_contrast), '.raw'], bw_lp_Xobj, 'int16');
end 

function [pt_img]    = pad_n_translate(small_lp_img, ref_img, ir, ic)
    [fNr, fNc]       = size(ref_img);
    [sNr, sNc]       = size(small_lp_img);
    small_lp_fpaddim = [round((fNr-sNr)/2), round((fNc-sNc)/2)];
    full_lp_img      = padarray(small_lp_img, small_lp_fpaddim, 0, 'both');
    full_lp_img      = full_lp_img(1:fNr, 1:fNc);
    pt_img           = trans_obj_in_img(full_lp_img, ir, ic, 'matlab_based');
end


