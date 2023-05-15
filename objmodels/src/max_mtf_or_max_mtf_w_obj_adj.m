%{
    Always remember that matlab reads column first 
    ND
    0.6629    0.5644    0.3709    0.2911    0.1832
    0.6717    0.5680    0.3634    0.2979    0.1882
    0.6722    0.5653    0.3521    0.2976    0.1882
    0.6680    0.5638    0.3443    0.2978    0.1948
    0.6650    0.5593    0.3433    0.2984    0.1977
    0.6643    0.5584    0.3419    0.2975    0.1967
    0.6658    0.5570    0.3451    0.3073    0.1947
    0.6680    0.5536    0.3569    0.3186    0.1869
    0.6735    0.5607    0.3704    0.3228    0.1804
    0.6808    0.5657    0.3709    0.3256    0.1834
    0.6781    0.5593    0.3692    0.3255    0.1846
    0.6752    0.5542    0.3678    0.3228    0.1791
    0.6753    0.5536    0.3641    0.3226    0.1775
    QD
    0.6662    0.5878    0.4009    0.3133    0.1956
    0.6728    0.5881    0.3947    0.3091    0.1979
    0.6641    0.5844    0.3863    0.3166    0.2051
    0.6686    0.5795    0.3773    0.3071    0.2028
    0.6760    0.5836    0.3656    0.3073    0.1898
    0.6766    0.5831    0.3628    0.3019    0.1890
    0.6731    0.5778    0.3658    0.3144    0.1980
    0.6784    0.5705    0.3713    0.3208    0.2043
    0.6783    0.5665    0.3675    0.3271    0.2004
    0.6773    0.5646    0.3696    0.3335    0.2042
    0.6780    0.5666    0.3858    0.3366    0.2035
    0.6851    0.5821    0.3888    0.3431    0.1909
    0.6940    0.5759    0.3889    0.3397    0.1876
    BM3D-ND
    0.6522    0.5577    0.3648    0.2898    0.1765
    0.6567    0.5547    0.3556    0.2872    0.1767
    0.6577    0.5510    0.3453    0.2869    0.1778
    0.6569    0.5501    0.3407    0.2863    0.1822
    0.6580    0.5501    0.3411    0.2864    0.1831
    0.6406    0.5334    0.3319    0.2847    0.1761
    0.6538    0.5437    0.3410    0.2963    0.1795
    0.6560    0.5416    0.3498    0.3056    0.1785
    0.6638    0.5498    0.3618    0.3120    0.1762
    0.6698    0.5548    0.3639    0.3141    0.1759
    0.6652    0.5481    0.3608    0.3125    0.1969
    0.6664    0.5470    0.3604    0.3122    0.1946
    0.6692    0.5447    0.3593    0.3127    0.1941
    BM3D-QD
    0.6434    0.5653    0.3846    0.3017    0.1847
    0.6468    0.5625    0.3778    0.2976    0.1861
    0.6448    0.5620    0.3731    0.2985    0.1860
    0.6601    0.5686    0.3734    0.3047    0.1872
    0.6648    0.5644    0.3656    0.3037    0.1848
    0.6578    0.5585    0.3578    0.2994    0.1866
    0.6676    0.5655    0.3635    0.3077    0.1936
    0.6608    0.5553    0.3615    0.3087    0.1966
    0.6587    0.5489    0.3594    0.3097    0.1912
    0.6625    0.5516    0.3655    0.3199    0.1888
    0.6635    0.5525    0.3756    0.3248    0.1846
    0.6719    0.5636    0.3821    0.3272    0.1817
    0.6780    0.5640    0.3807    0.3269    0.1831

%}
% creation_option = {'obj_lp_max_mtf_of_qd_slices', 'obj_lp_hu_adjusted'};
% creation_type   = creation_option{2};
function [] = max_mtf_or_max_mtf_w_obj_adj(creation_type, hu_contrast, save_output)
    output_folder   = ['./data/', creation_type, '/', num2str(hu_contrast)];
    
    if strcmp(save_output, "True")
            if not(isfolder(output_folder))
               mkdir(output_folder)
            end
    end
    
    if strcmp(creation_type, 'obj_lp_max_mtf_of_qd_slices')
        dir_path  = '../digiNoise/data/3mm_sharp/QD/';
        imginfo.Format = 'DICOM';
    elseif strcmp(creation_type, 'obj_lp_max_mtf_of_qd_n_hu_adj')
       dir_path  = '../digiNoise/data/3mm_sharp/QD/';
       imginfo.Format = 'DICOM';
    % elseif strcmp(creation_type, 'obj_lp_bm3d_of_nd_slices ')
    %    dir_path  = '../digiNoise/data/3mm_sharp/QD/';
    %    imginfo.Format = 'DICOM';
    % based on the mtf output of BM3D on ND/QD, it does not seem to 
    % yield spatially resolved object models for high lps
    else
        disp('re-check the create_type string');
        exit;
    
    end 
    
    dir_path
    file_info = dir(dir_path);
    nfiles    = length(file_info)-2;
    if (nfiles<=0)
        error('Error occured. No files in the read directory path!')
    end
    
    img_stack = [];
    LL  = 1000;
    ww  = 100;
    for i=1:nfiles
        filepath = [dir_path file_info(i+2).name];
        if strcmpi(imginfo.Format, 'tif')
            init_img     = single(imread(filepath))+imginfo.RescaleIntercept;
        elseif strcmpi(imginfo.Format, 'raw')
            init_img     = single(read_raw_img(filepath, [imginfo.Height imginfo.Width], imginfo.dtype))+imginfo.RescaleIntercept;
        else
            imginfo  = dicominfo(filepath);
            init_img = single(dicomread(filepath))+imginfo.RescaleIntercept;
        end
        ind_high = find(init_img>=(LL+ (ww*.5)));
        ing_low  = find(init_img<=(LL- (ww*.5)));
        img      = init_img;
        img(ind_high) = LL+ (ww*.5);
        img(ing_low)  = LL- (ww*.5);
        img_stack = cat(3, img_stack, img);
    end 
    [dimX, dimY, dimZ] = size(img_stack); 
    
    img_8lp = img_stack(:, :, 3);
    % figure, imshow(img_stack(:, :, 3), [])
    % ax = gca
    % [ix, iy] =getpts(ax)
    % ix_8lp = [321.1111  320.2222  400.6667  400.6667  411.3333];
    % iy_8lp = [327.6667  385.8889  385.8889  327.6667  340.5556];
    
    objX_8lp                  = zeros(dimX,dimY);
    objX_8lp(320:400, 327:385)= img_8lp(320:400, 327:385);
    objX_8lp = imbinarize(objX_8lp, 950);
    figure(1), imshow(objX_8lp, []);
    
    img_7lp = img_stack(:, :, 12);
    % figure, imshow(img_7lp, [])
    % ax = gca;
    % [ix, iy] =getpts(ax);
    % ix_7lp = 222   222   302   302;
    % iy_7lp = 374.0000  427.0000  430.0000  374.0000;
    objX_7lp                  = zeros(dimX,dimY);
    objX_7lp(374:427, 222:302)= img_7lp(374:427, 222:302);
     
    objX_7lp = imbinarize(objX_7lp, 950);
    figure(1), imshow(objX_7lp, []);
    
    img_6lp = img_stack(:, :, 10);
    % figure, imshow(img_6lp, [])
    % ax = gca;
    % [ix, iy] =getpts(ax);
    %ix = 121.0000  195.0000  120.0000  196.0000
    %iy =  332.0000  334.0000  386.0000  386.0000
    objX_6lp                  = zeros(dimX,dimY);
    objX_6lp(332:386, 121:195)= img_6lp(332:386, 121:195);
    objX_6lp = imbinarize(objX_6lp, 950);
    figure(1), imshow(objX_6lp, []);
    
    img_5lp = img_stack(:, :, 2);
    % figure, imagsec(img_5lp); axis off;
    % ax = gca;
    % [ix, iy] =getpts(ax);
    %ix = 79.8835   78.9440  154.1000  154.1000
    %iy =  232.9211  282.9662  283.9286  232.9211
    objX_5lp                  = zeros(dimX,dimY);
    objX_5lp(232:282, 79:154)= img_5lp(232:282, 79:154);
    objX_5lp = imbinarize(objX_5lp, 950);
    figure(1), imshow(objX_5lp, []);
    
    img_4lp = img_stack(:, :, 13);
    % figure, imagesc(img_4lp); axis off;
    % ax = gca;
    % [ix, iy] =getpts(ax);
    % ix = 124.1393  124.1393  191.8587  192.8848
    % iy =  130.4617  185.3889  186.3697  130.4617
    objX_4lp                  = zeros(dimX,dimY);
    objX_4lp(130:185, 124:191)= img_4lp(130:185, 124:191);
    objX_4lp = imbinarize(objX_4lp, 950);
    figure(1), imshow(objX_4lp, []);
    
    if strcmp(creation_type, 'obj_lp_max_mtf_of_qd_slices')
        bw_Xobj = (objX_4lp + objX_5lp + objX_6lp + objX_7lp + objX_8lp)*hu_contrast;
    elseif strcmp(creation_type, 'obj_lp_max_mtf_of_qd_n_hu_adj')
        bw_Xobj = (objX_4lp*1900/1900 + objX_5lp*1900/1900 + objX_6lp*1800/1900 + objX_7lp*1700/1900 + objX_8lp*1600/1900)*hu_contrast;
    end 
    
    water_area   = makecircle(512, 200, 0.0, -1024.0);
    bw_Xobj      = bw_Xobj + water_area; 
    figure(1), imshow(bw_Xobj, []); title(['Line Pair with ', num2str(hu_contrast), ' HU']);
    
    if strcmp(save_output, "True")
        write_raw_img([output_folder, '/lp_', num2str(hu_contrast), '.raw'], bw_Xobj, 'int16');
    end  
end 
 