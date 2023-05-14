% Purpose: calculate NPS from water phantom scans

%get image information
function [fr, nps1d, nps2d] = nps_of_catphanSim_4r_realization_path(file_path, file_type, misc)
    
    fileinfo = dir(file_path);
    file1    = [file_path fileinfo(3).name];

    if ~isfield(misc, 'remove_inflection')
        remove_inflection = 'F';
    else
        remove_inflection = misc.remove_inflection;
    end
    if strcmp(file_type, 'dicom')
        img_tmp = dicomread(file1);
        info = dicominfo(file1);
        dx = info.PixelSpacing(1);
    elseif strcmp(file_type, 'raw')
        
        if ~isfield(misc, 'size')
            dim = [256, 256];
        else
            dim = misc.size;
        end

        if ~isfield(misc, 'dx')
            dx= 0.7421875;
        else
            dx = misc.dx;
        end

        if ~isfield(misc, 'dtype')
            dtype = 'int16';
        else
            dtype = misc.dtype;
        end
        img_tmp = read_raw_img(file1, dim, dtype);
    else
        error('define image types: dicom or raw');
    end
        
    
    [nx, ny] = size(img_tmp);
    fov = nx*dx;
    Fs = 1/dx;
    pix_size = dx;

    roi_xfov = ceil(nx/2)+[-32:31];
    roi_yfov = ceil(ny/2)+[-32:31];
    nx_roi = length(roi_xfov);
       

    nsim = length(fileinfo)-2;
    img = zeros(nx,nx,nsim);
    for i=1:nsim
        i
        filename = fileinfo(i+2).name
        file = [file_path filename];

      if strcmp(file_type, 'dicom')
        img(:,:, i) = dicomread(file);
      elseif strcmp(file_type, 'raw')
        img(:,:, i) = read_raw_img(file, [nx, ny], 'int16');
        %test = = read_raw_img(file, [nx_r, ny_r], 'uint16');
        %allslice(:,:,j, i) = m_normalize(0, 1412,test);
      else
          error('define image types: dicom or raw');
      end
    end

    %extract noise only images
    img      = img(roi_xfov, roi_yfov, :);
    img_mean = mean(img,3);
    %figure, imshow(img_mean, []); colorbar, title('img mean');

    %noise_roi = zeros(nx_roi, nx_roi, nsim); 
    %figure;
    for i=1:nsim
        noise_roi(:, :, i)= img(:,:,i) - img_mean;
        %%noise_roi(:,:,i) =  noise(roi_xfov, roi_yfov);
        %img_title=sprintf(['diff:img:', num2str(i)]);
        %subplot(1, 1, 1);imshow(noise_roi(:, :, i), []);colorbar;
        %title(img_title);
        %pause(0.4);
    end
    noise_roi      = noise_roi*sqrt(nsim/(nsim-1));
    nps_zz         = abs(noise_roi).^2;
    nps_zz         = sum(nps_zz(:));
    nps_zz         = nps_zz;

    noise_roi_mean = mean(noise_roi, 3);
    %figure, imshow(noise_roi_mean, []); title('noise roi mean'); colorbar;

    nps2d = compute_nps(noise_roi);
    if strcmp(remove_inflection, 'T')
        max_nps_ind = find(nps2d==max(nps2d(:)));
        nps2d(max_nps_ind) = min(nps2d(:));
    end
    % fprintf("nps2d_zz=%f\n", nps2d(32,32));
    % figure, imshow(nps2d, []); title('nps2d'); colorbar;pause;

    
    ang = [0:1:180];
    [cx, cy,c, mc] = radial_profiles_fulllength(nps2d, ang);
    nps1d = mc;
    fr = Fs/2 *linspace(0, 1, (length(nps1d)));


    
    
    
