% Purpose: calculate NPS from water phantom scans
%clc;
%clear all; 


function [fr, nps1d, nps2d] = nps_of_catphanReal_4r_realization_path(file_path, file_type, data_source, misc)
    %defaults
    if ~isfield(misc, 'dtype')
	misc.dtype='uint16';
    end 
    fileinfo = dir(file_path);
    file1    = [file_path fileinfo(3).name]
    if strcmp(file_type, 'dicom')
        img_tmp = dicomread(file1);
        info = dicominfo(file1);
        dx = info.PixelSpacing(1);
    elseif strcmp(file_type, 'raw')
        nx_r=512; ny_r=512;
        img_tmp = read_raw_img(file1, [nx_r, ny_r], misc.dtype);
        if strcmp(data_source, 'acr')
            dx=0.4883;
        else
            dx=0.7421875;
        end
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
       
    if strcmp(data_source, 'nih')
        %Read in the repeatitive noisy scans
        n_slice = length(fileinfo)-2;
        n_scan = n_slice/2;
        n_sliceperperson = n_slice/n_scan;

        allslice = zeros(nx,ny,n_sliceperperson, n_scan);
        for i=1:(n_scan) 
            for j=1:n_sliceperperson
              (i-1)*n_sliceperperson+j
              file = [file_path fileinfo((i-1)*n_sliceperperson+j+2).name]
              if strcmp(file_type, 'dicom')
                allslice(:,:,j, i) = dicomread(file);
              elseif strcmp(file_type, 'raw')
                allslice(:,:,j, i) = read_raw_img(file, [nx_r, ny_r], 'uint16');
                %test = = read_raw_img(file, [nx_r, ny_r], 'uint16');
                %allslice(:,:,j, i) = m_normalize(0, 1412,test);
              else
                  error('define image types: dicom or raw');
              end
        end
        end

        %three options
        %img = reshape(allslice, nx, nx, n_slice); % Using all slices
        %img = squeeze(allslice(:,:,j,1)); %Using the slices from one scan
        %img = squeeze(allslice(:,:,2,:)); % Using the same one slice from all scans
        nps = 0;
        %{
        for j=1:size(allslice,3)
            img = squeeze(allslice(:,:,j,:)); % Using the same one slice from all scans
            %extract noise only images
            n_img = size(img,3);
            img_mean = mean(img,3);
            noise = zeros(nx, ny, n_img);
            for i=1:n_img
                noise(:,:,i) = img(:,:,i) - img_mean;
                noise_roi(:,:,i) =  noise(roi_xfov, roi_yfov);
            end
            noise = noise *sqrt(n_img/(n_img-1));
            noise_roi = noise(roi_xfov, roi_yfov,:);
            %Compute NPS (empirical method)
            nps = compute_nps(noise_roi) + nps;
        end
        %}
        
        figure;
        for j=1:size(allslice,3)
            img = squeeze(allslice(roi_xfov,roi_yfov,j,:)); % Using the same one slice from all scans
            %extract noise only images
            n_img = size(img,3);
            img_mean = mean(img,3);
            noise = zeros(nx_roi, nx_roi, n_img);
            for i=1:n_img
                noise_roi(:,:,i) = img(:,:,i) - img_mean;
                %noise_roi(:,:,i) =  noise(roi_xfov, roi_yfov);
                img_title=sprintf(['diff:img:', num2str(i)]);
                subplot(1, 1, 1);imshow(noise_roi(:, :, i), []);colorbar;
                title(img_title);
                pause(0.1);
            end
            noise_roi = noise_roi *sqrt(n_img/(n_img-1));
            %noise_roi = noise(roi_xfov, roi_yfov,:);
            %Compute NPS (empirical method)
            nps = compute_nps(noise_roi) + nps;
            img_title=sprintf(['nps:img:', num2str(j)]);
            subplot(1, 1, 1);imshow(nps, []);colorbar;
            title(img_title);
            pause(0.1);
            
        end
        
        nps2d = nps/size(allslice,3);
    %extract the 1D radial shape
    else
        nsim = length(fileinfo)-2;
        img = zeros(nx,nx,nsim);
        for i=1:nsim
            i
            filename = fileinfo(i+2).name
            file = [file_path filename];

          if strcmp(file_type, 'dicom')
            img(:,:, i) = dicomread(file);
          elseif strcmp(file_type, 'raw')
            img(:,:, i) = read_raw_img(file, [nx, ny], 'uint16');
            %test = = read_raw_img(file, [nx_r, ny_r], 'uint16');
            %allslice(:,:,j, i) = m_normalize(0, 1412,test);
          else
              error('define image types: dicom or raw');
          end
        end

        %extract noise only images
        img      = img(roi_xfov, roi_yfov, :);
        img_mean = mean(img,3);
        %noise_roi = zeros(nx_roi, nx_roi, nsim); 
        %figure;
        for i=1:nsim
            noise_roi(:, :, i)= img(:,:,i) - img_mean;
            %noise_roi(:,:,i) =  noise(roi_xfov, roi_yfov);
            img_title=sprintf(['diff:img:', num2str(i)]);
            %subplot(1, 1, 1);imshow(noise_roi(:, :, i), []);colorbar;
            %title(img_title);
            %pause(0.4);
        end
        noise_roi      = noise_roi *sqrt(nsim/(nsim-1));
        noise_roi_mean =mean(noise_roi, 3);
        % figure, imshow(noise_roi_mean, []); colorbar;
        
        %print('diff', '-dpdf')
        %Compute NPS
        nps2d = compute_nps(noise_roi);
        max_nps_ind = find(nps2d==max(nps2d(:)));
        nps2d(max_nps_ind) = min(nps2d(:));
    end
    
    ang = [0:1:180];
    [cx, cy,c, mc] = radial_profiles_fulllength(nps2d, ang);
    nps1d = mc;
    fr = Fs/2 *linspace(0, 1, (length(nps1d)));

%save nps, nps1d, one slice of sample image, freq_vec 
%figure;
%plot(fr,nps1d);
%hold on;
%plot(fr,localnps1d,'r');
%plot(fr,avgnps1d, 'g');
    
    
    
