function [transimg] = trans_obj_in_img (img, x0, y0, translate_type)
    % translates object in an image to a new point (x0, y0).
    % the frame of reference is the center of the image 
    % (x0, y0) is in matlab based coordinate
    % translate_type:
    % - fft_based: uses fourier shift theorem
    % - matlab_based: uses matlab's imtranslate
    [Nx, Ny] = size(img);
    midx     = round(Nx/2);
    midy     = round(Ny/2);
    kx0      = round(x0)-midx;
    ky0      = round(y0)-midy;
    if strcmp(translate_type, 'fft_based')
        [kx, ky] = meshgrid(-midx:(midx-1), -midy:(midy-1));
        kfacx = 2*pi/Nx;
        kfacy = 2*pi/Ny;
        kx = kx.*kfacx;
        ky = ky.*kfacy;
        
        c1=fftshift(fft2(img));
        c2=c1.*exp(-1i.*(kx*(kx0)+ky*(ky0)));
        transimg=real(ifft2(ifftshift(c2)));
    else
        transimg= imtranslate(img, [kx0, ky0]);
    end
    %figure, imshow((acr_four_lp), [])

end 
