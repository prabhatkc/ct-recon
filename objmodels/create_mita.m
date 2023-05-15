
clc; clear all;
addpath('src/')
%----------------------------------
% First creating object background
%----------------------------------
nx   = 512;
dx   = 0.48828; % based on dicom header for acr 3mm sharp data
fov  = nx*dx; 
xbkg = makecircle(nx, 200, 0.0, -1024); 
d = 40;     % distance in mm between the obj center and insert center

% -----------------------------------------------------
% insert_info [x_center, y_center, radius, HU values]
% inserts are at 45 deg from center with radius 
% -----------------------------------------------------

insert_info = [...
     d*cosd(45)  d*sind(45)   5/2   7;      % 5 mm, 7 HU
    -d*cosd(45)  d*sind(45)   3/2  14;      % 3 mm,  14 HU 
     d*cosd(45) -d*sind(45)   7/2   5;      % 7 mm, 5 HU
    -d*cosd(45) -d*sind(45)  10/2   3;      % 10 mm, 3 HU
     ];

num_inserts = size(insert_info, 1);

% convert roi locations from mm to pixels 
% i.e. px = mm*(1/p.s.); 
% p.s. = fov/nx 
insert_centers = round(insert_info(:,1:2) * (nx/fov) + (nx+1)/2);
insert_radii = insert_info(:,3) * (nx/fov);

%----------------------------
% select insert and add them to backgroudn
mita = xbkg;
for idx_insert = 1:4
    center_x = insert_centers(idx_insert, 1); % 203
    center_y = insert_centers(idx_insert, 2); %117
    insert_r = insert_radii(idx_insert);      %7.53 
    % crop_roi for the observer study is 3 times 
    mita = insert_circle(mita, insert_r, [center_x, center_y], insert_info(idx_insert, 4));
end 
write_raw_img('../digiNoise/data/true/mita_512.raw', mita, 'int16');
figure, imshow(mita(150:350, 150:350), []); title('zoomed view'); colorbar;
