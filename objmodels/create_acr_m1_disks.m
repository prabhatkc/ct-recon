
clc; clear all;
addpath('src/');
%----------------------------------
% Object background
%----------------------------------
nx   = 512;
dx   = 0.48828; % based on dicom header for LDGC 3mm sharp data
fov  = nx*dx; 
xbkg = makecircle(nx, 200, 0.0, -1000);

%----------------------------------
% 4-disks from acr's module1 
%----------------------------------
rr   = 24.0; % radius of each disk
img0 = single(zeros(nx, nx));
d1   = insert_circle(img0, rr, [166 164], -94);
d2   = insert_circle(img0, rr, [350 164], 955);
d3   = insert_circle(img0, rr, [166 346], 120);
d4   = insert_circle(img0, rr, [350 346], -1000);

disks = rot90(d1+d2+d3+d4,2);
acrm1 = xbkg + disks;

figure, imshow((acrm1), []);
write_raw_img('../digiNoise/data/true/acr_m1_512.raw', acrm1, 'int16');


