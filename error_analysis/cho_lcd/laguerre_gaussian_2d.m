function u=laguerre_gaussian_2d(x,J,h)
%function u=laguerre_gaussian_2d(x,J,h)
%Calculate the Laguerre-gaussian function
%Inputs
%       x: 1d vector of pixel locations
%       J: # of channels
%       h: the Guassian width 
%
%2009, R Zeng, FDA/CDRH/OSEL
%==========================================================================
%           Legal Disclaimer
%This software and documentation (the "Software") were developed at the 
%Food and Drug Administration (FDA) by employees of the Federal Government
%in the course of their official duties. Pursuant to Title 17, Section 105 
%of the United States Code, this work is not subject to copyright 
%protection and is in the public domain. Permission is hereby granted, 
%free of charge, to any person obtaining a copy of the Software, to deal 
%in the Software without restriction, including without limitation the 
%rights to use, copy, modify, merge, publish, distribute, sublicense, or 
%sell copies of the Software or derivatives, and to permit persons to whom 
%the Software is furnished to do so. FDA assumes no responsibility 
%whatsoever for use by other parties of the Software, its source code, 
%documentation or compiled executables, and makes no guarantees, expressed 
%or implied, about its quality, reliability, or any other characteristic. 
%Further, use of this code in no way implies endorsement by the FDA or 
%confers any advantage in regulatory decisions. Although this software can 
%be redistributed and/or modified freely, we ask that any derivative works 
%bear some notice that they are derived from it, and any modified versions 
%bear some notice that they have been modified.
%==========================================================================

xsize=size(x);
x=x(:);
L=laguerre(2*pi*x.^2/h^2,J);
for j=0:J
    u(:,j+1)= L(:,j+1).*exp(-pi*x.^2/h^2);
end

scale=sqrt(2)/h;%sqrt(2/2)/h/2;
u=u*scale;
u=reshape(u,[xsize J+1]);

