function [cx, cy, c, mc] = radial_profiles_fulllength(nps, ang, df)
%function [cx, cy,c, mc] = radial_profiles_fulllength(nps, ang, df);
%Return the radial profiles of the input NPS image. The radial
%line is specified by vector 'ang' (in degree)). 'df' is the frequency
%sampling interval, default 1. The length of each radial line equals to half of the
%diagonal line length.
%Outputs
%c: the intensity values for points (cx, cy)
%mc: mean of the c over all angles.
%cx, cy: the coordinate of each radial line
%
%09/01/2016, Rongping Zeng, FDA/CDRH/OSEL

if(nargin==2);
    df=1;
end

[xnps, ynps]=size(nps);
if (~mod(xnps,2)) %even number
    [ixnps,iynps]=ndgrid([-xnps/2:xnps/2-1]*df,[-ynps/2:ynps/2-1]*df);
    nl=floor((xnps/2)*sqrt(2));
    r=(nl-1)*df;
else
    [ixnps,iynps]=ndgrid([-(xnps-1)/2:(xnps-1)/2]*df,[-(ynps-1)/2:(ynps-1)/2-1]*df);
    nl=floor(((xnps-1)/2)*sqrt(2));
    r=(nl-1)*df;
   end
       
ang=ang*pi/180;
c=zeros(length(ang),nl);   
cx=c;cy=c;
for i=1:length(ang)
    %[cx,cy,c(i,:)]=improfile(ixnps, iynps, nps, [0 r*sin(ang(i))],[0 r*cos(ang(i))],nl,'bilinear');
    [cx(i,:),cy(i,:),c(i,:)]=improfile(ixnps, iynps, nps', [0 r*cos(ang(i))],[0 r*sin(ang(i))],nl,'bicubic');
end;
nan_mask=isnan(c);
c(isnan(c))=0;
csum=sum(c);
cweight=sum(~nan_mask);
cweight(cweight==0)=1;
mc=csum./cweight;