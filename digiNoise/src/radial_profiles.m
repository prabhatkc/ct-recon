function c = radial_profiles(nps, ang, df);
%function c = radial_profiles(nps, ang, df);
%Return the radial profiles of the input NPS image. The radial
%line is specified by vector 'ang' (in degree)). 'df' is the frequency
%sampling interval, default 1.

if(nargin==2);
    df=1;
end

   [xnps, ynps]=size(nps);
   if (~mod(xnps,2)) %even number
       [ixnps,iynps]=ndgrid([-xnps/2:xnps/2-1]*df,[-ynps/2:ynps/2-1]*df);
       nl=(xnps/2-1);
       r=nl*df;
       
   else
       [ixnps,iynps]=ndgrid([-(xnps-1)/2:(xnps-1)/2]*df,[-(ynps-1)/2:(ynps-1)/2-1]*df);
       nl=(xnps-1)/2;
       r=nl*df;
   end
       
   ang=ang*pi/180;
   
   c=zeros(length(ang),nl+1);   
   
  
   for i=1:length(ang)
       %[cx,cy,c(i,:)]=improfile(ixnps, iynps, nps, [0 r*sin(ang(i))],[0 r*cos(ang(i))],nl+1,'bilinear');
        [cx,cy,c(i,:)]=improfile(ixnps, iynps, nps', [0 r*cos(ang(i))],[0 r*sin(ang(i))],nl+1,'bilinear');
   end;
%   nfft=xnps;
%   meanc=(sum(c(1:2,:))+sum(c(6:7,:)))/4;
%   plot(([nfft/2:nfft-1]-nfft/2)*2/(ig.dx*nfft),meanc,'-k');
%   xlabel 'f (mm^{-1})';
%   legend('-45^o','-30^o','-15^o','0^o','15^o','30^o','45^o','90^o','mean');
   