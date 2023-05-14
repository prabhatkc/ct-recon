function [lsino] = yuct_ldsino(I0, hsino, a, Ne)
% Simulate low-dose projections from normal (high) dose projection
% -----------
%  INPUT:
% -------------
%     IO: incident flux
%     hsino: high dose sinogram data
%     a: ratio between low-dose and normal-dose acquistion
%     Ne: variance of electronic noise 
% --------------
% Output
% --------------
% low-dose sinogram that is a*100% of ND level
% ----------
% ref
% ----------
%     Yu, L., Shiung, M., Jondal, D. and McCollough, C.H., 2012. 
%     Development and validation of a practical lower-dose-simulation tool 
%     for optimizing computed tomography scan protocols. Journal of 
%     computer assisted tomography, 36(4), pp.477-487. 

mterm = exp(hsino)./I0;
fterm = ((1-a)/a)*mterm;
sterm = 1+(((1+a)/a)*Ne*mterm);
gterm = randn(size(hsino));
lsino = hsino + sqrt(fterm.*sterm).*gterm;
ind = find(lsino<0);
lsino(ind) = 0;
