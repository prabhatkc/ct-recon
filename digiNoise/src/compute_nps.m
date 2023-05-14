function [nps, fi] = compute_nps(in)
%function [nps, fi] = compute_nps(in)

nsize = size(in);

nrealization = nsize(end);

switch length(nsize)
    case 2
        nps = zeros(nsize(1),1);
        for i=1:nrealization
            s=fftshift(fft(in(:,i)));
            nps = abs(s).^2 + nps;
        end
        nps = nps/(nsize(1)*nsize(2));
    case 3
        nps = zeros(nsize(1),nsize(2));
        for i=1:nrealization
            s=fftshift(fft2(in(:,:,i)));
            nps = abs(s).^2 + nps;
        end
        nps = nps/(nsize(1)*nsize(2)*nsize(3));
    case 4
        nps = zeros(nsize(1),nsize(2),nsize(3));
        for i=1:nrealization
            s=fftshift(fftn(in(:,:,:,i)));
            nps = abs(s).^2 + nps;
        end
        nps = nps/(nsize(1)*nsize(2)*nsize(3)*nsize(4));
        
    otherwise
        disp 'Not implemented!'
end
            