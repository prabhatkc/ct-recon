function [cir] = makecircle(N, R, inside_val, outsizeval)
    cir = ones(N);
    cir = cir*inside_val;
    mid = N/2;
    x   = -(mid):1:(mid-1);
    y   = -(mid):1:(mid-1);
    
    [X, Y]           = meshgrid(x, y);
    circ_index       = X.^2 + Y.^2;
    outsize_ind      = circ_index > R*R;
    cir(outsize_ind) = outsizeval;
end 
	
