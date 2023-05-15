function[inserted_cir] = insert_circle(img, rr, sc_center, in_value)
 [nx, ny] = size(img);
 mx =nx/2; my=ny/2;
 x   = -(mx):1:(mx-1);
 y   = -(my):1:(my-1);

 [X, Y]           = meshgrid(x, y);
 circ_index       = (X+sc_center(1)-mx).^2 + (Y+sc_center(2)-my).^2;
 inside_ind       = (circ_index <= rr*rr);
 inserted_cir     = img;
 inserted_cir(inside_ind) = in_value;
end 