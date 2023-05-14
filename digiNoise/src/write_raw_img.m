function img = write_raw_img(path, arr, img_type)
fid = fopen(path, 'w');
%img = fread(fid,[256 256], 'int16');
img = fwrite(fid, arr', img_type);
fclose(fid);
%img = img';