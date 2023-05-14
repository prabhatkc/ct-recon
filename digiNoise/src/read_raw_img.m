function img = read_raw_img(path, size, img_type)
fid = fopen(path);
img = fread(fid, size, img_type);
fclose(fid);
img = img';