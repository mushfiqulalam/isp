clc;
clear;
close all;

raw = read_raw_img('DSC_1339_768x512_rggb.raw', [768 Inf], 'uint16');

width  = size(raw, 2);
height = size(raw, 1);

raw1 = raw(:);

a = randi(length(raw1), [200 1]);
raw1(a) = 65535;

b = randi(length(raw1), [200 1]);
raw1(b) = 600;

raw2 = reshape(raw1, [height width]);

write_raw_img('DSC_1339_768x512_rggb_bpc_test.raw', raw2, 'uint16')

temp = raw2 - raw;
min(temp(:))
max(temp(:))


figure;
imshow(uint16(raw2))