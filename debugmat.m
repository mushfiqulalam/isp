clc;
clear;
close all;

img = imread('images/out_gamma.png');

figure;
imshow(img)

r = double( img(:, :, 1) );
g = double( img(:, :, 2) );
b = double( img(:, :, 3) );

rs = stdfilt(r);
gs = stdfilt(g);
bs = stdfilt(b);

figure('name', 'rs'); imshow(rs, []); colorbar;
figure('name', 'gs'); imshow(gs, []); colorbar;
figure('name', 'bs'); imshow(bs, []); colorbar;




disp('rs: min and max: ');
[min(rs(:)) max(rs(:))]
disp('gs: min and max: ');
[min(gs(:)) max(gs(:))]
disp('bs: min and max: ');
[min(bs(:)) max(bs(:))]