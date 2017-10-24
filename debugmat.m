clc;
clear;
close all;

addpath( genpath('/Users/mehedi/Google Drive/Code/matlabFunctions/') )

filename = '/Users/mehedi/Google Drive/mPerpetuo/images/dataSetD750/abc/DSC_1339.dng';

if ~exist('gamma_table', 'var')
    load GammaE.txt;
    gamma_table = GammaE;
end

%-------------------------------------------------------------------------
% Reading the raw file (the .dng file)
%-------------------------------------------------------------------------
t = Tiff(filename, 'r'); % opens tiff file for reading mode
offsets = getTag(t, 'SubIFD');

setSubDirectory(t, offsets(1));
raw = read(t); % Create variable 'raw', the Bayer CFA data
close(t);

%-------------------------------------------------------------------------
% Meta data
%-------------------------------------------------------------------------
meta_info = imfinfo(filename);

%-------------------------------------------------------------------------
% Crop to the valid area
%-------------------------------------------------------------------------
x_origin = meta_info.SubIFDs{1}.ActiveArea(2) + 1; % +1 due to Matlab indexing
width    = meta_info.SubIFDs{1}.DefaultCropSize(1);

y_origin = meta_info.SubIFDs{1}.ActiveArea(1) + 1;
height   = meta_info.SubIFDs{1}.DefaultCropSize(2);
raw      = double( raw(y_origin:y_origin+height-1, x_origin:x_origin+width-1) );

%-------------------------------------------------------------------------
% from sensor to storage a non-linear tranformation might have performed
% increase the bit depth; many cases this field does not exist
%-------------------------------------------------------------------------
if isfield(meta_info.SubIFDs{1}, 'LinearizationTable')
    lin_table  = meta_info.SubIFDs{1}.LinearizationTable;
    raw = lin_table(raw+1);
end

%-------------------------------------------------------------------------
% black and saturation level adjustment
%-------------------------------------------------------------------------
black       = meta_info.SubIFDs{1}.BlackLevel(1);
saturation  = meta_info.SubIFDs{1}.WhiteLevel;

lin_bayer   = (raw-black)/(saturation-black);
lin_bayer   = max(0, min(lin_bayer, 1));

%-------------------------------------------------------------------------
% Which Bayer Pattern
%-------------------------------------------------------------------------
temp = meta_info.SubIFDs{1,1}.UnknownTags;
for i = 1 : length(temp)
    if (temp(i).ID == 33422)
        pat = temp(i).Value;
        if (sum( abs(pat - [0 1 1 2]) ) == 0)
            bayer_pattern = 'rggb';
        elseif (sum( abs(pat - [2 1 1 0]) ) == 0)
            bayer_pattern = 'bggr';
        elseif (sum( abs(pat - [1 0 2 1]) ) == 0)
            bayer_pattern = 'grbg';
        elseif (sum( abs(pat - [1 2 0 1]) ) == 0)
            bayer_pattern = 'gbrg';
        end
    end
end

%-------------------------------------------------------------------------
% White Balance
%-------------------------------------------------------------------------
wb_multipliers = (meta_info.AsShotNeutral).^-1;
wb_multipliers = wb_multipliers/wb_multipliers(2);
mask = wbmask(size(lin_bayer, 1), size(lin_bayer, 2), wb_multipliers, bayer_pattern);
balanced_bayer = lin_bayer .* mask;

%-------------------------------------------------------------------------
% Demosaic/DeBayer
% Algo: Malvar, H.S., L. He, and R. Cutler, "High quality linear
%       interpolation for demosaicing of Bayer-patterned color images", 
%       ICASPP, 2004.
% (input uint8/uint16/uint32)
%-------------------------------------------------------------------------

lin_rgb = demosaic( uint16( balanced_bayer * (2^16 - 1) ), bayer_pattern );
lin_rgb = double( lin_rgb ) / (2^16 - 1);
lin_rgb( lin_rgb < 0 ) = 0;
lin_rgb( lin_rgb > 1 ) = 1;


%-------------------------------------------------------------------------
% Color Correction
%-------------------------------------------------------------------------
temp = meta_info.ColorMatrix2;
xyz2cam = [temp(1:3);temp(4:6);temp(7:9)];
rgb2xyz = [ 0.4124564  0.3575761  0.1804375;
            0.2126729  0.7151522  0.0721750;
            0.0193339  0.1191920  0.9503041 ]; % sRGB with D65
rgb2cam = xyz2cam * rgb2xyz;
rgb2cam = rgb2cam ./ repmat(sum(rgb2cam, 2), 1, 3); % normalize rows to 1
cam2rgb = rgb2cam^-1;
lin_srgb = apply_cmatrix(lin_rgb, cam2rgb);
lin_srgb = max(0, min(lin_srgb, 1));


%-------------------------------------------------------------------------
% Gamma Correction
%-------------------------------------------------------------------------
gamma_table = gamma_table./max(gamma_table);
lin_table = linspace(0, 1, length(gamma_table));
img_gamma = interp1( lin_table, gamma_table, lin_srgb );

figure;
imshow(img_gamma)

%==========================================================================