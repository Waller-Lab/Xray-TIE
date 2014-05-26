%This image is healthy 0 from the norm folder

%Uncomment one or the other but not both depending on which one
%you want to see
%I2 = imread('.\norm\20131116_192634_GWSS_D5_Healthy_0.tif'); %rectangular image
I2 = imread('.\input\xray_tie_input_0.tif'); %square power of 2 image

%remove the glue part (approximately the last 100 rows of the image)
[rows,cols] = size(I2);
%I2_cropped = I2(1:(rows-100),1:2560); %for rectangular image
I2_cropped = I2; %for square power of two image, no glue removal
figure();
imagesc(I2_cropped);
colormap gray;
colorbar;
title('Original Defocused Image');

%Parameters to projected thickness function
IinVal=1; %incident intensity
Mag = 1.0; %magnification
R2 = 30; %in millimeters (defocus distance)
mu = 0.00828; %in millimeters^-1 (linear attenuation coefficient)
delta = .0001;  %unitless (change in refractive index)
ps = .00325;  %in millimeters (pixel size)
reg = 0.1; %unitless (regularization)

%Above values were obtained by the information below:
%incident intensity: 1 (since the images I gave you were normalized using brightfield images taken without the sample present, I think this should be right)
%linear attenuation coefficient: 0.00828 mm^-1
%pixel size: .00325 mm
%magnification: 2x
%distance of defocused image: unsure, but on the order of 30-40 mm



%Now call the function with te above parameters and output the thickness
output = xray_tie(I2_cropped, IinVal, Mag, R2, mu, delta, ps, reg);
figure();
imagesc(output);
colormap gray;
colorbar;
title('Output Image (Projected Thickness)');