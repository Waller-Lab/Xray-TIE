function [projected_thickness] = xray_tie (I2, IinVal, Mag, R2, mu, delta, ps, reg)
% xray_tie (I2, IinVal, Mag, R2, mu, delta, ps)
%I2 - phase contrast image at R2, input image assumed to be
%     I = IinVal .* exp(-mu*thickness), IinVal constant
% Mag is the magnification
% R2 is the distance the phase contrast image is from contact image
% mu is the linear attenuation coefficient (Beer's Law material
% is assumed to be homogenous and suffciently thin.
% delta is the real part of the deviation of the material's refractive
% index from unityR
% ps is pixel size
% algorithm obtained from Paganin Phase paper
[M, N] = size(I2);
%M
%N
%Proper way of creating frequency axis
wx =2*pi*(0:(M-1))/M; %Create unshifted default omega axis
wy =2*pi*(0:(N-1))/N; %Create unshifted default omega axis
%fx =1/ps*unwrap(fftshift(wx)-2*pi)/2/pi; 
fx = 1/ps*(wx-pi*(1-mod(M,2)/M))/(2*pi); %Shift zero to centre - for even case, pull back by pi, for odd case by pi(1-1/N)
fy = 1/ps*(wy-pi*(1-mod(N,2)/N))/(2*pi); %Shift zero to centre - for even case, pull back by pi, for odd case by pi(1-1/N)
[Fx,Fy] = meshgrid(fx,fy);
Fx = transpose(Fx);
Fy = transpose(Fy);

%Set up done time for the real action!
I2fft = fftshift(fft2(Mag^2 * I2));
I2fftScaled = (I2fft / IinVal) * mu;
denominator = (R2 * delta) * ((Fx.^2 + Fy.^2) / Mag) + mu + reg;

%I could have done this all at once but I split it up to
%check the values in the matrix with the CUDA version
%to see if my cuda kernels were working correctly
invTerm = I2fftScaled ./ denominator;
outputIFFT = real(ifft2(ifftshift(invTerm)));
log_output = log(outputIFFT);
projected_thickness = -(1/mu) * log_output;
end