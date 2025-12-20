function y_hat_RW = sadct_blind_deblur_gray (input) 
% Pointwise Shape-Adaptive DCT image deblurring (using low-complexity SA-DCT)
%
% Alessandro Foi - Tampere University of Technology - 2005 - 2016   Public release v1.43 (15 May 2016)   C-code implementation by Kostadin Dabov
% -----------------------------------------------------------------------------------------------------------------------------------------------
%
% Observation model:
%
%  z = y * v + n
%
% z : blurred noisy observation
% y : true image (assumed as unknown)
% v : point-spread function (assumed as known)
% n : Gaussian white noise (with unknown variance)
% * : convolution
%
% Other key variables:
% zRI      : "unfiltered" regularized inverse estimate
% y_hat_RI : filtered RI estimate
% zRW      : "unfiltered" regularized Wiener inverse estimate
% y_hat_RW : filtered RW estimate (final estimate)
%
%
% The code implements the deblurring algorithm and reproduces the experiments published in:
% Foi, A., K. Dabov, V. Katkovnik, and K. Egiazarian, "Shape-Adaptive DCT for Denoising and Image Reconstruction",
% Proc. SPIE Electronic Imaging 2006, Image Processing: Algorithms and Systems V, 6064A-18, San Jose, January 2006.
%

clear yh_RI yh_RW h_opt_Q stdh_RI stdh_RW gh hadper;
close all; drawnow;

% ----------------------------------------------------------------------
% INPUT SIGNAL SETTING
%-----------------------------------------------------------------------
% Four experiments are prepared as described in the aforementioned paper.
% Each experiment has either different Point-Spread Function (PSF) or noise variance as follows:
% Exp. 1 - Cameraman 256x256, blur: 9x9 uniform "box-car" blur, Gaussian white noise: BSNR 40dB (sigma^2=0.308)
% Exp. 2 - Cameraman 256x256, blur: 15x15 1/(x^2+y^2), x,y=-7,...,7, kernel, Gaussian white noise: sigma^2=2
% Exp. 3 - Cameraman 256x256, blur: 15x15 1/(x^2+y^2), x,y=-7,...,7, kernel, Gaussian white noise: sigma^2=8
% Exp. 4 - Lena 512x512, blur: 5x5 separable [1, 4, 6, 4, 1]/16 filter, Gaussian white noise: sigma^2=49

 % experiment number ( 1, 2, 3 or 4 )

% Some options
estimate_sigma  = 1; % estimate variance of the noise or use true value? (1 or 0)
do_wiener       = 1; % do_wiener = 0 does not perform RW stage ( = 1 performs, dafault )
detail_figures  = 1; % shows also figures with details of scales and estimates


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ALGORITHM'S PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ICI threshold
% -------------
% GammaParameterRI  ICI Gamma for RI  (default =0.9)
% GammaParameterRW  ICI Gamma for RW  (default =1.7)
% (the larger the Gamma, the larger the adaptive neighborhoods)
%
% SA-DCT filtering parameters:
% ------------------------
% DCTthrCOEF  Threshold factor for Hard-Thresholding in SA-DCT domain (RI stage)      (default =0.88)
% DCTwieCOEF  Regularization factor for Wiener-filtering in SA-DCT domain (RW stage)  (should be =1, i.e. standard Wiener-filtering)
%

%%% RI %%%
GammaParameterRI=0.9;
DCTthrCOEF=0.88;
%%% RW %%%
GammaParameterRW=1.7;
DCTwieCOEF=1.0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%---------------------------------------------------------
% LPA WINDOWS PARAMETERS
%---------------------------------------------------------
h1RI=[1 2 4]; % set of scales used for LPA in RI
h1RW=[1 2 4]; % set of scales used for LPA in RW
% for sake of computation time, it is STRONGLY recommended not to use too large
% (e.g. >6) scales nor too many (e.g. >5) different scales.
% [1 2 4] is a reasonable (and recommended) set of scales for both RI and RWI
max_overlap=49;    %% limits overcompleteness (marginal speed-up)  (recommended =49)  (note: lowering the value may give some speed-up in Pointwise SA-DCT RI hard-thresholding)
max_overlapW=49;   %% limits overcompleteness (marginal speed-up)  (recommended =49)  (note: lowering the value may give some speed-up in Pointwise SA-DCT RW Wiener-filtering)

alphaorderRI=-0.7;    % LPA order-mixture parameter (-1 zero order, 0 first order,  recommended =-0.7)
alphaorderRW=-0.8;    % LPA order-mixture parameter (-1 zero order, 0 first order,  recommended =-0.8)

%---------------------------------------------------------
% size of the FFT domain used for the approximate and FAST computation of
% the SA transform coefficients' variance
%---------------------------------------------------------
small_size_RI=32;     % size of the FFT domain used for the approximate and FAST computation of the SA transform coefficients' variance (recommended =32)
small_size_RW=32;     % size of the FFT domain used for the approximate and FAST computation of the SA transform coefficients' variance (recommended =32)
% Recommendations: Avoid anything larger than 64;
%                  There is no significant speed-up decreasing further below 16;
%                  32 is a good compromise between accuracy and speed.
%                  (IMPORTANT NOTE: since version 1.30, only small_size = 32 is implemented using FFT, for all other values DFT is used)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  EXPERIMENT SETUP  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Observation Parameters
% ----------------------
% BSNR is the desired Blurred-Signal-to-Noise-Ratio
% v is the blur Point-Spread Function (PSF)
%
% Regularization Parameters:
% --------------------------
% Regularization_epsilon_RI  Regularization parameter of Regularized Inverse operator of RI stage
% Regularization_epsilon_RW  Regularization parameter of Regularized Wiener Inverse operator of RW stage
%

K = [
  1.48504099e-05 6.45483930e-05 2.02405349e-04 4.57876225e-04 7.47246273e-04 8.79769657e-04 7.47246273e-04 4.57876225e-04 2.02405349e-04 6.45483930e-05 1.48504099e-05;
  6.45483930e-05 2.80564313e-04 8.79769657e-04 1.99019251e-03 3.24796059e-03 3.82398318e-03 3.24796059e-03 1.99019251e-03 8.79769657e-04 2.80564313e-04 6.45483930e-05;
  2.02405349e-04 8.79769657e-04 2.75870670e-03 6.24067603e-03 1.01846780e-02 1.19909205e-02 1.01846780e-02 6.24067603e-03 2.75870670e-03 8.79769657e-04 2.02405349e-04;
  4.57876225e-04 1.99019251e-03 6.24067603e-03 1.41174983e-02 2.30395191e-02 2.71255549e-02 2.30395191e-02 1.41174983e-02 6.24067603e-03 1.99019251e-03 4.57876225e-04;
  7.47246273e-04 3.24796059e-03 1.01846780e-02 2.30395191e-02 3.76001065e-02 4.42684480e-02 3.76001065e-02 2.30395191e-02 1.01846780e-02 3.24796059e-03 7.47246273e-04;
  8.79769657e-04 3.82398318e-03 1.19909205e-02 2.71255549e-02 4.42684480e-02 5.21194132e-02 4.42684480e-02 2.71255549e-02 1.19909205e-02 3.82398318e-03 8.79769657e-04;
  7.47246273e-04 3.24796059e-03 1.01846780e-02 2.30395191e-02 3.76001065e-02 4.42684480e-02 3.76001065e-02 2.30395191e-02 1.01846780e-02 3.24796059e-03 7.47246273e-04;
  4.57876225e-04 1.99019251e-03 6.24067603e-03 1.41174983e-02 2.30395191e-02 2.71255549e-02 2.30395191e-02 1.41174983e-02 6.24067603e-03 1.99019251e-03 4.57876225e-04;
  2.02405349e-04 8.79769657e-04 2.75870670e-03 6.24067603e-03 1.01846780e-02 1.19909205e-02 1.01846780e-02 6.24067603e-03 2.75870670e-03 8.79769657e-04 2.02405349e-04;
  6.45483930e-05 2.80564313e-04 8.79769657e-04 1.99019251e-03 3.24796059e-03 3.82398318e-03 3.24796059e-03 1.99019251e-03 8.79769657e-04 2.80564313e-04 6.45483930e-05;
  1.48504099e-05 6.45483930e-05 2.02405349e-04 4.57876225e-04 7.47246273e-04 8.79769657e-04 7.47246273e-04 4.57876225e-04 2.02405349e-04 6.45483930e-05 1.48504099e-05
];

K = K / sum(K(:));        % ensure normalized PSF

P = abs(fft2(K, 1024, 1024)).^2;   % power spectrum samples

% Kernel-only rule: pick power-floor as high percentiles of |H|^2
Regularization_epsilon_RI = prctile(P(:), 89);
Regularization_epsilon_RW = prctile(P(:), 92);
z = im2double(input);
y = im2double(input);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  GENERATES BLURRED AND NOISY OBSERVATION  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  ESTIMATION OF THE SIGMA  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if estimate_sigma==0
    sigma=org_sigma;
else
    sigma_est=function_stdEst(z);
    sigma=sigma_est;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Algorithm's Structural Parameters (DO NOT TOUCH THESE) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h2RI=ones(size(h1RI)); % To have line LPA kernels in RI
h2RW=ones(size(h1RW)); % To have line LPA kernels in RW
lenhRI=length(h2RI); % number of scales in RI
lenhRW=length(h1RW); % number of scales in RW

version -release; % get matlab release
matlab_R=str2num(ans);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%    FILTERING STARTS     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Regularized Inverse   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('  ')
disp('starting RI ...     ')
[kernels0, kernels_higher_order0]=function_CreateLPAKernels([0,0],h1RI,h2RI,10,1,1,ones(2,lenhRI),0);
[kernels1, kernels_higher_order1]=function_CreateLPAKernels([1,0],h1RI,h2RI,10,1,1,ones(2,lenhRI),0);

h_max=max(h1RI);
h1=h1RI;
lenh=lenhRI;
directional_resolution=8;
clear Trian TrianRot TrianRotu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BUILDS TRIANGLE MASKS FOR STARSHAPED SET  (kernel lenghts as verteces)
for h_opt_1=h1RI
    for h_opt_2=h1RI
        Trian{h_opt_1,h_opt_2}=zeros(2*h_max-1);
        for i1=h_max-h_opt_2+1:h_max
            for i2=2*h_max-i1:(h_max-1+h_opt_1-(h_max-i1)*((h_opt_1-h_opt_2)/(h_opt_2-1+eps)))
                Trian{h_opt_1,h_opt_2}(i1,i2)=1;
            end
        end
    end
end
% BUILDS ROTATED TRIANGLE MASKS   (for the eight directions)
for ii=1:8
    for h_opt_1=h1RI
        for h_opt_2=h1RI
            if mod(ii,2)==0
                TrianRotu{h_opt_1,h_opt_2,ii}=uint8(rot90(Trian{h_opt_2,h_opt_1}',mod(2+floor((ii-1)/2),8)));
            else
                TrianRotu{h_opt_1,h_opt_2,ii}=uint8(rot90(Trian{h_opt_1,h_opt_2},mod(floor((ii-1)/2),8)));
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[size_z_1,size_z_2]=size(z);
[ghy,ghx]=size(K);
%%  BLURRING  %%%%%%%%%
big_v=zeros(size_z_1,size_z_2); big_v(1:ghy,1:ghx)=K; big_v=circshift(big_v,-round([(ghy-1)/2 (ghx-1)/2])); % pad PSF with zeros to whole image domain, and centers it.
V=fft2(big_v);
%%%%%%%%%%%%%%% Transfer matrix for RI
RI= conj(V)./( (abs(V).^2) +Regularization_epsilon_RI^2); % Transfer Matrix for RI    %% Standard Tikhonov Regularization
ri=real(ifft2(RI));    % Impulse response of the RI
zRI=real(ifft2( fft2(z).* RI ));   % Regularized Inverse Estimate (RI OBSERVATION)
%%  (approximate) square of RI modulus in downsampled Fourier domain  (sect. 6.5 of paper)
small_size=small_size_RI;
tran_fftmatrix=(ifft(small_size*eye(2*h_max-1),small_size))';
tran_fftmatrix_fast=tran_fftmatrix(:,1:end/2+1);
conj_fftmatrix=conj(tran_fftmatrix');
low_pass_k = ones(round(size_z_1/small_size),round(size_z_2/small_size));  %% low-pass kernel
low_pass_k = conv2(low_pass_k,ones(2));
low_pass_k = low_pass_k/(sum(low_pass_k(:)));
RIsmallabs2 = conv2(repmat(abs(RI).^2,[3 3]),low_pass_k,'same');     %% low-pass filtering
RIsmallabs2 = RIsmallabs2(size_z_1+1+round((size_z_1/small_size)*(0:small_size-1)),size_z_2+1+round((size_z_2/small_size)*(0:small_size-1)));    %% decimation

%%%%%%%%%%%%%  ANISOTROPIC LPA-ICI  %%%%%%%%%%%%%%%%%%%
for s1=1:8 % cycle along the directions
    for s=1:lenhRI,
        gh=(1+alphaorderRI)*kernels1{1,s}-alphaorderRI*kernels0{1,s}; % gets single kernel from the cell array
        if ~rem(s1,2) % rotates kernel 45 degrees if even direction
            for iziz=1:size(gh,2)
                gh(:,iziz)=circshift(gh(:,iziz),[(size(gh,2)+1)/2-iziz,0]);
            end
        end
        gh=rot90(gh,floor((s1-1)/2)); % rotates kernel 90 degrees every second direction
        %%%%%%%%%%%%%%%%%% LPA-RI  %%%%%%%%%%
        yh_RI(:,:,s)=conv2(zRI-100000,gh,'same')+100000;   %%% LPA Filtering (in spatial domain)
        stdh_RI(:,:,s)=repmat(sqrt(sum(sum(conv2(ri([size_z_1-h1RI(s)+2:size_z_1 1:size_z_1 1:h1RI(s)-1],[size_z_2-h1RI(s)+2:size_z_2 1:size_z_2 1:h1RI(s)-1]),gh,'valid').^2)))*sigma,[size_z_1,size_z_2]);   % STD of THE ESTIMATE LPA-RI
    end %%%%%%%%%%%%%%% end for H  %%%%%%%%%%%%%%%%%%%%
    %%%%%% ICI %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [YICI_RIT,h_optRI,std_optRI1]=function_ICI(yh_RI,stdh_RI,GammaParameterRI,2*(s1-1)*pi/8);
    h_opt_Q(:,:,s1)=h_optRI; %% STORES RESULTS OF ADAPTIVE SCALES (USED FOR ADAPTIVE NEIGHBORHOOD)
end   %%% END THETA LOOP
clear std_optRI1 YICI_RIT stdh_RI yh_RI;
if detail_figures
    h_optRI=int8(h_optRI);  %% kept only for visualization
else
    clear h_optRI;
end   %%%%%%%%%%%%% END OF ANISOTROPIC LPA-ICI %%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%  SA-DCT PART STARTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SSS=[1:(2*h_max-1)^2];              %%% vector with the possible sizes of the adaptive-shapes
T=DCTthrCOEF*sqrt(2*log(SSS)+1);    %%% Universal-threshold (depends on the size of the adaptive-shape)


% BUILDS DCT BASES (ALL SIZES ARE NEEDED)
for h=1:2*h_max-1;
    hadper{h}=dct(eye(h));
end

% We process neighborhoods with identical shape one after another, simply by following the lexicographic order induced by the directional adaptive scales h_opt_Q which define the shape.
h_opt_Q = double(h_opt_Q);
h_opts_matrix=h_opt_Q(:,:,1);
for iiii=2:8
    h_opts_matrix=h_opts_matrix+h_opt_Q(:,:,iiii)*10^(iiii-1);
end
P2P=[[1:(size_z_1*size_z_2)]',h_opts_matrix(:)];
P2P=P2P(randperm((size_z_1*size_z_2)),:);
P2P=sortrows(P2P,2);  %% sort according to lexicographic order
h_opt_Q=uint8(h_opt_Q);

y_hat_RI = function_SADCT_RI_thresholding (h1, h_opt_Q, hadper, TrianRotu, conj_fftmatrix, tran_fftmatrix_fast, small_size, RIsmallabs2, zRI, uint32(P2P),  sigma, DCTthrCOEF,T,h_max,max_overlap);
clear P2P
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  End of Regularized Inverse   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RI_toc=toc;
disp(sprintf(repmat('\b',[1 2+20]))),
disp(['RI completed in ',num2str(RI_toc),' seconds.           '])
[Err_RI,Err_labels]=function_Errors(y,y_hat_RI,z);  %% computes error criteria
%%% PRINT RESULTS TO SCREEN
number_of_digits=7;
if do_wiener>=1
    disp('starting RW ...     ')
    Err_RW=repmat('wait ...',[size(Err_RI,1) 1]);
else
    Err_RW=[repmat(' ',[size(Err_RI,1) floor((number_of_digits-1)/2)]),repmat('--',[size(Err_RI,1) 1]),repmat(' ',[size(Err_RI,1) ceil((number_of_digits-1)/2)])];
end
disp(' ');
tab_title=[repmat(' ',[1 size(Err_labels,2)+1+floor(number_of_digits/2)]),'RI',repmat(' ',[1 number_of_digits+1]),'RW'];
tab_content=[Err_labels,repmat(' ',[size(Err_RI,1) 1]),num2str(Err_RI,number_of_digits),repmat(' ',[size(Err_RI,1) 2]),Err_RW];
disp(tab_title);
disp(tab_content);
figure, imshow(y_hat_RI), title('RI SA-DCT estimate'), drawnow
if do_wiener>=1    %%%%%%%%%%%%  RW Estimation %%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Regularized Wiener   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Wiener_Pilot=abs(fft2(y_hat_RI));   %%% WIENER PILOT ESTIMATE
    [kernels0, kernels_higher_order0]=function_CreateLPAKernels([0,0],h1RW,h2RW,10,1,1,ones(2,lenhRW),0);
    [kernels1, kernels_higher_order1]=function_CreateLPAKernels([1,0],h1RW,h2RW,10,1,1,ones(2,lenhRW),0);

    directional_resolution=8;
    lenh=lenhRW;
    h_max=max(h1RW);
    h1=h1RW;
    clear Trian TrianRot TrianRotu
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % BUILDS TRIANGLE MASKS FOR STARSHAPED SET  (kernel lenghts as verteces)
    for h_opt_1=h1RW
        for h_opt_2=h1RW
            Trian{h_opt_1,h_opt_2}=zeros(2*h_max-1);
            for i1=h_max-h_opt_2+1:h_max
                for i2=2*h_max-i1:(h_max-1+h_opt_1-(h_max-i1)*((h_opt_1-h_opt_2)/(h_opt_2-1+eps)))
                    Trian{h_opt_1,h_opt_2}(i1,i2)=1;
                end
            end
        end
    end
    % BUILDS ROTATED TRIANGLE MASKS   (for the eight directions)
    for ii=1:8
        for h_opt_1=h1RW
            for h_opt_2=h1RW
                if mod(ii,2)==0
                    TrianRotu{h_opt_1,h_opt_2,ii}=uint8(rot90(Trian{h_opt_2,h_opt_1}',mod(2+floor((ii-1)/2),8)));
                else
                    TrianRotu{h_opt_1,h_opt_2,ii}=uint8(rot90(Trian{h_opt_1,h_opt_2},mod(floor((ii-1)/2),8)));
                end
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%% Transfer matrix for RW
    PSD=(size_z_1*size_z_2)*sigma^2;  %Power Spectrum of AWGN noise
    RW=conj(V).*Wiener_Pilot.^2./(Wiener_Pilot.^2.*(abs(V).^2) + PSD*Regularization_epsilon_RW);   % Transfer Matrix for RW    %% Standard Regularization 'a-la-Tikhonov'
    rw=real(ifft2(RW));
    zRW=real(ifft2(fft2(z).*RW));   % RW OBSERVATION
    %%  (approximate) square of RW modulus in downsampled Fourier domain (sect. 6.5 of paper)
    small_size=small_size_RW;
    tran_fftmatrix=(ifft(small_size*eye(2*h_max-1),small_size))';
    tran_fftmatrix_fast=tran_fftmatrix(:,1:end/2+1);
    conj_fftmatrix=conj(tran_fftmatrix');
    low_pass_k = ones(round(size_z_1/small_size),round(size_z_2/small_size));  %% low-pass kernel
    low_pass_k = conv2(low_pass_k,ones(2));
    low_pass_k = low_pass_k/(sum(low_pass_k(:)));
    RWsmallabs2 = conv2(repmat(abs(RW).^2,[3 3]),low_pass_k,'same');     %% low-pass filtering
    RWsmallabs2 = RWsmallabs2(size_z_1+1+round((size_z_1/small_size)*(0:small_size-1)),size_z_2+1+round((size_z_2/small_size)*(0:small_size-1)));    %% decimation
clear gh
    %%%%%%%%%%%%%  ANISOTROPIC LPA-ICI  %%%%%%%%%%%%%%%%%%%
    for s1=1:8  % cycle along the directions
        for s=1:lenhRW,
            gh=(1+alphaorderRW)*kernels1{1,s}-alphaorderRW*kernels0{1,s}; % gets single kernel from the cell array
            if ~rem(s1,2) % rotates kernel 45 degrees if even direction
                for iziz=1:size(gh,2)
                    gh(:,iziz)=circshift(gh(:,iziz),[(size(gh,2)+1)/2-iziz,0]);
                end
            end
            gh=rot90(gh,floor((s1-1)/2)); % rotates kernel 90 degrees every second direction
            %%%%%%%%%%%%%%%%%% LPA-RW  %%%%%%%%%%
            yh_RW(:,:,s)=conv2(zRW-100000,gh,'same')+100000;   %%% LPA Filtering (in spatial domain)
            stdh_RW(:,:,s)=repmat(sqrt(sum(sum(conv2(rw([size_z_1-h1RW(s)+2:size_z_1 1:size_z_1 1:h1RW(s)-1],[size_z_2-h1RW(s)+2:size_z_2 1:size_z_2 1:h1RW(s)-1]),gh,'valid').^2)))*sigma,[size_z_1,size_z_2]);   % STD of THE ESTIMATE LPA-RW
        end %%%%%%%%%%%%%%% end for H  %%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%% ICI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [YICI_RWT,h_optRW,std_optRW1]=function_ICI(yh_RW,stdh_RW,GammaParameterRW,2*(s1-1)*pi/8);
        h_opt_Q(:,:,s1)=h_optRW;   %% STORES RESULTS OF ADAPTIVE SCALES (USED FOR ADAPTIVE NEIGHBORHOOD)
    end  % end theta
    %%%%%%%%%%%%% END OF ANISOTROPIC LPA-ICI %%%%%%%%%%%%%%%%
    clear h_optRW std_optRW1 YICI_RWT stdh_RW yh_RW;
    %%%%%%% SA-TRANSFORM FILTERING STARTS %%%%%%%%%%

    % BUILDS DCT BASES (ALL SIZES ARE NEEDED)
    clear hadper
    for h=1:2*h_max-1;
        hadper{h}=dct(eye(h));
    end

    % We process neighborhoods with identical shape one after another, simply by following the lexicographic order induced by the directional adaptive scales h_opt_Q which define the shape.
    h_opt_Q = double(h_opt_Q);
    h_opts_matrix=h_opt_Q(:,:,1);
    for iiii=2:8
        h_opts_matrix=h_opts_matrix+h_opt_Q(:,:,iiii)*10^(iiii-1);
    end
    P2P=[[1:(size_z_1*size_z_2)]',h_opts_matrix(:)];
    P2P=P2P(randperm((size_z_1*size_z_2)),:);
    P2P=uint32(sortrows(P2P,2));     %% sort according to lexicographic order
    clear h_opts_matrix
    h_opt_Q=uint8(h_opt_Q);

    y_hat_RW = function_SADCT_RW_wiener (h1, h_opt_Q, hadper, TrianRotu, conj_fftmatrix, tran_fftmatrix_fast, small_size, RWsmallabs2, zRW, P2P, sigma, T, h_max, y_hat_RI, DCTwieCOEF,max_overlapW); %%%% Pointwise SA-DCT RWI-Wiener function
    clear P2P
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%  End of Regularized Wiener   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    RW_toc=toc;
    Err_RW=function_Errors(y,y_hat_RW,z);  %% computes error criteria
    %%% PRINT RESULTS TO SCREEN
    disp(sprintf(repmat('\b',[1 numel(tab_content)+size(tab_content,1)+numel(tab_title)+5+20]))),
    disp(['RW completed in ',num2str(RW_toc-RI_toc),' seconds.   Total time: ',num2str(RW_toc),' seconds.'])
    disp(' ');
    disp(tab_title);
    disp([Err_labels,repmat(' ',[size(Err_RI,1) 1]),num2str(Err_RI,number_of_digits),repmat(' ',[size(Err_RI,1) 2]),num2str(Err_RW,number_of_digits)]);
    disp(' ')
    figure, imshow(y_hat_RW);title('RW SA-DCT estimate')
else
    disp(' ')
    disp('(!!!) RW skipped (!!!)                 ( do_wiener=0 )')
    disp(' ')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%     FILTERING ENDS      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%                     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%     END OF CODE     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%                     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%
%   RESULTS OBTAINED WITH THE CURRENT SOFTWARE   (as a reference)
%
%
% -------------------------------------------------------------------------------------
%  Pointwise Shape-Adaptive DCT Deblurring (Deconvolution)   -   Public demo release
% -------------------------------------------------------------------------------------
%
% Performing Experiment #1 ...
% Estimated noise sigma = 0.0021852   (true = 0.0021766,  misestimation = 0.39834%)
%
% RI completed in 6.9014 seconds.
% RW completed in 5.2601 seconds.   Total time: 12.1616 seconds.
%
%           RI        RW
% ISNR:  7.970072   8.57327
% SNR:   23.15578  23.75898
% PSNR:  28.73962  29.34282
% MSE:   86.91975  75.64821
% RMSE:  9.323076  8.697598
% MAE:   5.262208  4.941503
% MAX:   109.4001  104.7265
%
%
% -------------------------------------------------------------------------------------
%  Pointwise Shape-Adaptive DCT Deblurring (Deconvolution)   -   Public demo release
% -------------------------------------------------------------------------------------
%
% Performing Experiment #2 ...
% Estimated noise sigma = 0.0055542   (true = 0.0055459,  misestimation = 0.1496%)
%
% RI completed in 6.6783 seconds.
% RW completed in 5.2887 seconds.   Total time: 11.9669 seconds.
%
%           RI        RW
% ISNR:   7.51802  8.245852
% SNR:   24.16223  24.89006
% PSNR:  29.74607   30.4739
% MSE:   68.94038  58.30294
% RMSE:  8.303034  7.635636
% MAE:   4.702861  4.378378
% MAX:   83.68174  78.72955
%
%
% -------------------------------------------------------------------------------------
%  Pointwise Shape-Adaptive DCT Deblurring (Deconvolution)   -   Public demo release
% -------------------------------------------------------------------------------------
%
% Performing Experiment #3 ...
% Estimated noise sigma = 0.011013   (true = 0.011092,  misestimation = -0.71501%)
%
% RI completed in 6.5758 seconds.
% RW completed in 5.1538 seconds.   Total time: 11.7296 seconds.
%
%           RI        RW
% ISNR:  5.574842   6.34338
% SNR:   22.15198  22.92051
% PSNR:  27.73582  28.50435
% MSE:   109.5213   91.7582
% RMSE:  10.46524   9.57905
% MAE:   5.755498  5.307411
% MAX:      108.2  98.41741
%
%
% -------------------------------------------------------------------------------------
%  Pointwise Shape-Adaptive DCT Deblurring (Deconvolution)   -   Public demo release
% -------------------------------------------------------------------------------------
%
% Performing Experiment #4 ...
% Estimated noise sigma = 0.027376   (true = 0.027451,  misestimation = -0.27324%)
%
% RI completed in 27.3514 seconds.
% RW completed in 22.7637 seconds.   Total time: 50.1151 seconds.
%
%           RI        RW
% ISNR:  3.551216  4.518338
% SNR:   26.67928   27.6464
% PSNR:  32.36066  33.32778
% MSE:    37.7585  30.22057
% RMSE:  6.144795  5.497323
% MAE:   4.227383  3.739437
% MAX:   57.43433  58.35006
%