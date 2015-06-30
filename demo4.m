% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Only remove DC, don't normalize raw data
% % % % % % % % % % % % % % % % % % % % % % % % % % %

clear
clc

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Prepare raw data
% % % % % % % % % % % % % % % % % % % % % % % % % % %

RawInpLoad = load('15814m_ltdbECG_1h.mat');
RawInpLoad = RawInpLoad.val;
n_dl_max = 512;
epochs = floor(length(RawInpLoad) / n_dl_max);    % 3600
RawInpLoad = RawInpLoad(1:n_dl_max * epochs);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Setting parameters for training
% % % % % % % % % % % % % % % % % % % % % % % % % % %

param.K = 512;  
param.lambda = 0.15;            % sparsity constraint 
param.numThreads = -1; 
param.batchsize = 400;
param.verbose = false;
param.iter = 10; 

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Compressive sensing
% % % % % % % % % % % % % % % % % % % % % % % % % % %

n_dl = 102;

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Prepare training and testing data
% % % % % % % % % % % % % % % % % % % % % % % % % % %

RawInp = RawInpLoad(1:n_dl*epochs);
RawInp = reshape(RawInp , n_dl, epochs);
crossValidFactor = 0.7;
TrainInp = RawInp(:, 1:floor(epochs*crossValidFactor));
TrainInp = TrainInp - repmat(mean(TrainInp),[size(TrainInp,1),1]);
% TrainInp = TrainInp ./ repmat(sqrt(sum(TrainInp.^2)),[size(TrainInp,1),1]);

TestInp = RawInp(:, (size(TrainInp,2)+1):epochs);
TestInp = TestInp - repmat(mean(TestInp),[size(TestInp,1),1]);
% TestInp = TestInp ./ repmat(sqrt(sum(TestInp.^2)),[size(TestInp,1),1]);

disp('Starting to  train the dictionary');
D = mexTrainDL(TrainInp,param);
alpha = mexLasso(TrainInp,D,param);
MSE = mean(0.5*sum((TrainInp-D*alpha).^2));
fprintf('objective function: %f\n',MSE);

psi_dl = D;
rsnr = 0;

m_dl = floor(n_dl/2);
phi_dl = randn(m_dl,n_dl);
A_dl = phi_dl * psi_dl;
xhat_dl_mat = zeros(size(TestInp,1), size(TestInp,2));

for ep = 1:size(TestInp,2)
    y_dl = phi_dl * TestInp(:,ep);
    x0_dl = pinv(A_dl) * y_dl; 
    xs_dl = l1eq_pd(x0_dl, A_dl, [], y_dl); 
    xhat_dl = psi_dl * xs_dl;
    xhat_dl_mat(:,ep) = xhat_dl;
    rsnr = rsnr + 20 * (log10 (norm(TestInp(:,ep),2) / norm(TestInp(:,ep) - xhat_dl,2)));   

    if (ep == 100)
        figure;
        subplot(2,1,1);
        plot(TestInp(:,ep));
        subplot(2,1,2)
        plot(xhat_dl);
    end

end
rsnr_dl = rsnr / size(TestInp,2);
rsnr = 0;
xorig = TestInp(:);
xhat_dl_vec = xhat_dl_mat(:);

subplot(2,1,1);
plot(xorig);
title('Original signal');
xlabel('Time interval');
ylabel('Amplitude');

subplot(2,1,2);
plot(xhat_dl_vec);
title('Reconstructed signal');
xlabel('Time interval');
ylabel('Amplitude');