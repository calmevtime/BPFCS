% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Test sparsity vs time vs M
% Under fixed wavelet dictionary
% % % % % % % % % % % % % % % % % % % % % % % % % % %

clear
clc

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Prepare raw data
% % % % % % % % % % % % % % % % % % % % % % % % % % %

RawInpLoad = load('15814m_ltdbECG_1h.mat');
RawInpLoad = RawInpLoad.val;
n_dl = 128;
m_dl = 51;
epochs = floor(length(RawInpLoad) / n_dl);    % 4517
RawInpLoad = RawInpLoad(1:n_dl * epochs);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Setting parameters for training
% % % % % % % % % % % % % % % % % % % % % % % % % % %

param.K = 512;
dimMin = 51;
dimMax = 2048;
param.lambda = 0.15;            % sparsity constraint 
param.numThreads = -1; 
param.batchsize = 400;
param.verbose = false;
param.iter = 10; 

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Prepare training and testing data
% % % % % % % % % % % % % % % % % % % % % % % % % % %

RawInp = RawInpLoad(1:n_dl*epochs);
RawInp = reshape(RawInp , n_dl, epochs);
crossValidFactor = 0.14;
TrainInp = RawInp(:, 1:floor(epochs*crossValidFactor));
TrainInp = TrainInp - repmat(mean(TrainInp),[size(TrainInp,1),1]);
TrainInp = TrainInp ./ repmat(sqrt(sum(TrainInp.^2)),[size(TrainInp,1),1]);

TestInp = RawInp(:, (size(TrainInp,2)+1):epochs);
TestInp = TestInp - repmat(mean(TestInp),[size(TestInp,1),1]);
TestInp = TestInp ./ repmat(sqrt(sum(TestInp.^2)),[size(TestInp,1),1]);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Compressive sensing
% % % % % % % % % % % % % % % % % % % % % % % % % % %

rsnr_dl = zeros(1,20);
res_dl = zeros(1,20);
sparsity_dl = zeros(1,20);
m_dl_cnt = 1;
n_dl_cnt = 1;
rsnr = 0;
res = 0;
spar = 0;

qmf = MakeONFilter('Daubechies',20);
XI=eye(n_dl);
psi_dwt=zeros(n_dl);

for i=1:n_dl
  psi_dwt(:,i)=IWT_PO(XI(:,i),1,qmf);
end

for m_dl = floor(n_dl/20: n_dl/20: n_dl)
    phi_dwt = randn(m_dl,n_dl);

    A_dl = phi_dwt * psi_dwt;

    for ep = 1:size(TestInp,2)
        y_dl = phi_dwt * TestInp(:,ep);
        x0_dl = pinv(A_dl) * y_dl; 
        xs_dl = l1eq_pd(x0_dl, A_dl, [], y_dl); 
        xhat_dl = psi_dwt * xs_dl;
        rsnr = rsnr + 20 * (log10 (norm(TestInp(:,ep),2) / norm(TestInp(:,ep) - xhat_dl,2)));   
        res = res + norm(TestInp(:,ep) - xhat_dl,2);
        spar = spar + length(find(abs(xs_dl)>0.001) );
    end
    
    rsnr_dl(m_dl_cnt) = rsnr / size(TestInp,2);
    res_dl(m_dl_cnt) = res / size(TestInp,2);
    sparsity_dl(m_dl_cnt) = 1 - spar / size(TestInp,2) / length(xs_dl);
    m_dl_cnt = m_dl_cnt + 1;
    rsnr = 0;
    res = 0;
    spar = 0;      
end

save './Results/SPARSITYvsTIMEvsM_DWT.mat'

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Plot results
% % % % % % % % % % % % % % % % % % % % % % % % % % %

figure
subplot(3,1,1)

plot(floor(n_dl/20: n_dl/20: n_dl)./n_dl, rsnr_dl ) ;
xlabel('m/n');
ylabel('RSNR(dB)');

[M,I] = min(abs(rsnr_dl - 20));

plot(floor(n_dl/20: n_dl/20: n_dl)./n_dl, res_dl );
xlabel('m/n');
ylabel('MSE');


plot(floor(n_dl/20: n_dl/20: n_dl)./n_dl, sparsity_dl);
xlabel('m/n');
ylabel('Sparsity');


figure
[M,I] = min(abs(rsnr_dl(1:19,:)'-15) );
a = [epV(I(17)),mV(17); epV(I(11)),mV(11); epV(I(5)),mV(5); epV(I(4)),mV(4); epV(I(3)),mV(3); epV(I(2)),mV(2)];
plot(a(:,1),a(:,2),'r')
hold on

% I(find(M>1,5)) = [];

[M,I] = min(abs(rsnr_dl(1:19,:)'-20) );

epV = floor(size(TrainInp,2)/100 : size(TrainInp,2)/100 : size(TrainInp,2) );
mV = floor(n_dl/20: n_dl/20: n_dl);
a = [epV(I(17)),mV(17); epV(I(12)),mV(12); epV(I(9)),mV(9); epV(I(8)),mV(8); epV(I(6)),mV(6); epV(I(5)),mV(5); epV(I(4)),mV(4)];
plot(a(:,1),a(:,2),'g')
hold on

[M,I] = min(abs(rsnr_dl(1:19,:)'-25) );
b = [epV(I(17)),mV(17); epV(I(15)),mV(15); epV(I(14)),mV(14); epV(I(13)),mV(13); epV(I(12)),mV(12); epV(I(11)),mV(11)];
plot(b(:,1),b(:,2),'b')

str = {'RSNR = 15dB','RSNR = 20dB','RSNR = 25dB)'};
legend(str);
xlabel('Iterations');
ylabel('m');
axis([0 1800 0 10])

