start_spams

clear
clc

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Prepare raw data
% % % % % % % % % % % % % % % % % % % % % % % % % % %
RawInpLoad = load('15814m_ltdbECG_1h.mat');
RawInpLoad = RawInpLoad.val;
n_dl = 128;
epochs = floor(length(RawInpLoad) / n_dl);    % 4517
RawInpLoad = RawInpLoad(1:n_dl * epochs);

RawInp = RawInpLoad(1:n_dl*epochs);
RawInp = reshape(RawInp , n_dl, epochs);
crossValidFactor = 0.7;

TrainInp = RawInp(:, 1:floor(epochs*crossValidFactor));
TrainInp = TrainInp - repmat(mean(TrainInp),[size(TrainInp,1),1]);
TrainInp = TrainInp ./ repmat(sqrt(sum(TrainInp.^2)),[size(TrainInp,1),1]);

TestInp = RawInp(:, (size(TrainInp,2)+1):epochs);
TestInp = TestInp - repmat(mean(TestInp),[size(TestInp,1),1]);
TestInp = TestInp ./ repmat(sqrt(sum(TestInp.^2)),[size(TestInp,1),1]);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Setting parameters for training
% % % % % % % % % % % % % % % % % % % % % % % % % % %

param.K = 1024;  
param.lambda = 0.1;            % sparsity constraint 
param.numThreads = -1; 
param.batchsize = 50;
param.verbose = false;
param.batchsize = 50;
param.iter = 50; 
param.clean = false;
param.iter_updateD = 1;
% param.verbose = true;


param_cs.lambda = 0.1;
param_cs.mode = 1;
param_cs.numThreads = -1;


% % % % % % % % % % % % % % % % % % % % % % % % % % %
% DL & CS
% % % % % % % % % % % % % % % % % % % % % % % % % % %

m_dl = floor(n_dl / 5);
phi_dl = randn(m_dl,n_dl);
reconSig = cell(1,50);

alpha = cell(1,50);
R1 = cell(1,50);
R2 = cell(1,50);
R3 = cell(1,50);

samplesTrain = size(TrainInp,2);
samplesTest  = size(TestInp,2);

rsnr_dl = zeros(1,length(1:floor(samplesTrain /50)));
res_dl = zeros(1,length(1:floor(samplesTrain /50)));
sparsity_dl = zeros(1,length(1:floor(samplesTrain /50)));
prd_dl = zeros(1,length(1:floor(samplesTrain /50)));


for i = 50 : floor(samplesTrain / 50)      % adjust iter
    res = 0;
    x2 = 0;
    spar = 0;
    prd = 0;
    
    y_dl = [];
    xs_dl = [];
    x0_dl = [];
    xhat_dl = [];
    
    param.iter = i;
    epochesD = floor(i * param.batchsize);
    X = TrainInp(:,1:epochesD);
    D = mexTrainDL(X,param);
    alpha{i} = mexLasso(X,D,param);
    R1{i} = mean(0.5*sum((X-D*alpha{i}).^2) + param.lambda*sum(abs(alpha{i})));
    R2{i} = mean(0.5*sum(X-D*alpha{i}).^2);
    R3{i} = mean(param.lambda*sum(abs(alpha{i})));
    fprintf('Objective function for i=%d is %f\n', i, R1{i});
    
    psi_dl = D;
    A_dl = phi_dl * psi_dl;
    
    for ep = 1:samplesTest
        y_dl = phi_dl * TestInp(:,ep);
        x0_dl = pinv(A_dl) * y_dl; 
        xs_dl = l1eq_pd(x0_dl, A_dl, [], y_dl, 5e-7, 50); 
%         xs_dl = mexLasso(y_dl,A_dl,param_cs);
        xhat_dl = psi_dl * xs_dl;
        
        subplot(211)
        plot(TestInp(:,ep));
        subplot(212)
        plot(xhat_dl);

        
		reconSig{i}(:,ep) = {xhat_dl};
        res = res + sum(norm(TestInp(:,ep) - xhat_dl).^2);
        x2 = x2 + sum(TestInp(:,ep).^2);
        spar = spar + length(find(abs(xs_dl)>0.001) );     
    end
    rsnr_dl(i) = 20 * log10(sqrt(x2 / res));
    cr_dl = n_dl / m_dl;
    sparsity_dl(i) = 1 - spar / samplesTest / length(xs_dl);
    prd_dl(i) = sqrt(res / x2);
end


% subplot(211)
% plot(TestInp(:,ep));
% subplot(212)
% plot(xhat_dl);

% 
filename = sprintf('./Results/reconProcess_batchsize%d_lambda%d.mat', param.batchsize, param.lambda);
save(filename)

%%
delay = 0.1;
writerObj  = VideoWriter('./Results/reconstruction.avi');
writerObj.FrameRate = 5; 
open(writerObj);
fig = figure('units','normalized','outerposition',[0 0 1 1]);
plot(TestInp(:,1));
axis([1 n_dl -0.3 0.2]);
hold on

for i = 1 : floor(samplesTrain / 50) 
    reconSigMat = cell2mat(reconSig{i}(:,20));
    
	h = plot(1:n_dl,reconSigMat);
    axis([1 n_dl -0.3 0.2]);
% 	hold on
	frame = getframe(fig);
	writeVideo(writerObj,frame);
    pause(delay);
    delete(h);
end

close(writerObj);


