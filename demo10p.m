% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Test sparsity vs time vs M
% Pre-processing data through wavelet transform
% % % % % % % % % % % % % % % % % % % % % % % % % % %

clear
clc

mdivision = 10;
division = 30;

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
param.batchsize = 10;
param.verbose = false;
param.iter = 10; 

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Prepare training and testing data
% % % % % % % % % % % % % % % % % % % % % % % % % % %

RawInp = RawInpLoad(1:n_dl*epochs);
RawInp = reshape(RawInp , n_dl, epochs);
crossValidFactor = 0.2;

TrainInp = RawInp(:, 1:floor(epochs*crossValidFactor));
TestInp = RawInp(:, (size(TrainInp,2)+1):epochs);

qmf = MakeONFilter('Daubechies',20);
XI=eye(n_dl);
psi_dwt=zeros(n_dl);

for i=1:n_dl
   wt(:,i) = FWT_PO(XI(:,i),1,qmf);
end

for i=1:n_dl
  psi_dwt(:,i)=IWT_PO(XI(:,i),1,qmf);
end

TrainInpFWT = wt * TrainInp;
TestInpFWT = wt * TestInp;
% figure
% subplot(2,1,1)
% plot(TrainInp(:,1));
% subplot(2,1,2)
% plot(TrainInpFWT(:,1));
spar = 1- length(find(abs(TrainInpFWT)>15) ) / size(TrainInpFWT,1) / size(TrainInpFWT,2);
disp(sprintf('Sparsity after wavelet transformation is : %0.2f', spar));

TrainInp = TrainInpFWT - repmat(mean(TrainInpFWT),[size(TrainInpFWT,1),1]);
TrainInp = TrainInpFWT ./ repmat(sqrt(sum(TrainInpFWT.^2)),[size(TrainInpFWT,1),1]);

TestInp = TestInpFWT - repmat(mean(TestInpFWT),[size(TestInpFWT,1),1]);
TestInp = TestInpFWT ./ repmat(sqrt(sum(TestInpFWT.^2)),[size(TestInpFWT,1),1]);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Compressive sensing
% % % % % % % % % % % % % % % % % % % % % % % % % % %

samplesTrain = size(TrainInp,2);
samplesTest  = size(TestInp,2);

rsnr_dl = zeros(mdivision,length(1:floor(samplesTrain /50)));
res_dl = zeros(mdivision,length(1:floor(samplesTrain /50)));
sparsity_dl = zeros(mdivision,length(1:floor(samplesTrain /50)));
basis = cell(mdivision,length(1:floor(samplesTrain /50)));

%%

parpool('local',24);
poolobj = gcp('nocreate'); % If no pool, do not create new one.
if isempty(poolobj)
    poolsize = 0;
else
    poolsize = poolobj.NumWorkers;
end

%%

for i = 1:mdivision 
    m_dl = floor(i * n_dl / mdivision)
    phi_dl = randn(m_dl,n_dl);
        
     parfor j = 1 : floor(samplesTrain / 50)      % adjust iter
        param = struct;
        param.iter = j;
        param.batchsize = 50;
        param.K = 512;
        param.lambda = 0.15;
        param.numThreads = -1; 
        param.verbose = false;

        rsnr = 0;
        res = 0;
        spar = 0;
        xs_dl = [];
        x0_dl = [];
        xhat_dl = [];
        D = [];
        
        epochesD = floor(j * param.batchsize);
        D = mexTrainDL(TrainInp(:,1:epochesD),param);

        basis(i, j) = {D};

        psi_dl = D;
        A_dl = phi_dl * psi_dl;

        for ep = 1:samplesTest
            y_dl = phi_dl * TestInp(:,ep);
            x0_dl = pinv(A_dl) * y_dl; 
            xs_dl = l1eq_pd(x0_dl, A_dl, [], y_dl, 1e-5); 
            xhat_dl = psi_dl * xs_dl;
            rsnr = rsnr + 20 * (log10 (norm(TestInp(:,ep),2) / norm(TestInp(:,ep) - xhat_dl,2)));   
            res = res + norm(TestInp(:,ep) - xhat_dl,2);
            
            spar = spar + length(find(abs(xs_dl)>0.001) );
        end
        rsnr_dl(i,j) = rsnr / samplesTest;
        res_dl(i,j) = res / samplesTest;
        sparsity_dl(i,j) = 1 - spar / samplesTest / length(xs_dl);
    end
end

delete(poolobj)

filename = sprintf('./Results/Pre_m%d_batchsize%d.mat', mdivision, 50);
save(filename)

%% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Plot results
% % % % % % % % % % % % % % % % % % % % % % % % % % %

cc = jet(20);
str = cell(1,19);

%% figure
j = 1 : floor(samplesTrain / 50) 
subplot(3,1,1)
for i = 1 : 19
    plot(floor(j * 50),rsnr_dl(i,:),'Color',cc(i,:) ) ;
    str{i}=['m=',num2str(floor(i * n_dl / mdivision))];
    hold on
end
legend(str)
xlabel('Iterations');
ylabel('RSNR(dB)');

subplot(3,1,2)
for i = 1 : 19
    plot(floor(j * 50),res_dl(i,:),'Color',cc(i,:) );
    str{i}=['m=',num2str(floor(i * n_dl / mdivision))];
    hold on
end
% legend(str)
xlabel('Iterations');
ylabel('MSE');

subplot(3,1,3)
for i = 1 : 19
    plot(floor(j * 50),sparsity_dl(i,:),'Color',cc(i,:) );
    str{i}=['m=',num2str(floor(i * n_dl / mdivision))];
    hold on
end
% legend(str)
xlabel('Iterations');
ylabel('Sparsity');

% 
% %%
% 
% % load ./Results/m20_batchsize50.mat
% % load ./Results/DWT40.mat
% 
% figure
% epV = 50 * (1 : floor(samplesTrain / 50)); 
% mV = floor(n_dl/mdivision: n_dl/mdivision: n_dl);
% 
% [M,I] = min(abs(rsnr_dl(1:19,:)'-20) );
% i = [19, 12, 9, 8, 7, 6, 4];
% plot(epV(I(i)),mV(i),'Color',cc(1,:));
% hold on
% 
% [M,I] = min(abs(rsnr_dl(1:19,:)'-23) );
% i = [19, 14, 12, 11, 10, 9];
% plot(epV(I(i)),mV(i),'Color',cc(3,:));
% hold on
% 
% 
% [M,I] = min(abs(rsnr_dl(1:19,:)'-25) );
% i = [19, 16, 15, 14, 13, 11];
% plot(epV(I(i)),mV(i),'Color',cc(5,:));
% hold on
% 
% [M,I] = min(abs(rsnr_dl(1:19,:)'-27) );
% i = [17, 16, 15, 14, 13];
% plot(epV(I(i)),mV(i),'Color',cc(7,:));
% hold on
% 
% 
% %%
% 
% [M,I] = min(abs(rsnr_dl_dwt - 20) );
% plot([200,3200], [m_dl_dwt(I),m_dl_dwt(I)],'Color',cc(11,:));
% hold on
% 
% [M,I] = min(abs(rsnr_dl_dwt - 23) );
% plot([200,3200], [m_dl_dwt(I),m_dl_dwt(I)],'Color',cc(13,:));
% hold on
% 
% [M,I] = min(abs(rsnr_dl_dwt - 25) );
% plot([200,3200], [m_dl_dwt(I),m_dl_dwt(I)],'Color',cc(15,:));
% hold on
% 
% [M,I] = min(abs(rsnr_dl_dwt - 27) );
% plot([200,3200], [m_dl_dwt(I),m_dl_dwt(I)],'Color',cc(17,:));
% hold on
% 
% str = {'RSNR\_DL = 20dB','RSNR\_DL = 23dB','RSNR\_DL = 25dB', 'RSNR\_DL = 27dB', ...
%        'RSNR\_DWT = 20dB','RSNR\_DWT = 23dB','RSNR\_DWT = 25dB', 'RSNR\_DWT = 27dB'};
% 
% legend(str);
% xlabel('Iterations');
% ylabel('m');
% axis([0 3300 0 130])
% 
% %%
% figure
% 
% delay = 0.5;
% for j = 1:floor(samplesTrain/50)
%     clf;
%     plot(basis{10,j}(:,1));
% %     hold on;
%     pause(delay);
% end
