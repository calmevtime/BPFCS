% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Test sparsity vs time vs M
% Pre-processing data through 
% % % % % % % % % % % % % % % % % % % % % % % % % % %

start_spams

clear
clc

mdivision = 20;

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
% Prepare training and testing data
% % % % % % % % % % % % % % % % % % % % % % % % % % %

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
% Compressive sensing
% % % % % % % % % % % % % % % % % % % % % % % % % % %

samplesTrain = size(TrainInp,2);
samplesTest  = size(TestInp,2);

rsnr_dl = zeros(mdivision,length(1:floor(samplesTrain / 50)));
cr_dl = zeros(mdivision,length(1:floor(samplesTrain / 50)));
prd_dl = zeros(mdivision,length(1:floor(samplesTrain / 50)));
sparsity_dl = zeros(mdivision,length(1:floor(samplesTrain / 50)));
basis = cell(mdivision,length(1:floor(samplesTrain / 50)));
R1 = cell(mdivision,length(1:floor(samplesTrain / 50)));
% R2 = cell(mdivision,length(1:floor(samplesTrain / 50)));
alpha = cell(mdivision,length(1:floor(samplesTrain / 50)));
reconSig = cell(mdivision,length(1:floor(samplesTrain / 50)));


%%

% poolobj = gcp('nocreate'); % If no pool, do not create new one.
% if isempty(poolobj)
%     poolsize = 0;
%     parpool('local',12);
% else
%     poolsize = poolobj.NumWorkers;
% end

%%

 for i = 1 : mdivision 
    m_dl = floor(i * n_dl / mdivision);
    phi_dl = randn(m_dl,n_dl);
%     phi_dl = phi_dl ./ repmat(sqrt(sum(phi_dl.^2)),[size(phi_dl,1),1]);
        
    for j = 40 : floor(samplesTrain / 50)      % adjust iter
        param = struct;
        param.iter = j;
        param.batchsize = 50;
        param.K = 512;
        param.lambda = 10;
        param.numThreads = -1; 
        param.verbose = false;
        param.iter_updateD = 1;
        
        res = 0;
        x2 = 0;
        spar = 0;
        
        y_dl = [];
        xs_dl = [];
        x0_dl = [];
        xhat_dl = [];
        D = [];
        
        epochesD = floor(j * param.batchsize);
        X = TrainInp(:,1:epochesD);
        D = mexTrainDL(X,param);
        
%         coef = mexLasso(X,D,param);
%         alpha{i,j} = coef;
%         R1{i,j} = mean(0.5*sum((X-D*coef).^2) + param.lambda*sum(abs(coef)));
%         R2{i,j} = 0.5*sum(X-D*coef).^2;
%         fprintf('Objective function for i=%d, j=%d is %f', i, j, R1{i,j});
        
        basis(i, j) = {D};

        psi_dl = D;
        A_dl = phi_dl * psi_dl;

        for ep = 1:samplesTest
            y_dl = phi_dl * TestInp(:,ep);
            x0_dl = pinv(A_dl) * y_dl; 
            xs_dl = l1eq_pd(x0_dl, A_dl, [], y_dl, 1e-6); 
            xhat_dl = psi_dl * xs_dl;
            
            reconSig{i,j}(:,ep) = {xhat_dl};
            res = res + sum(norm(TestInp(:,ep) - xhat_dl).^2);
            x2 = x2 + sum(TestInp(:,ep).^2);
            spar = spar + length(find(abs(xs_dl)>0.001) ); 
            
            subplot(211)
            plot(TestInp(:,ep));
            subplot(212)
            plot(xhat_dl);
        end
        rsnr_dl(i,j) = 20 * log10(sqrt(x2 / res));
        cr_dl(i,j) = n_dl / m_dl;
        sparsity_dl(i,j) = 1 - spar / samplesTest / length(xs_dl);
        prd_dl(i,j) = sqrt(res / x2);
    end
end

delete(poolobj)

filename = sprintf('./Results/m%d_batchsize%d_lambda%.2f.mat', mdivision, 50, 0.10);
save(filename)



%% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Plot reconstruction process
% % % % % % % % % % % % % % % % % % % % % % % % % % %

delay = 1;
epSel = 1000;
writerObj  = VideoWriter('./Results/reconstruction.avi');
writerObj.FrameRate = 5; 
open(writerObj);
fig = figure('units','normalized','outerposition',[0 0 1 1]);
plot(TestInp(:,epSel));
axis([1 n_dl -0.3 0.3]);
hold on

for i = 1 : floor(samplesTrain / 50) 
    reconSigMat = cell2mat(reconSig{5,i}(:,epSel));
    
	h = plot(1:n_dl,reconSigMat);
    axis([1 n_dl -0.3 0.3]);
% 	hold on
	frame = getframe(fig);
	writeVideo(writerObj,frame);
    pause(delay);
    delete(h);
end

plot(1:n_dl,reconSigMat);
close(writerObj);

%% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Plot results
% % % % % % % % % % % % % % % % % % % % % % % % % % %

load ./Results/m20_batchsize50_lambda0.10.mat

cc = jet(mdivision);
str = cell(1,mdivision-1);

figure
j = 1 : floor(samplesTrain / 50);
subplot(3,1,1)
for i = 1 : mdivision-1
    plot(floor(j * 50),rsnr_dl(i,:),'Color',cc(i,:) ) ;
    str{i}=['m=',num2str(floor(i * n_dl / mdivision))];
    hold on
end
legend(str)
xlabel('Iterations');
ylabel('RSNR(dB)');

subplot(3,1,2)
for i = 1 : mdivision-1
    plot(floor(j * 50),prd_dl(i,:),'Color',cc(i,:) );
    str{i}=['m=',num2str(floor(i * n_dl / mdivision))];
    hold on
end
% legend(str)
xlabel('Iterations');
ylabel('PRD');

subplot(3,1,3)
for i = 1 : mdivision-1
    plot(floor(j * 50),sparsity_dl(i,:),'Color',cc(i,:) );
    str{i}=['m=',num2str(floor(i * n_dl / mdivision))];
    hold on
end
% legend(str)
xlabel('Iterations');
ylabel('Sparsity');
% 

%%

figure
epV = 50 * (1 : floor(samplesTrain / 50)); 
mV = floor(n_dl/mdivision: n_dl/mdivision: n_dl);

[M,I] = min(abs(rsnr_dl(1:19,:)'-13) );
obj = {M',I'};
obj{1,2}(:,2) = obj{:,1};
i = [19, 17, 8, 5, 4, 3,2];
plot(epV(I(i)),mV(i),'Color',cc(1,:));
hold on

[M,I] = min(abs(rsnr_dl(1:19,:)'-20) );
obj = {M',I'};
obj{1,2}(:,2) = obj{:,1};
i = [19, 13, 12, 11, 10, 9, 8, 7, 6];
plot(epV(I(i)),mV(i),'Color',cc(3,:));
hold on

[M,I] = min(abs(rsnr_dl(1:19,:)'-22) );
obj = {M',I'};
obj{1,2}(:,2) = obj{:,1};
i = [16, 14, 13, 12, 11];
plot(epV(I(i)),mV(i),'Color',cc(5,:));
hold on

[M,I] = min(abs(rsnr_dl(1:19,:)'-24) );
obj = {M',I'};
obj{1,2}(:,2) = obj{:,1};
i = [18, 16, 15, 14, 13];
plot(epV(I(i)),mV(i),'Color',cc(7,:));
hold on

[M,I] = min(abs(rsnr_dl(1:19,:)'-26) );
obj = {M',I'};
obj{1,2}(:,2) = obj{:,1};
i = [19, 17, 16, 15];
plot(epV(I(i)),mV(i),'Color',cc(9,:));
hold on


%

load ./Results/m40_haar_basis_1e-6.mat

[M,I] = min(abs(rsnr_dl_haar - 20) );
plot([50,2400], [m_dl_haar(I),m_dl_haar(I)],'Color',cc(11,:));
% hold on
% 
% [M,I] = min(abs(rsnr_dl_haar - 22) );
% plot([200,3200], [m_dl_dwt(I),m_dl_dwt(I)],'Color',cc(13,:));
% hold on
% 
% [M,I] = min(abs(rsnr_dl_haar - 24) );
% plot([200,3200], [m_dl_dwt(I),m_dl_dwt(I)],'Color',cc(15,:));
% hold on
% 
% [M,I] = min(abs(rsnr_dl_haar - 26) );
% plot([200,3200], [m_dl_dwt(I),m_dl_dwt(I)],'Color',cc(17,:));
% hold on
% 
% str = {'RSNR\_DL = 20dB','RSNR\_DL = 22dB','RSNR\_DL = 24dB', 'RSNR\_DL = 26dB', ...
%        'RSNR\_DWT = 20dB','RSNR\_DWT = 22dB','RSNR\_DWT = 24dB', 'RSNR\_DWT = 26dB'};
   
str = {'RSNR\_DL = 13dB','RSNR\_DL = 20dB','RSNR\_DL = 22dB','RSNR\_DL = 24dB', 'RSNR\_DL = 26dB', ...
       'RSNR\_HAAR = 13dB'};

legend(str);
xlabel('Iterations');
ylabel('m');
% axis([0 3300 0 130])
% 
% %%

for i = 1 : 20   
    for j = 1 : 50
        psi_dl = basis{i,j};
        mtxSig = cell2mat(reconSig{i,j});
        mtxCoe{i,j} = psi_dl \ mtxSig;
    end
end
