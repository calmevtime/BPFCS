% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Test sparsity vs time vs M
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
n_dl = 102;
epochs = floor(length(RawInpLoad) / n_dl);    % 4517
RawInpLoad = RawInpLoad(1:n_dl * epochs);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Setting parameters for training
% % % % % % % % % % % % % % % % % % % % % % % % % % %

dimMin = 51;
dimMax = 2048;


% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Prepare training and testing data
% % % % % % % % % % % % % % % % % % % % % % % % % % %

RawInp = RawInpLoad(1:n_dl*epochs);
RawInp = reshape(RawInp , n_dl, epochs);
crossValidFactor = 0.15;
TrainInp = RawInp(:, 1:floor(epochs*crossValidFactor));
TrainInp = TrainInp - repmat(mean(TrainInp),[size(TrainInp,1),1]);
TrainInp = TrainInp ./ repmat(sqrt(sum(TrainInp.^2)),[size(TrainInp,1),1]);

% TestInp = RawInp(:, (size(TrainInp,2)+1):epochs);
TestInp = RawInp(:, (size(TrainInp,2)+1):size(TrainInp,2)*2);
TestInp = TestInp - repmat(mean(TestInp),[size(TestInp,1),1]);
TestInp = TestInp ./ repmat(sqrt(sum(TestInp.^2)),[size(TestInp,1),1]);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Compressive sensing
% % % % % % % % % % % % % % % % % % % % % % % % % % %

rsnr_dl = zeros(mdivision,10);
res_dl = zeros(mdivision,10);
sparsity_dl = zeros(mdivision,10);
basis = cell(mdivision,10);

samplesTrain = size(TrainInp,2);
samplesTest  = size(TestInp,2);

% %%
% parpool('local',20);
% poolobj = gcp('nocreate'); % If no pool, do not create new one.
% if isempty(poolobj)
%     poolsize = 0;
% else
%     poolsize = poolobj.NumWorkers;
% end

%%
for i = 1:mdivision 
    m_dl = floor(i * n_dl / mdivision)
    phi_dl = randn(m_dl,n_dl);
        
%     parfor j = 1 : floor(samplesTrain / 10)      % adjust iter
      for j = 1:10
        param = struct;
        param.iter = 10;
        param.batchsize = 10 + 5 * (j-1);
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
        D = mexTrainDL(TrainInp,param);

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

filename = sprintf('./Results/m%d_iteration%d_batchsize%d.mat', mdivision, floor(samplesTrain), 1);
save(filename)

%% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Plot results
% % % % % % % % % % % % % % % % % % % % % % % % % % %

cc = jet(10);
str = cell(1,9);

figure
subplot(3,1,1)
for i = 1 : 9
    plot(1:10,rsnr_dl(i,:),'Color',cc(i,:) ) ;
    str{i}=['m=',num2str(floor(i * n_dl / mdivision))];
    hold on
end
legend(str)
xlabel('Iterations');
ylabel('RSNR(dB)');

subplot(3,1,2)
for i = 1 : 9
    plot(1:10,res_dl(i,:),'Color',cc(i,:) );
    str{i}=['m=',num2str(floor(i * n_dl / mdivision))];
    hold on
end
% legend(str)
xlabel('Iterations');
ylabel('MSE');

subplot(3,1,3)
for i = 1 : 9
    plot(1:10,sparsity_dl(i,:),'Color',cc(i,:) );
    str{i}=['m=',num2str(floor(i * n_dl / mdivision))];
    hold on
end
% legend(str)
xlabel('Iterations');
ylabel('Sparsity');

