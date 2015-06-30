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
epochs = floor(length(RawInpLoad) / n_dl);    % 4517
RawInpLoad = RawInpLoad(1:n_dl * epochs);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Prepare training and testing data
% % % % % % % % % % % % % % % % % % % % % % % % % % %

RawInp = RawInpLoad(1:n_dl*epochs);
RawInp = reshape(RawInp , n_dl, epochs);
crossValidFactor = 0.7;

TrainInp = RawInp(:, 1:floor(epochs*crossValidFactor));

wt = dctmtx(n_dl);
TrainInpDCT = wt * TrainInp;

TrainInpDCT = TrainInpDCT - repmat(mean(TrainInpDCT),[size(TrainInpDCT,1),1]);
TrainInpDCT = TrainInpDCT ./ repmat(sqrt(sum(TrainInpDCT.^2)),[size(TrainInpDCT,1),1]);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Compressive sensing
% % % % % % % % % % % % % % % % % % % % % % % % % % %

samplesTrain = size(TrainInpDCT,2);

sweepParam = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

objFun = zeros(length(sweepParam),length(1:floor(samplesTrain / 50)));
normErr = zeros(length(sweepParam),length(1:floor(samplesTrain / 50)));
sparCoef = zeros(length(sweepParam),length(1:floor(samplesTrain / 50)));

%%

% poolobj = gcp('nocreate'); % If no pool, do not create new one.
% if isempty(poolobj)
%     poolsize = 0;
%     parpool('local',12);
% else
%     poolsize = poolobj.NumWorkers;
% end

%%
 for i = 1 : length(sweepParam)
    parfor k = 1 : floor(samplesTrain / 50)      % adjust iter
        param = struct;
        param.iter = k;
        param.batchsize = 50;
        param.K = 512;
        param.lambda = 0.2;
        param.numThreads = -1; 
        param.verbose = false;
        param.iter_updateD = 1;

        epochesD = floor(k * param.batchsize);
        X = TrainInpDCT(:,1:epochesD);
        [D,~,~] = mexTrainDL(X,param);        
        
        coef = mexLasso(X,D,param);
        
        objFun(i,k) = mean(0.5*sum((X-D*coef).^2) + param.lambda*sum(abs(coef)));
        normErr(i,k) = mean(0.5*sum((X-D*coef).^2));
        sparCoef(i,k) = 1 - length(find((coef))) / length(coef(:));
        
        disp(sprintf('Iteration (%d, %d) preDCT: objective function is %f', i, k, objFun(i,k)));
        disp(sprintf('Iteration (%d, %d) preDCT: L-2 norm of error is %f\n', i, k, normErr(i,k)));
%         disp(sprintf('Iteration (%d, %d) without pre: sparsity of coeff is %f\n', i, k, sparCoef(i,k)));
    end
 end
 
maxObjFunc = max(objFun');
maxNormErr = max(normErr');

% delete(poolobj)

filename = sprintf('./Results/sweeplambda_PreDCT_batchsize%d.mat', 50);
save(filename,'-v7.3')

