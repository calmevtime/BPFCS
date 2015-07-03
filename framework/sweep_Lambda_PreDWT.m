% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Test sparsity vs time vs M
% Pre-processing data through 
% % % % % % % % % % % % % % % % % % % % % % % % % % %

% start_spams

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

batchsize = 50;
atoms = 512;

RawInp = RawInpLoad(1:n_dl*epochs);
RawInp = reshape(RawInp , n_dl, epochs);
crossValidFactor = 0.7;

indexD = randperm(atoms);
initD = RawInp(:, indexD);
wt = haarmtx(n_dl);
initD = wt * initD;
initD = initD - repmat(mean(initD),[size(initD,1),1]);
initD = initD ./ repmat(sqrt(sum(initD.^2)),[size(initD,1),1]);

RawInp = RawInp(:,atoms+1:end);
epochs = epochs - atoms;

TrainInp = RawInp(:, 1 : floor(epochs*crossValidFactor));
TrainInpDWT = wt * TrainInp;
TrainInpDWT = TrainInpDWT - repmat(mean(TrainInpDWT),[size(TrainInpDWT,1),1]);
TrainInpDWT = TrainInpDWT ./ repmat(sqrt(sum(TrainInpDWT.^2)),[size(TrainInpDWT,1),1]);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Compressive sensing
% % % % % % % % % % % % % % % % % % % % % % % % % % %

samplesTrain = size(TrainInpDWT,2);

sweepParam = [1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

objFun = zeros(length(sweepParam),length(1:floor(samplesTrain / batchsize)));
normErr = zeros(length(sweepParam),length(1:floor(samplesTrain / batchsize)));
sparCoef = zeros(length(sweepParam),length(1:floor(samplesTrain / batchsize)));

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
    parfor k = 1 : floor(samplesTrain / batchsize)      % adjust iter
        param = struct;
        param.iter = k;
        param.batchsize = batchsize;
        param.K = atoms;
        param.lambda = sweepParam(i);
        param.numThreads = -1; 
        param.verbose = false;
        param.iter_updateD = 1;
        param.D = initD;
        
        epochesD = floor(k * param.batchsize);
        X = TrainInpDWT(:,1:epochesD);
        [D,~,~] = mexTrainDL(X,param);
        
        coef = mexLasso(X,D,param);
       
        objFun(i,k) = mean(0.5*sum((X-D*coef).^2) + param.lambda*sum(abs(coef)));
        normErr(i,k) = mean(0.5*sum((X-D*coef).^2));
        sparCoef(i,k) = 1 - length(find((coef))) / length(coef(:));
        
        disp(sprintf('Iteration (%d, %d) preDWT: objective function is %f', i, k, objFun(i,k)));
        disp(sprintf('Iteration (%d, %d) preDWT: L-2 norm of error is %f\n', i, k, normErr(i,k)));
%         disp(sprintf('Iteration (%d, %d) without pre: sparsity of coeff is %f\n', i, k, sparCoef(i,k)));
    end
 end
 
% maxObjFunc = max(objFun');
% maxNormErr = max(normErr');

% delete(poolobj)

filename = sprintf('../Results/preDWT/sweeplambda_PreDWT_batchsize%d.mat', batchsize);
save(filename,'-v7.3')

