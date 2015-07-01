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

batchsize = 50;
atoms = 512;
lambda = 1e-2;

RawInp = RawInpLoad(1:n_dl*epochs);
RawInp = reshape(RawInp , n_dl, epochs);
crossValidFactor = 0.7;

indexD = randperm(atoms);
initD = RawInp(:, indexD);
initD = initD - repmat(mean(initD),[size(initD,1),1]);
initD = initD ./ repmat(sqrt(sum(initD.^2)),[size(initD,1),1]);

RawInp = RawInp(:,atoms+1:end);
epochs = epochs - atoms;

TrainInp = RawInp(:, 1 : floor(epochs*crossValidFactor));
TrainInp = TrainInp - repmat(mean(TrainInp),[size(TrainInp,1),1]);
TrainInp = TrainInp ./ repmat(sqrt(sum(TrainInp.^2)),[size(TrainInp,1),1]);

TestInp = RawInp(:, (size(TrainInp,2)+1):epochs);
TestInp = TestInp - repmat(mean(TestInp),[size(TestInp,1),1]);
TestInp = TestInp ./ repmat(sqrt(sum(TestInp.^2)),[size(TestInp,1),1]);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Compressive sensing
% % % % % % % % % % % % % % % % % % % % % % % % % % %

samplesTrain = size(TrainInp,2);

sweepParam = [128, 256, 512, 1024, 2048];

objFun = zeros(length(sweepParam),length(1:floor(samplesTrain / batchsize)));
normErr = zeros(length(sweepParam),length(1:floor(samplesTrain / batchsize)));
sparCoef = zeros(length(sweepParam),length(1:floor(samplesTrain / batchsize)));

%%
% poolobj = gcp('nocreate'); % If no pool, do not create new one.
% 
% if isempty(poolobj)
%     poolsize = 0;
%     parpool('local',8);
% else
%     poolsize = poolobj.NumWorkers;
% end

%%
 for i = 1 : length(sweepParam)
    parfor k = 1 : floor(samplesTrain / batchsize)      % adjust iter
        param = struct;
        param.iter = k;
        param.batchsize = batchsize;
        param.K = sweepParam(i);
        param.lambda = lambda;
        param.numThreads = -1; 
        param.verbose = false;
        param.iter_updateD = 1;
        param.D = initD;

        epochesD = floor(k * param.batchsize);
        X = TrainInp(:,1:epochesD);
        [D,~,~] = mexTrainDL(X,param);
        
        coef = mexLasso(X,D,param);
        
        objFun(i,k) = mean(0.5*sum((X-D*coef).^2) + param.lambda*sum(abs(coef)));
        normErr(i,k) = mean(0.5*sum((X-D*coef).^2));
        sparCoef(i,k) = 1 - length(find((coef))) / length(coef(:));
        
        disp(sprintf('Iteration (%d, %d) without pre: objective function is %f', i, k, objFun(i,k)));
        disp(sprintf('Iteration (%d, %d) without pre: L-2 norm of error is %f\n', i, k, normErr(i,k)));
%         disp(sprintf('Iteration (%d, %d) without pre: sparsity of coeff is %f\n', i, k, sparCoef(i,k)));
    end
 end
 
maxObjFunc = max(objFun');
maxNormErr = max(normErr');

% delete(poolobj)

filename = sprintf('./Results/sweep_K_withoutPre_batchsize%d_lambda%.2f.mat', batchsize, lambda);
save(filename,'-v7.3')

