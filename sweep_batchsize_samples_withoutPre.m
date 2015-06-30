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
TrainInp = TrainInp - repmat(mean(TrainInp),[size(TrainInp,1),1]);
TrainInp = TrainInp ./ repmat(sqrt(sum(TrainInp.^2)),[size(TrainInp,1),1]);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Compressive sensing
% % % % % % % % % % % % % % % % % % % % % % % % % % %

samplesTrain = size(TrainInp,2);

sweepVec = [10, 50, 100, 300, 500, 700, 1000];

%%

% poolobj = gcp('nocreate'); % If no pool, do not create new one.
% if isempty(poolobj)
%     poolsize = 0;
%     parpool('local',12);
% else
%     poolsize = poolobj.NumWorkers;
% end

%%
 for i = 1 : length(sweepVec)
    for k = 1 : floor(samplesTrain / 10)      % adjust iter
        param = struct;
        param.iter = k;
        param.batchsize = sweepVec(i);
        param.K = 512;
        param.lambda = 0.2;
        param.numThreads = -1; 
        param.verbose = false;
        param.iter_updateD = 1;

        D = [];
        basis = [];
        spCoef = [];

        epochesD = floor(k * param.batchsize);
        X = TrainInp(:,1:epochesD);
        [D,~,~] = mexTrainDL(X,param);
        
        coef = mexLasso(X,D,param);
       
        objFun(i,k) = mean(0.5*sum((X-D*coef).^2) + param.lambda*sum(abs(coef)));
        normErr(i,k) = mean(0.5*sum((X-D*coef).^2));
        sparCoef(i,k) = mean(sum(abs(coef)));sparCoef(i,k) = mean(sum(abs(coef)));sparCoef(i,k) = mean(sum(abs(coef)));
        
        disp(sprintf('Iteration (%d, %d) without pre: objective function is %f', i, k, objFun(i,k)));
        disp(sprintf('Iteration (%d, %d) without pre: L-2 norm of error is %f\n', i, k, normErr(i,k)));
%         disp(sprintf('Iteration (%d, %d) without pre: sparsity of coeff is %f\n', i, k, sparCoef(i,k)));
    end
 end

[i_row, i_col] = ind2max(objFun);
disp(sprintf('maximum objective function without pre is %f, index is (%d, %d)', objFun(i,k), i_row, i_col));

[i_row, i_col] = ind2max(normErr);
disp(sprintf('L-2 norm of error without pre is %f, index is (%d, %d)', normErr(i,k), i_row, i_col));

[i_row, i_col] = ind2max(sparCoef);
disp(sprintf('sparsity of coeff without pre: is %f\n', sparCoef(i,k), i_row, i_col));

% delete(poolobj)

filename = sprintf('./Results/sweep_batchsize_WithoutPre_lambda%.2f.mat', 0.10);
save(filename,'-v7.3')

