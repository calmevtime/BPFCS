% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Test sparsity vs time vs M
% Pre-processing data through 
% % % % % % % % % % % % % % % % % % % % % % % % % % %

start_spams

clear
clc

load ./Results/sweeplambda_WithoutPre_batchsize50.mat  normErr

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
crossValidFactor = 0.8;

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
samplesTest  = size(TestInp,2);

sweepParam = [1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

rsnr_dl = zeros(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
cr_dl = zeros(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
prd_dl = zeros(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
sparsity_dl = zeros(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
maxCorrVal = zeros(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
% basis = cell(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
% R1 = cell(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
% R2 = cell(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
% alpha = cell(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
% spCoeff = cell(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
% reconSig = cell(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));

%%

for i = 1 : mdivision 
    m_dl = floor(i * n_dl / mdivision);
    phi_dl = randn(m_dl,n_dl);

    parfor j = 1 : floor(samplesTrain / batchsize)      % adjust iter
        param = struct;
        param.iter = j;
        param.batchsize = batchsize;
        param.K = atoms;
        param.lambda = 0.01;
        param.numThreads = -1; 
        param.verbose = false;
        param.iter_updateD = 1;
        param.D = initD;

        res = 0;
        x2 = 0;
        spar = 0;

        y_dl = [];
        xs_dl = [];
        x0_dl = [];
        xhat_dl = [];

        epochesD = floor(j * param.batchsize);
        X = TrainInp(:,1:epochesD);
        D = mexTrainDL(X,param);

        psi_dl = D;
        A_dl = phi_dl * psi_dl;
        [maxCorrVal(i,j), ind1, ind2] = maxCorr(phi_dl', psi_dl);
        disp(['i = ',num2str(i),' j = ',num2str(j)]);
    end
end


delete(poolobj)

filename = sprintf('./Results/maxCorrPhiPsi_withoutPre_m%d_batchsize%d.mat', mdivision, batchsize);
save(filename,'-v7.3');
