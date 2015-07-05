% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Test sparsity vs time vs M
% Pre-processing data through 
% % % % % % % % % % % % % % % % % % % % % % % % % % %

% start_spams

clear
clc

load ../Results/preDWT/sweeplambda_PreDWT_batchsize50.mat normErr

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

TestInp = RawInp(:, (size(TrainInpDWT,2)+1):epochs);
TestInp = TestInp - repmat(mean(TestInp),[size(TestInp,1),1]);
TestInp = TestInp ./ repmat(sqrt(sum(TestInp.^2)),[size(TestInp,1),1]);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Compressive sensing
% % % % % % % % % % % % % % % % % % % % % % % % % % %

samplesTrain = size(TrainInpDWT,2);
samplesTest  = size(TestInp,2);

sweepParam = [1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

rsnr_dl = zeros(mdivision,length(1:floor(samplesTrain / batchsize)));
rsnr_thres = zeros(mdivision,length(1:floor(samplesTrain / batchsize)));
prd_dl = zeros(mdivision,length(1:floor(samplesTrain / batchsize)));
prd_thres = zeros(mdivision,length(1:floor(samplesTrain / batchsize)));
sparsity_thres = zeros(mdivision,length(1:floor(samplesTrain / batchsize)));

% basis = cell(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
% R1 = cell(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
% R2 = cell(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
% alpha = cell(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
% spCoeff = cell(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));
% reconSig = cell(length(sweepParam),mdivision,length(1:floor(samplesTrain / batchsize)));

%%

% poolobj = gcp('nocreate'); % If no pool, do not create new one.
% if isempty(poolobj)
%     poolsize = 0;
%     parpool('local',20);
% else
%     poolsize = poolobj.NumWorkers;
% end

%%

thres = 0.0001;
lambda = sweepParam(3);

for i = 1 : mdivision 
    m_dl = floor(i * n_dl / mdivision);
    phi_dl = randn(m_dl,n_dl);

    parfor j = 1 : floor(samplesTrain / batchsize)      % adjust iter
        param = struct;
        param.iter = j;
        param.batchsize = batchsize;
        param.K = atoms;
        param.lambda = lambda;
        param.numThreads = -1; 
        param.verbose = false;
        param.iter_updateD = 1;
        param.D = initD;

        res = 0;
        norm2x = 0;        
        res_thres = 0;
        spar_thres = 0;

        y_dl = [];
        xs_dl = [];
        x0_dl = [];
        xhat_dl = [];
        D = [];

        epochesD = floor(j * param.batchsize);
        X = TrainInpDWT(:,1:epochesD);
        D = mexTrainDL(X,param);

        psi_dl = D;
        A_dl = phi_dl * wt' * psi_dl;

        for ep = 1:samplesTest
            y_dl = phi_dl * TestInp(:,ep);
            x0_dl = pinv(A_dl) * y_dl; 
            xs_dl = l1eq_pd(x0_dl, A_dl, [], y_dl, normErr(3,j)); 
            xhat_dl = wt' * psi_dl * xs_dl;

            res = res + sum(norm(TestInp(:,ep) - xhat_dl).^2);
            norm2x = norm2x + sum(TestInp(:,ep).^2);
            
            xs_thres = xs_dl;
            xs_thres(abs(xs_dl) < thres) = 0;
            xhat_thres = psi_dl * xs_thres;
            res_thres = res_thres + sum(norm(TestInp(:,ep) - xhat_thres).^2);
            spar_thres = spar_thres + length(find(xs_thres)); 
        end
        rsnr_dl(i,j) = 20 * log10(sqrt(norm2x / res));
        prd_dl(i,j) = sqrt(res / norm2x);
        
        rsnr_thres(i,j) = 20 * log10(sqrt(norm2x / res_thres));
        prd_thres(i,j) = sqrt(res_thres / norm2x);
        sparsity_thres(i,j) = 1 - spar_thres / samplesTest / length(xs_dl);
    end
end


% delete(poolobj)

filename = sprintf('../Results/preDWT/demoSweepDwtPre_m%d_batchsize%d.mat', mdivision, batchsize);
save(filename,'-v7.3');

%% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Plot results
% % % % % % % % % % % % % % % % % % % % % % % % % % %

% dateToday = '05-Jul-2015';
% plotRSNR_Sparsity(dateToday, 'preDWT', mdivision, batchsize, samplesTrain, n_dl, rsnr_dl, rsnr_thres, sparsity_thres, lambda);

%% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Plot reconstruction process
% % % % % % % % % % % % % % % % % % % % % % % % % % %
% 
% delay = 1;
% epSel = 1000;
% writerObj  = VideoWriter('./Results/reconstruction.avi');
% writerObj.FrameRate = 5; 
% open(writerObj);
% fig = figure('units','normalized','outerposition',[0 0 1 1]);
% plot(TestInp(:,epSel));
% axis([1 n_dl -0.3 0.3]);
% hold on
% 
% for i = 1 : floor(samplesTrain / 50) 
%     reconSigMat = cell2mat(reconSig{5,i}(:,epSel));
%     
% 	h = plot(1:n_dl,reconSigMat);
%     axis([1 n_dl -0.3 0.3]);
% % 	hold on
% 	frame = getframe(fig);
% 	writeVideo(writerObj,frame);
%     pause(delay);
%     delete(h);
% end
% 
% plot(1:n_dl,reconSigMat);
% close(writerObj);

