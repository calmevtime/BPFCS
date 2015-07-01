% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Test sparsity vs time vs M
% Pre-processing data through 
% % % % % % % % % % % % % % % % % % % % % % % % % % %

start_spams

clear
clc

load ./Results/sweeplambda_WithoutPre_batchsize50.mat

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
for k = 1 : length(sweepParam)
    for i = 1 : mdivision 
        m_dl = floor(i * n_dl / mdivision);
        phi_dl = randn(m_dl,n_dl);
%         phi_dl = orth(phi_dl')';

        for j = 1 : floor(samplesTrain / batchsize)      % adjust iter
            param = struct;
            param.iter = j;
            param.batchsize = batchsize;
            param.K = atoms;
            param.lambda = sweepParam(k);
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
            D = [];

            epochesD = floor(j * param.batchsize);
            X = TrainInp(:,1:epochesD);
            D = mexTrainDL(X,param);

%             coef = mexLasso(X,D,param);
%             alpha{i,j} = coef;
%             R1{i,j} = mean(0.5*sum((X-D*coef).^2) + param.lambda*sum(abs(coef)));
%             R2{i,j} = mean(0.5*sum(X-D*alpha{i,j}).^2);
%             fprintf('Objective function for i=%d, j=%d is %f', i, j, R1{i,j});

%             basis(i, j) = {D};

            psi_dl = D;
            A_dl = phi_dl * psi_dl;

            for ep = 1:samplesTest
                y_dl = phi_dl * TestInp(:,ep);
                x0_dl = pinv(A_dl) * y_dl; 
                
                xs_dl = l1eq_pd(x0_dl, A_dl, [], y_dl, normErr(k,j)); 
                xhat_dl = psi_dl * xs_dl;
                
                subplot(211)
                plot(TestInp(:,ep));
                subplot(212)
                plot(xhat_dl);
%                 spCoeff{k,i,j}(:,ep) = {xs_dl};
%                 reconSig{k,i,j}(:,ep) = {xhat_dl};
                res = res + sum(norm(TestInp(:,ep) - xhat_dl).^2);
                x2 = x2 + sum(TestInp(:,ep).^2);
                spar = spar + length(find(abs(xs_dl)>0.001) ); 
            end
            rsnr_dl(k,i,j) = 20 * log10(sqrt(x2 / res));
            cr_dl(k,i,j) = n_dl / m_dl;
            sparsity_dl(k,i,j) = 1 - spar / samplesTest / length(xs_dl);
            prd_dl(k,i,j) = sqrt(res / x2);
        end
    end
end

delete(poolobj)

filename = sprintf('./Results/sweeplambda_withoutPre_m%d_batchsize%d.mat', mdivision, batchsize);
save(filename,'-v7.3');

% subplot(211)
% plot(TestInp(:,ep));
% subplot(212)
% plot(xres_dl);

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
% for i = 1 : floor(samplesTrain / batchsize) 
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

%% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Plot results
% % % % % % % % % % % % % % % % % % % % % % % % % % %
