% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Test signal dimension
% % % % % % % % % % % % % % % % % % % % % % % % % % %

clear
clc

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Prepare raw data
% % % % % % % % % % % % % % % % % % % % % % % % % % %
RawInpLoad = load('15814m_ltdbECG_1h.mat');
RawInpLoad = RawInpLoad.val;
n_dl = 102;
epochs = floor(length(RawInpLoad) / n_dl);    % 4517
RawInpLoad = RawInpLoad(1:n_dl * epochs);

RawInp = RawInpLoad(1:n_dl*epochs);
RawInp = reshape(RawInp , n_dl, epochs);
crossValidFactor = 0.7;
% TrainInp = RawInp(:, 1:floor(epochs*crossValidFactor));
TrainInp = RawInp;
TrainInp = TrainInp - repmat(mean(TrainInp),[size(TrainInp,1),1]);
TrainInp = TrainInp ./ repmat(sqrt(sum(TrainInp.^2)),[size(TrainInp,1),1]);
% 
% TestInp = RawInp(:, (size(TrainInp,2)+1):epochs);
% % TestInp = RawInp(:, (size(TrainInp,2)+1):size(TrainInp,2)*2);
% TestInp = TestInp - repmat(mean(TestInp),[size(TestInp,1),1]);
% TestInp = TestInp ./ repmat(sqrt(sum(TestInp.^2)),[size(TestInp,1),1]);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Setting parameters for training
% % % % % % % % % % % % % % % % % % % % % % % % % % %

param.K = 512;  
param.lambda = 10;            % sparsity constraint 
param.numThreads = -1; 
param.batchsize = 400;
param.verbose = false;
param.iter = 10;
param.clean = false;

param2 = param;
% param.verbose = true;

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

basis = cell(1,floor(size(TrainInp,2) / param.batchsize));

X = TrainInp(:,1:param.batchsize);
[D,model] = mexTrainDL(X,param);
basis{1} = {D};
    
for i = 2 : floor(size(TrainInp,2) / param.batchsize)
    X = TrainInp(:,(i-1)*param.batchsize+1:i*param.batchsize);
    param2 = param;
    param2.D = D;
    [D,model] = mexTrainDL(X,param2,model);
    basis{i} = {D};
    
    coef = mexLasso(X,D,param2);
    alpha = coef;
    R = mean(0.5*sum((X-D*coef).^2) + param2.lambda*sum(abs(coef)));
end

%%

delay = 0.01;
numOfAtoms = 10;
randAtoms = randperm(param2.K);

writerObj  = VideoWriter('./Results/DictEvo.avi');
writerObj.FrameRate = 5; 
open(writerObj);
fig = figure('units','normalized','outerposition',[0 0 1 1 ]);

%%

% for i = 1 : param.iter  
for i = 1 : floor(size(TrainInp,2) / param2.batchsize)
    for j = 1 : numOfAtoms
        h = subplot(2,5,j);
        cla(h);
        matBasis = cell2mat(basis{i});
        plot(matBasis(:,randAtoms(j)));
        title(['Atom: ',num2str(randAtoms(j))]);
        xlabel('Time Samples')
        ylabel('Normalized Amplitude');
        axis([0 n_dl -0.5 0.5]);
        
        frame = getframe(fig);
        writeVideo(writerObj,frame);
        pause(delay);
    end
end

close(writerObj);