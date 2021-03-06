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

batchsize = 50;
atoms = 512;

RawInp = RawInpLoad(1:n_dl*epochs);
% RawInp = reshape(RawInp , n_dl, epochs);
% crossValidFactor = 0.7;
% 
% indexD = randperm(atoms);
% initD = RawInp(:, indexD);
% initD = initD - repmat(mean(initD),[size(initD,1),1]);
% initD = initD ./ repmat(sqrt(sum(initD.^2)),[size(initD,1),1]);
% 
% RawInp = RawInp(:,atoms+1:end);
% epochs = epochs - atoms;
% 
% TrainInp = RawInp(:, 1 : floor(epochs*crossValidFactor));
% TrainInp = TrainInp - repmat(mean(TrainInp),[size(TrainInp,1),1]);
% TrainInp = TrainInp ./ repmat(sqrt(sum(TrainInp.^2)),[size(TrainInp,1),1]);
% 
% TestInp = RawInp(:, (size(TrainInp,2)+1):epochs);
% TestInp = TestInp - repmat(mean(TestInp),[size(TestInp,1),1]);
% TestInp = TestInp ./ repmat(sqrt(sum(TestInp.^2)),[size(TestInp,1),1]);

%%

Fs = 128;
adc = 12;
NFFT = 1024;
X = fftshift(fft(RawInp,NFFT));
subplot(211)
fvals = Fs * (-NFFT / 2 : NFFT / 2 - 1) / NFFT;
plot(fvals,abs(X));
xlabel('Frenquency(Hz)');
ylabel('|DFT Value| of original signal');

%%

N = 20;
fp = 3;
fs = 20;

lpFilt = designfilt('lowpassfir', 'PassbandFrequency', fp/Fs*2, ...
                    'StopbandFrequency', fs/Fs*2, 'PassbandRipple', 0.5, ...
                    'StopbandAttenuation', 40, 'DesignMethod', 'equiripple');
fvtool(lpFilt);
rawInpFir = filter(lpFilt, RawInp);

X = fftshift(fft(rawInpFir,NFFT));
fvals = Fs * (-NFFT / 2 : NFFT / 2 - 1) / NFFT;
subplot(212)
plot(fvals,abs(X));
xlabel('Frenquency(Hz)');
ylabel('|DFT Value| of filtered signal');

crossValidFactor = 0.8;
TrainInp = rawInpFir(:, 1 : floor(epochs*crossValidFactor));
TrainInp = TrainInp - repmat(mean(TrainInp),[size(TrainInp,1),1]);
TrainInp = TrainInp ./ repmat(sqrt(sum(TrainInp.^2)),[size(TrainInp,1),1]);

subplot(211)
plot(RawInp)
xlabel('samples')
ylabel('original signal');
subplot(212)
plot(rawInpFir)
xlabel('samples')
ylabel('filtered signal');

%%

RawInp = rawInpFir(1:n_dl*epochs);
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

%% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Setting parameters for training
% % % % % % % % % % % % % % % % % % % % % % % % % % %

param.K = 512;  
param.lambda = 0.01;            % sparsity constraint 
param.numThreads = -1; 
param.batchsize = 50;
param.verbose = false;
param.iter = floor(size(TrainInp,2) / param.batchsize); 
% param.clean = false;
% param.verbose = true;

%%

disp('Starting to  train the dictionary');

[D,basis,~] = mexTrainDL(TrainInp,param);
coef = mexLasso(TrainInp,D,param);
R1 = mean(0.5*sum((TrainInp-D*coef).^2) + param.lambda*sum(abs(coef)));

objBasis = cell(1,param.iter);
for i = 1 : param.iter
    objBasis{i} = basis(:,(i - 1) * param.K + 1 : i * param.K);
end

%%

delay = 0.01;
numOfAtoms = 10;
randAtoms = randperm(param.K);

writerObj  = VideoWriter('./Results/DictEvo.avi');
writerObj.FrameRate = 5; 
open(writerObj);
fig = figure('units','normalized','outerposition',[0 0 1 1]);

%%
han = annotation('textbox', [0.4,0.89,0.1,0.1], 'String', 'batch index for training: 1');
% for i = 1 : param.iter  
for i = 1 : 60
    for j = 1 : numOfAtoms
        h = subplot(2,5,j);
        cla(h);
        plot(objBasis{i}(:,randAtoms(j)));
        
        axis([0 n_dl -0.4 0.4]);
        title(['Atom: ',num2str(randAtoms(j))]);
        xlabel('Time Samples');
        ylabel('Normalized Amplitude');
        
        frame = getframe(fig);
        writeVideo(writerObj,frame);
        
    end
	pause(delay);
    delete(han);
    han = annotation('textbox', [0.4,0.89,0.1,0.1], 'String', ['batch index for training: ',num2str(i)]);
end

close(writerObj);
