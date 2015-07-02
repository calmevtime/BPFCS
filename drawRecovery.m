% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Test signal dimension
% % % % % % % % % % % % % % % % % % % % % % % % % % %

clear
clc

%% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Prepare raw data
% % % % % % % % % % % % % % % % % % % % % % % % % % %

RawInpLoad = load('15814m_ltdbECG_1h.mat');
RawInpLoad = RawInpLoad.val;
n_dl = 128;
epochs = floor(length(RawInpLoad) / n_dl);    % 4517
RawInpLoad = RawInpLoad(1:n_dl * epochs);

%%
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

samplesTrain = size(TrainInp,2);
samplesTest  = size(TestInp,2);

% % % % % % % % % % % % % % % % % % % % % % % % % % %
% Setting parameters for training
% % % % % % % % % % % % % % % % % % % % % % % % % % %

param.K = atoms;  
param.lambda = 1e-2;            % sparsity constraint 
param.numThreads = -1; 
param.verbose = false;
% param.clean = false;
% param.verbose = true;
param.batchsize = 50;
param.iter = floor(size(TrainInp,2) / param.batchsize);
param.D = initD;

disp('Starting to  train the dictionary');
[D,~,~] = mexTrainDL(TrainInp,param);

coef = mexLasso(TrainInp,D,param);
objFun = mean(0.5*sum((TrainInp-D*coef).^2) + param.lambda*sum(abs(coef)));
normErr = mean(0.5*sum((TrainInp-D*coef).^2));
sparCoef = 1 - length(find((coef))) / length(coef(:));

disp(sprintf('without pre: objective function is %f', objFun));
disp(sprintf('without pre: L-2 norm of error is %f\n', normErr));

%%

CR = 5;
m_dl = floor(n_dl / CR);
phi_dl = randn(m_dl,n_dl);
psi_dl = D;
A_dl = phi_dl * psi_dl;

for ep = 1:samplesTest
    y_dl = phi_dl * TestInp(:,ep);
    x0_dl = pinv(A_dl) * y_dl; 

    xs_dl = l1eq_pd(x0_dl, A_dl, [], y_dl, 1e-3); 
    xhat_dl = psi_dl * xs_dl;

    subplot(311)
    plot(TestInp(:,ep));
    subplot(312)
    plot(xhat_dl);
    subplot(313)
    if(ep <= param.K)
        plot(param.D(:,ep));
    end
% 
%     res = res + sum(norm(TestInp(:,ep) - xhat_dl).^2);
%     x2 = x2 + sum(TestInp(:,ep).^2);
%     spar = spar + length(find(abs(xs_dl)>0.001) ); 
end


%%
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
        xlabel('Time Samples')
        ylabel('Normalized Amplitude');
        
        frame = getframe(fig);
        writeVideo(writerObj,frame);
        pause(delay);
    end
    delete(han);
    han = annotation('textbox', [0.4,0.89,0.1,0.1], 'String', ['batch index for training: ',num2str(i)]);
end

% close(writerObj);


%%
% field1 = 'muA'; value1 = zeros(1,param.iter);
% field2 = 'j';   value2 = zeros(1,param.iter);
% field3 = 'k';   value3 = zeros(1,param.iter);
% 
% s = struct(field1,value1,field2,value2,field3,value3);
% 
% for i = 1 : param.iter
%     temp = 0;
%     for j = (i - 1) * param.K + 1 : i * param.K
%         for k = j+1 : i * param.K
%             temp = abs(basis(:,j)' * basis(:,k)) / (norm(basis(:,j)) * norm(basis(:,k)));
%             if(temp > s.muA(i))
%                 s.muA(i) = temp;
%                 s.j(i) = j;
%                 s.k(i) = k;
%             end
%         end
%     end
% end
% 