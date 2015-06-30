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
param.lambda = 0.15;            % sparsity constraint 
param.numThreads = -1; 
param.batchsize = 50;
param.verbose = false;
param.iter = 60; 
param.clean = false;
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

disp('Starting to  train the dictionary');
[D,basis] = mexTrainDL(TrainInp,param);

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

objBasis = cell(1,param.iter);
for i = 1 : param.iter
    objBasis{i} = basis(:,(i - 1) * param.K + 1 : i * param.K);
end

%%

delay = 0.1;
numOfAtoms = 10;
randAtoms = randperm(param.K);

writerObj  = VideoWriter('./Results/DictEvo.avi');
writerObj.FrameRate = 5; 
open(writerObj);
fig = figure('units','normalized','outerposition',[0 0 1 1 ]);


for i = 1 : param.iter  
    for j = 1 : numOfAtoms
        h = subplot(2,5,j);
        cla(h);
        plot(objBasis{i}(:,randAtoms(j)));
%         plot(objBasis{i}(:,j));
        
        hold on
%         axis([0 130 -0.3 0.3])
        title(['Atom: ',num2str(randAtoms(j))]);
%         title(['Atom: ',num2str(j)]);
        xlabel('Time Samples')
        ylabel('Normalized Amplitude');
        axis([0 n_dl -0.4 0.4]);
        frame = getframe(fig);
        writeVideo(writerObj,frame);
        pause(delay);
    end
end

close(writerObj);