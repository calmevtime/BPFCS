function plotRSNR(dateToday, mdivision, batchsize, samplesTrain, n_dl, rsnr_dl, prd_dl, sparsity_dl, sweepParam)

cc = jet(mdivision);
str = cell(1,mdivision);

dirpath = sprintf('./Results/%s/',dateToday);
if ~isequal(exist(dirpath, 'dir'),7)
    mkdir(dirpath)
end

if size(rsnr_dl,3) == 1
    figure
    j = 1 : floor(samplesTrain / batchsize);
    subplot(3,1,1)
    for i = 1 : mdivision
        plot(floor(j * batchsize),rsnr_dl(i,:),'Color',cc(i,:) ) ;
        str{i}=['m=',num2str(floor(i * n_dl / mdivision))];
        hold on
    end
    legend(str)
    xlabel('Iterations');
    ylabel('RSNR(dB)');

    subplot(3,1,2)
    for i = 1 : mdivision
        plot(floor(j * batchsize),prd_dl(i,:),'Color',cc(i,:) );
        str{i}=['m=',num2str(floor(i * n_dl / mdivision))];
        hold on
    end
    % legend(str)
    xlabel('Iterations');
    ylabel('PRD');

    subplot(3,1,3)
    for i = 1 : mdivision
        plot(floor(j * batchsize),sparsity_dl(i,:),'Color',cc(i,:) );
        str{i}=['m=',num2str(floor(i * n_dl / mdivision))];
        hold on
    end
    % legend(str)
    xlabel('Iterations');
    ylabel('Sparsity');
else
    k = 1 : floor(samplesTrain / batchsize);
    for i = 1 : size(rsnr_dl,1)
        h = figure('units','normalized','outerposition',[0 0 1 1]);
        subplot(3,1,1)
        for j = 1 : mdivision
            rsnr = reshape(rsnr_dl(i,j,:),1,length(k));
            plot(k * batchsize,rsnr,'Color',cc(j,:) ) ;
            str{j}=['m=',num2str(floor(j * n_dl / mdivision))];
            hold on
        end
        legend(str, 'Location', [0.88,0.66,0.1,0.1]);
        xlabel('Iterations');
        ylabel('RSNR(dB)');
        title(['lambda=',num2str(sweepParam(i))])

        subplot(3,1,2)
        for j = 1 : mdivision
            prd = reshape(prd_dl(i,j,:),1,length(k));
            plot(k * batchsize,prd,'Color',cc(j,:) );
            str{j}=['m=',num2str(floor(j * n_dl / mdivision))];
            hold on
        end
        % legend(str)
        xlabel('Iterations');
        ylabel('PRD');

        subplot(3,1,3)
        for j = 1 : mdivision
            sparsity = reshape(sparsity_dl(i,j,:),1,length(k));
            plot(k * batchsize,sparsity,'Color',cc(j,:) );
            str{j}=['m=',num2str(floor(j * n_dl / mdivision))];
            hold on
        end
        % legend(str)
        xlabel('Iterations');
        ylabel('Sparsity');
        
        fPath = sprintf('./Results/%s/lambda%s.fig',dateToday,num2str(sweepParam(i)));
        fPathBMP = sprintf('./Results/%s/lambda%s.bmp',dateToday,num2str(sweepParam(i)));
        savefig(h, fPath)
        saveas(h,fPathBMP);
    end
end
% 