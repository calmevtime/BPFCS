clear
clc

baseDir = '/home/kaixu/myGitHub/BPFCS/';

addpath(fullfile(baseDir, 'test_release'));
addpath(fullfile(baseDir, 'src_release'));
addpath(fullfile(baseDir, 'build'));
addpath(fullfile(baseDir, 'Results'));
setenv('MKL_NUM_THREADS','1')
setenv('MKL_SERIAL','YES')
setenv('MKL_DYNAMIC','NO')

cd(fullfile(baseDir, 'framework'));

poolobj = gcp('nocreate'); % If no pool, do not create new one.
if isempty(poolobj)
    poolsize = 0;
    parpool('local',20);
else
    poolsize = poolobj.NumWorkers;
end

%%
% 
% fileName = {'sweep_Lambda_preFilter.m','sweep_Lambda_preFilterDCT.m','sweep_Lambda_withoutPre.m','sweep_Lambda_PreDCT.m', 'sweep_Lambda_PreDWT.m', ...
%             'demoSweepFilter.m','demoSweepDctPre.m', 'demoSweepDwtPre.m', ...
%             'demoSweepFilterDCT.m'};      
%         
% runDir = fullfile(baseDir, 'runtimeFolder');
% if ~isequal(exist(runDir,'dir'),7)
%     mkdir(runDir);
% end
% 
% for i = 1 : length(fileName)
%     dstFile = fullfile(runDir,fileName{i});
%     srcFile = fullfile(baseDir, 'framework', fileName{i});
%     copyfile(srcFile, dstFile);
%     run(dstFile);
% end

%%

% sweep_Lambda_preFilter;
% sweep_Lambda_preFilterDCT;
% sweep_Lambda_withoutPre;
% sweep_Lambda_PreDCT;
% sweep_Lambda_PreDWT;
% 
demoSweepFilter;
demoSweepFilterDCT;
demoSweepWithoutPre;
demoSweepDctPre;
demoSweepDwtPre;
% demoSweepWithoutPreThres;


delete(poolobj)