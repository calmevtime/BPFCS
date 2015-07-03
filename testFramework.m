clear
clc

start_spams


poolobj = gcp('nocreate'); % If no pool, do not create new one.
if isempty(poolobj)
    poolsize = 0;
    parpool('local',24);
else
    poolsize = poolobj.NumWorkers;
end

%%

srcFileName = {'preDCT/sweep_Lambda_PreDCT.m', 'preDWT/sweep_Lambda_PreDWT.m', ...
            'preDCT/demoSweepDctPre.m', 'preDWT/demoSweepDwtPre.m', ...
            'preFilterDCT/sweep_Lambda_preFilterDCT.m'};
dstFileName = {'sweep_Lambda_PreDCT.m', 'sweep_Lambda_PreDWT.m', ...
            'demoSweepDctPre.m', 'demoSweepDwtPre.m', ...
            'demoSweepFilterDCT.m'};      
        
runDir = './runtimeFolder/';
if ~isequal(exist(runDir,'dir'),7)
    mkdir(tmp);
end

for i = 1 : length(srcFileName)
    dstFile = fullfile(runDir,dstFileName{i});
    srcFile = fullfile(srcFileName{i});
    copyfile(srcFile, dstFile);
    run(fullfile(runDir,dstFileName{i}));
end

%%

sweep_Lambda_samples_withoutPre;
% sweep_K_samples_withoutPre;


% demoSweepFilter;
% demoSweepWithoutPre;
% 
% demoSweepDctPre;
% demoSweepDwtPre;

delete(poolobj)
