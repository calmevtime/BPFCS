clear
clc

poolobj = gcp('nocreate'); % If no pool, do not create new one.
if isempty(poolobj)
    poolsize = 0;
    parpool('local',10);
else
    poolsize = poolobj.NumWorkers;
end

sweep_Lambda_samples_withoutPre;
sweep_Lambda_samples_PreDCT;
sweep_Lambda_samples_PreDWT;
sweep_batchsize_samples_withoutPre;
sweep_K_samples_withoutPre;

delete(poolobj);