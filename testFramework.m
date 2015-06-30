clear
clc

poolobj = gcp('nocreate'); % If no pool, do not create new one.
if isempty(poolobj)
    poolsize = 0;
    parpool('local',2);
else
    poolsize = poolobj.NumWorkers;
end

%sweep_Lambda_samples_withoutPre;
%sweep_Lambda_samples_PreDCT;
%sweep_Lambda_samples_PreDWT;
%sweep_K_samples_withoutPre;

demoSweepWithoutPre;
demoSweepDctPre;
demoSweepDwtPre;

delete(poolobj)
