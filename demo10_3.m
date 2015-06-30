clear
clc

waveletName = {'db1','db2','bior2.4','bior3.7','rbio2.4','rbio2.6','rbio3.7','rbio3.9'};
for i = 4 : length(waveletName)
    demo10_2(waveletName{i});
end
