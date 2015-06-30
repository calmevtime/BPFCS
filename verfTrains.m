% clear 
% clc
% 
% %%
% fs = 240;
% dt = 1/fs;
% stopTime = 1;
% t = 0 : dt : stopTime-dt;
% t = t(1,1:128);
% 
% fc = 60;
% x1 = cos(2*pi*fc*t)';
% 
% fc = 120;
% t = stopTime + t;
% x1 = [x1;cos(2*pi*fc*t)'];
% 
% % plot(x)
% 
% dctx1 = dct(x1);
% dctmtxx1 = dctmtx(length(x1)) * x1;
% res1 = norm(dctx1-dctmtxx1);
% 
% xres1 = idct(dctx1);
% xmtxres1 = dctmtx(length(x1))' * dctmtxx1;
% 
% subplot(311)
% plot(x1);
% title('Original Signal')
% axis([0 length(x1) -1 1])
% 
% subplot(312)
% plot(dctx1);
% title('After DCT')
% axis([0 length(x1) -10 10])
% 
% subplot(313)
% plot(xres1);
% title('Restored Signal')
% axis([0 length(x1) -1 1])
% 
% 
% %%
% fs = 240;
% dt = 1/fs;
% stopTime = 1;
% t = 0 : dt : stopTime-dt;
% t = t(1,1:128);
% 
% fc = 120;
% x2 = cos(2*pi*fc*t)';
% 
% fc = 60;
% t = stopTime + t;
% x2 = [x2;cos(2*pi*fc*t)'];
% 
% % plot(x2)
% 
% dctx2 = dct(x2);
% dctmtxx2 = dctmtx(length(x2)) * x2;
% res2 = norm(dctx2-dctmtxx2);
% 
% xres2 = idct(dctx2);
% xmtxres2 = dctmtx(length(x2))' * dctmtxx2;
% 
% figure
% subplot(311)
% plot(x2);
% title('Original Signal')
% axis([0 length(x2) -1 1])
% 
% subplot(312)
% plot(dctx2);
% title('After DCT')
% axis([0 length(x2) -10 10])
% 
% subplot(313)
% plot(xres2);
% title('Restored Signal')
% axis([0 length(x2) -1 1])
% 
% %%
% fs = 240;
% dt = 1/fs;
% stopTime = 1;
% t = 0 : dt : stopTime-dt;
% t = t(1,1:128);
% 
% fc = 120;
% x3 = cos(2*pi*fc*t)';
% 
% fc = 60;
% t = stopTime + t;
% x3 = [x3;cos(2*pi*fc*t)'];
% x3 = [zeros(100,1);x3(101:end,:)];
% 
% % plot(x2)
% 
% dctx3 = dct(x3);
% dctmtxx3 = dctmtx(length(x3)) * x3;
% res3 = norm(dctx3-dctmtxx3);
% 
% xres3 = idct(dctx3);
% xmtxres3 = dctmtx(length(x3))' * dctmtxx3;
% 
% figure
% subplot(311)
% plot(x3);
% title('Original Signal')
% axis([0 length(x3) -1 1])
% 
% subplot(312)
% plot(dctx3);
% title('After DCT')
% axis([0 length(x3) -10 10])
% 
% subplot(313)
% plot(xres3);
% title('Restored Signal')
% axis([0 length(x3) -1 1])
% 
% res = norm(dctx3-dctx3);





%% DWT 

fs = 240;
dt = 1/fs;
stopTime = 1;
t = 0 : dt : stopTime-dt;
t = t(1,1:128);

fc = 60;
x1 = cos(2*pi*fc*t)';

fc = 120;
t = stopTime + t;
x1 = [x1;cos(2*pi*fc*t)'];

% plot(x)

dwtx1 = dwt(x1);
dwtmtxx1 = dwtmtx(length(x1)) * x1;
res1 = norm(dwtx1-dwtmtxx1);

xres1 = idwt(dwtx1);
xmtxres1 = dwtmtx(length(x1))' * dwtmtxx1;

subplot(311)
plot(x1);
title('Original Signal')
axis([0 length(x1) -1 1])

subplot(312)
plot(dwtx1);
title('After dwt')
axis([0 length(x1) -10 10])

subplot(313)
plot(xres1);
title('Restored Signal')
axis([0 length(x1) -1 1])


%%
fs = 240;
dt = 1/fs;
stopTime = 1;
t = 0 : dt : stopTime-dt;
t = t(1,1:128);

fc = 120;
x2 = cos(2*pi*fc*t)';

fc = 60;
t = stopTime + t;
x2 = [x2;cos(2*pi*fc*t)'];

% plot(x2)

dwtx2 = dwt(x2);
dwtmtxx2 = dwtmtx(length(x2)) * x2;
res2 = norm(dwtx2-dwtmtxx2);

xres2 = idwt(dwtx2);
xmtxres2 = dwtmtx(length(x2))' * dwtmtxx2;

figure
subplot(311)
plot(x2);
title('Original Signal')
axis([0 length(x2) -1 1])

subplot(312)
plot(dwtx2);
title('After dwt')
axis([0 length(x2) -10 10])

subplot(313)
plot(xres2);
title('Restored Signal')
axis([0 length(x2) -1 1])

%% 

fs = 240;
dt = 1/fs;
stopTime = 1;
t = 0 : dt : stopTime-dt;
t = t(1,1:128);

fc = 120;
x3 = cos(2*pi*fc*t)';

fc = 60;
t = stopTime + t;
x3 = [x3;cos(2*pi*fc*t)'];
x3 = [zeros(100,1);x3(101:end,:)];

% plot(x2)

dwtx3 = dwt(x3);
dwtmtxx3 = dwtmtx(length(x3)) * x3;
res3 = norm(dwtx3-dwtmtxx3);


figure
subplot(311)
plot(x3);
title('Original Signal')
axis([0 length(x3) -1 1])

subplot(312)
plot(dwtx3);
title('After dwt')
axis([0 length(x3) -10 10])

subplot(313)
plot(xres3);
title('Restored Signal')
axis([0 length(x3) -1 1])

res = norm(dwtx3-dwtx3);
