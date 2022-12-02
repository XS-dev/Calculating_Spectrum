clear; 
close all; 
clc;
rng(0);  % for reproducibility, do not change it!
m = 50;  % num samples
n = 200; % num variables, note that n > m

A = rand(m, n);  % simulate matrix A 模拟矩阵A
x = zeros(n, 1); % simulate true sparse signal x: line 13-16 模拟稀疏信号
nz = 10;         % 
nz_idx = randperm(n);
x(nz_idx(1:nz)) = 2 * rand(nz, 1);
y = A*x;         % simulate a degraded signal
y = y + 0.1 * rand(m, 1); % add some noise to the degraded signal




[m, n] = size(A);


b = zeros(n,1);  %n行1列
b1 = zeros(m,1);
b2 = zeros(n,1);

beta1 = 1e-1;  % The para. in Algrithm 1
beta2 = 1e-2;
lambda = 1e-2;

k=0;
ReErr=1;
maxitr = 500;
tol  = 1e-8;
b_old = b;
b1_old = b1;
b2_old = b2;

while ReErr> tol && k<maxitr

u_old = sign(y-A*b_old-b1_old).*max(abs(y-A*b_old-b1_old)-1/beta1,0) ;%第一次计算u



v_old = sign(b_old-b2_old).*max(abs(b_old-b2_old)-lambda /beta2,0) ;%第一次计算v




I = eye(n,n);
b_new = (inv(beta1*((A.')*A)+beta2.*I))*(beta2*(v_old+b2_old)-beta1*(A.')*(u_old-y +b1_old));%更新b的值







b1_new = b1_old+1.618*(u_old-(y-A*b_new));
b2_new = b2_old+1.618*(v_old-b_new);
b1_old = b1_new;
b2_old = b2_new;


ReErr = norm((b_new-b_old),2)./ norm(b_new,2);
b_old = b_new;



k=k+1;
    
end


figure,
subplot(3,1, 1);
plot(x,'b-','Linewidth',2), axis auto;
title('Original signal: b');

subplot(3,1, 2);
plot(y,'r-','Linewidth',2), axis auto;
title('Noisy signal: y=Ab');

subplot(3,1, 3);
plot(b_old,'g-','Linewidth',2), axis auto;
title('recoverd signal: b by the solution of (1)');


