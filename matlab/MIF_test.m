softsensor=true;

N=1;
T=300;
dt=0.1;

A=[...
    1 0 dt 0;...
    0 1 0 dt;...
    0 0 1 0;...
    0 0 0 1];
B=[...
    dt*dt/2 0;...
    0 dt*dt/2;...
    dt 0;...
    0 dt];
G=eye(4);
Q=0.1*eye(4);
H=eye(4);
if softsensor
    R=3*[...
        1 0 0 0;...
        0 1 0 0;...
        0 0 2/dt 0;...
        0 0 0 2/dt];
else
    R=3*[...
        1 0 0 0;...
        0 1 0 0;...
        0 0 1 0;...
        0 0 0 1];
end

vmean=zeros(4,1);
wmean=zeros(4,1);
v=zeros(N*size(A,1),T);
w=zeros(N*size(H,1),T);
vr=randn(N*size(A,1),T);
wr=randn(N*size(H,1),T);
cholQ=chol(Q);
cholR=chol(R);
for n=1:N
    ax=(n-1)*size(A,1)+1;
    bx=ax+size(A,1)-1;
    az=(n-1)*size(H,1)+1;
    bz=az+size(H,1)-1;
    v(ax:bx,:)=vmean+cholQ*vr(ax:bx,:);
    w(az:bz,:)=wmean+cholR*wr(az:bz,:);
end

x=zeros(N*size(A,1),T);
x(:,1)=20*rand(N*size(A,1),1);
u=rand(N*size(B,2),T);
u(0.4<u)=0;
u=u-0.5;
z=zeros(N*size(H,1),T);
for n=1:N
    ax=(n-1)*size(A,1)+1;
    bx=ax+size(A,1)-1;
    az=(n-1)*size(H,1)+1;
    bz=az+size(H,1)-1;
    z(az:bz,1)=H*x(ax:bx,1)+w(az:bz,1);
    if softsensor
        z((az+2):(az+3),1)=0; % indifferentiable at t=0
    end
end
for k=2:T
    for n=1:N
        ax=(n-1)*size(A,1)+1;
        bx=ax+size(A,1)-1;
        au=(n-1)*size(B,2)+1;
        bu=au+size(B,2)-1;
        az=(n-1)*size(H,1)+1;
        bz=az+size(H,1)-1;
        x(ax:bx,k)=A*x(ax:bx,k-1)+B*u(au:bu,k-1)+v(ax:bx,k-1);
        z(az:bz,k)=H*x(ax:bx,k)+w(az:bz,k);
        if softsensor
            z((az+2):(az+3),k)=(z(az:(az+1),k)-z(az:(az+1),k-1))/dt; % overwrite velocity with differentiated positions
        end
    end
end

mif=MIF(struct('A',A,'B',B,'G',G,'Q',Q,'H',H,'R',R,'history',true),3.0);
for k=1:T
    zk=reshape(z(:,k),[size(H,2),N]);
    mif=mif.update(zk);
end

comp1=1;
comp2=2;

for n=1:N
    ax=(n-1)*size(A,1)+1;
    az=(n-1)*size(H,1)+1;
    plot(x(ax+comp1-1,:),x(ax+comp2-1,:),'DisplayName',['{\it x_k}^{(' num2str(n) ')}']);
    hold('on');
    plot(z(az+comp1-1,:),z(az+comp2-1,:),'DisplayName',['{\it z_k}^{(' num2str(n) ')}']);
end

for i=1:length(mif.filters)
    kf=mif.filters{i};
    plot(kf.x_hist(comp1,:),kf.x_hist(comp2,:),'DisplayName',['{\it x_{k|k}}^{(' num2str(i) ')}']);
end

hold('off');
legend('show');
