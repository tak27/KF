softsensor=true;

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
v=vmean+chol(Q)*randn(size(A,1),T);
w=wmean+chol(R)*randn(size(H,1),T);

x=zeros(size(A,1),T);
x(:,1)=20*rand(size(A,2),1);
u=rand(size(B,2),T);
u(0.4<u)=0;
u=u-0.5;
z=zeros(size(H,1),T);
z(:,1)=H*x(:,1)+w(:,1);
if softsensor
    z(3:4,1)=0; % indifferentiable at t=0
end
for k=2:T
    x(:,k)=A*x(:,k-1)+B*u(:,k-1)+v(:,k-1);
    z(:,k)=H*x(:,k)+w(:,k);
    if softsensor
        z(3:4,k)=(z(1:2,k)-z(1:2,k-1))/dt; % overwrite velocity with differentiated positions
    end
end

kf=KF(z(:,1),R,struct('A',A,'B',B,'G',G,'Q',Q,'H',H,'R',R,'history',true));
for k=2:T
    %pred=kf.predict(u(:,k)); % in case that control vector is known
    pred=kf.predict(); % in case that control vector is unknown
    kf=kf.update(z(:,k),pred);
end

comp1=1;
comp2=2;

plot(kf.x_hist(comp1,:),kf.x_hist(comp2,:),'DisplayName','x_{k|k}');
hold('on');
plot(z(comp1,:),z(comp2,:),'DisplayName','z_k');
plot(x(comp1,:),x(comp2,:),'DisplayName','x_k');
hold('off');
legend('show');
