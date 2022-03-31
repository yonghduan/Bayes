clear all;
close all;
clc;

mu1=[0 0];
S1=[0.8 0.1];
data1=mvnrnd(mu1,S1,1000);
plot(data1(:,1),data1(:,2),'r.');
hold on;

mu2=[2 4];
S2=[0.4 1.3];
data2=mvnrnd(mu2,S2,1000);
plot(data2(:,1),data2(:,2),'g.');

mu3=[-2 3];
S3=[2.4 1.3];
data3=mvnrnd(mu3,S3,1000);
plot(data3(:,1),data3(:,2),'b.');

%利用EM算法对高斯混合模型聚类
data=[data1;data2;data3];
mu{1} = rand(1,2);
mu{2} = rand(1,2);
mu{3} = rand(1,2);
sigma{1} = rand(1,2);
sigma{2} = rand(1,2);
sigma{3} = rand(1,2);

p = [0.3 0.4 0.4];
w=zeros(length(data),3);
for i=1:1000
    
    %E-step
    for j=1:3
        w(:,j) = p(j)*mvnpdf(data,mu{j},sigma{j});
    end
    w = w./repmat(sum(w,2),[1 3]);
    
    %M-step
    for j=1:3
        mu{j} = w(:,j)'* data / sum(w(:,j));
        sigma{j} = sqrt(w(:,j)'*((data-mu{j}).*(data-mu{j})) / sum(w(:,j)));
    end
    p = sum(w) / length(data);
    
end

figure;
w = uint8(w);
data1 = data(w(:,1)==1,:);
data2 = data(w(:,2)==1,:);
data3 = data(w(:,3)==1,:);

plot(data1(:,1),data1(:,2),'r.');
hold on;
plot(data2(:,1),data2(:,2),'g.');
plot(data3(:,1),data3(:,2),'b.');