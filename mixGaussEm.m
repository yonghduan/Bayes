function [label, model, llh] = mixGaussEm(X, init)
% Perform EM algorithm for fitting the Gaussian mixture model.
% Input: 
%   X: d x n data matrix
%   init: k (1 x 1) number of components or label (1 x n, 1<=label(i)<=k) or model structure
% Output:
%   label: 1 x n cluster label
%   model: trained model structure
%   llh: loglikelihood
% Written by Mo Chen (sth4nth@gmail.com).
%% init
fprintf('EM for Gaussian mixture: running ... \n');
tol = 1e-6;
maxiter = 500;
llh = -inf(1,maxiter);
R = initialization(X,init);
for iter = 2:maxiter
    [~,label(1,:)] = max(R,[],2);      %label表示1*n的类别向量
    R = R(:,unique(label));            % remove empty clusters，unique输出label向量中不重复的元素，表示非空类别向量    
    model = maximization(X,R);         %EM算法的M step，X表示数据矩阵，R表示类别矩阵，model是结构体，表示模型，其属性是模型的参数
    [R, llh(iter)] = expectation(X,model);            %EM算法的E step，X表示数据矩阵，model表示模型结构体，R表示返回的隶属度矩阵，llh表示似然函数的目标值
    if abs(llh(iter)-llh(iter-1)) < tol*abs(llh(iter)); break; end
end
llh = llh(2:iter);


function R = initialization(X, init)   %X是数据矩阵，init用于初始化MoG的成分，R返回的是一个n行k列的矩阵，第ij个元素表示第i个样本由第j个成分生成的概率
n = size(X,2);                         %n是样本个数
if isstruct(init)  % init with a model %isstruct判断输入是否是一个matlab结构体
    R  = expectation(X,init);          %如果init是一个结构体，直接用该模型进行E step
elseif numel(init) == 1                %如果init是一个整数
    k = init;                          %用init表示混合成分的个数，即类别个数
    label = ceil(k*rand(1,n));         %ceil用于向数轴的正方向取整，初始化样本的label
    R = full(sparse(1:n,label,1,n,k,n));              %sparse通过记录稀疏矩阵非负元素的索引和值来节省内存，full是一相反作用；R是n行k列矩阵，n表示样本个数，k表示类别数，每一行
                                                      %是一个one-hot向量，表示该样本属于哪一类
elseif all(size(init)==[1,n])  % init with labels     %若init是一个一行n列的向量，则为样本类别的向量
    label = init;
    k = max(label);
    R = full(sparse(1:n,label,1,n,k,n));
else
    error('ERROR: init is not valid.');
end


%EM算法的E step，X表示数据矩阵，model表示模型结构体，R表示返回的隶属度矩阵，llh表示似然函数的目标值
function [R, llh] = expectation(X, model)
mu = model.mu;
Sigma = model.Sigma;
w = model.w;                           %w为MoG的混合系数向量

n = size(X,2);                         %n为样本个数
k = size(mu,2);                        %k为MoG混合成分的个数，即类别个数
R = zeros(n,k);                        %R隶属度矩阵，行数为样本个数，列数为类别个数，第ij个元素表示第i个样本由第j个成分生成的概率
for i = 1:k                            %计算样本的每个gauss概率的对数
    R(:,i) = loggausspdf(X,mu(:,i),Sigma(:,:,i));
end
R = bsxfun(@plus,R,log(w));            %计算隶属度（未归一化）矩阵的对数
T = logsumexp(R,2);                    %对R取指数加和再取对数
llh = sum(T)/n; % loglikelihood        %似然函数的均值
R = exp(bsxfun(@minus,R,T));           %计算隶属度矩阵

%EM算法的M step，X表示数据矩阵，R表示隶属度矩阵，第ij个元素表示第i个样本由第j个成分生成的概率，model是结构体，表示模型，其属性是模型的参数
function model = maximization(X, R)
[d,n] = size(X);                                    %d表示样本维数，n表示样本个数
k = size(R,2);                                      %k表示MoG成分的个数
nk = sum(R,1);                                      %nk表示求隶属度矩阵R的列和
w = nk/n;                                           %w表示混合成分系数               
mu = bsxfun(@times, X*R, 1./nk);                    %mu是一个m行k列的矩阵，表示k个高斯成分的期望，每个都是m元随机变量

Sigma = zeros(d,d,k);                               %Sigma是一个三维张量，表示第k个高斯成分的协方差矩阵是d*d的
r = sqrt(R);
for i = 1:k                                         %循环计算每个成分的协方差
    Xo = bsxfun(@minus,X,mu(:,i));
    Xo = bsxfun(@times,Xo,r(:,i)');
    Sigma(:,:,i) = Xo*Xo'/nk(i)+eye(d)*(1e-6);
end

model.mu = mu;
model.Sigma = Sigma;
model.w = w;

function y = loggausspdf(X, mu, Sigma)              %计算Gauss概率分布函数的对数的函数，输入变量分别为数据X，期望mu，期望协方差Sigma
d = size(X,1);                                      %d表示样本维数
X = bsxfun(@minus,X,mu);                            %样本与均值作差
[U,p]= chol(Sigma);                                 %chol表示将协方差矩阵Sigma进行一个上三角矩阵分解，U表示上三角因子矩阵，Sigma=U'的逆与U作积（将协方差矩阵分解求逆加快计算效率）
if p ~= 0                                           %如果p不为0则Sigma不是正定矩阵，报错
    error('ERROR: Sigma is not PD.');
end
Q = U'\X;                                           %Q=U'的逆与X的乘积
q = dot(Q,Q,1);  % quadratic term (M distance)      %dot表示点乘之后求列和
c = d*log(2*pi)+2*sum(log(diag(U)));   % normalization constant
y = -(c+q)/2;


function s = logsumexp(X, dim)
% Compute log(sum(exp(X),dim)) while avoiding numerical underflow.
%   By default dim = 1 (columns).
% Written by Mo Chen (sth4nth@gmail.com).
if nargin == 1
    % Determine which dimension sum will use
    dim = find(size(X)~=1,1);
    if isempty(dim), dim = 1; end
end

% subtract the largest in each dim
y = max(X,[],dim);
s = y+log(sum(exp(bsxfun(@minus,X,y)),dim));   % TODO: use log1p
i = isinf(y);
if any(i(:))
    s(i) = y(i);
end
