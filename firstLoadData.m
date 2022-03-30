function firstLoadData(filename)
clc;

if(nargin < 1)
    filename = "wine.data";
end

uiimport(filename);

%加载变量在工作区后，可以利用save函数将数据保存在.mat文件中，这样后面
%就可以直接使用load函数加载变量
end