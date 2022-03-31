function accurate = bayes(mleORem,dataMissing,filename)

if nargin < 1
    filename = 'wine.mat';
    mleORem = true; %true代表mle
    dataMissing = 0;
end

if nargin == 1
    dataMissing = 0;
    filename = 'wine.mat';
end

if nargin == 2
    filename = 'wine.mat';
end

load(filename);

wine_array = table2array(wine);
totalRow = size(wine_array,1);

dataReserve = 1 - dataMissing;
reservedNumber = uint16(totalRow * dataReserve);
missingNumber = totalRow - reservedNumber;

missingIndex = randperm(totalRow,missingNumber);

wine_reserved_array = [];
wine_missing_array = [];

for i = 1 : totalRow
    isExist = find(missingIndex == )


wine_table_label=wine(:,1);
wine_table_data=wine(:,2:end);
wine_label=table2array(wine_table_label);
wine_data = table2array(wine_table_data);

attributeNumber = size(wine_data,2);
lableProbability = tabulate(wine_label);

flag_1=lableProbability(1,2);
flag_2=lableProbability(2,2);
flag_3=lableProbability(3,2);

P_y1=lableProbability(1,3)/100;
P_y2=lableProbability(2,3)/100;
P_y3=lableProbability(3,3)/100;

%计算每个分类下不同属性的高斯分布模型，使用mle参数估计法估计出高斯分布的均值和
%标准差。测试数据中以此模型来估计它的概率
probability_1 = zeros(2,attributeNumber);
probability_2 = zeros(2,attributeNumber);
probability_3 = zeros(2,attributeNumber);

for i=1:3
    for j=1:attributeNumber
       if i==1
           data=wine_data(1:flag_1,j);
           phat = mle(data);
           probability_1(1,j)=phat(1);
           probability_1(2,j)=phat(2);
       end

       if i==2
           data=wine_data(flag_1+1:flag_1+flag_2,j);
           phat = mle(data);
           probability_2(1,j)=phat(1);
           probability_2(2,j)=phat(2);
       end

       if i==3
           data=wine_data(flag_1+flag_2+1:flag_1+flag_2+flag_3,j);
           phat=mle(data);
           probability_3(1,j)=phat(1);
           probability_3(2,j)=phat(2);
       end

    end
end

%使用测试集测试
%分别计算在y1，y2，y3下的概率，
%首先使用全集开始测试
predictData=wine_array;
rowNumber = size(wine_array,1);
errorCount = 0;

for row = 1 : rowNumber
    rightLabel = predictData(row,1);
    P_xy1=1;
    P_xy2=1;
    P_xy3=1;
    
    for i=1:attributeNumber
        P_xy1=P_xy1 * normpdf(predictData(row,i),probability_1(1,i),probability_1(2,i));
        P_xy2=P_xy2 * normpdf(predictData(row,i),probability_2(1,i),probability_2(2,i));
        P_xy3=P_xy3 * normpdf(predictData(row,i),probability_3(1,i),probability_3(2,i));
    end
    
    P_xy1P_y1=P_xy1 * P_y1;
    P_xy2P_y2=P_xy2 * P_y2;
    P_xy3P_y3=P_xy3 * P_y3;
    result = [P_xy1P_y1,P_xy2P_y2,P_xy3P_y3];
    [B,index] = sort(result);
    if index(3) == 1
        disp("classification 1");
        if rightLabel ~= 1
            errorCount = errorCount + 1;
        end
    elseif index(3) == 2
        disp("classification 2");
         if rightLabel ~= 2
            errorCount = errorCount + 1;
        end
    else
        disp("classification 3");
         if rightLabel ~= 3
            errorCount = errorCount + 1;
        end
    end
end

accurate = errorCount / rowNumber;
disp("accurate = " + accurate);
end