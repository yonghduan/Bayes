function accurate = bayes(mleORem,dataMissing,filename)

if nargin < 1
    filename = 'wine.mat';
    mleORem = true; %true代表mle
    dataMissing = 0.91z;
end

if nargin == 1
    dataMissing = 0.1;
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
wine_missing_data = [];
reserveIndex = 1;
array_missingIndex = 1;

for i = 1 : totalRow
    isExist = find(missingIndex == i)
    if(isempty(isExist))
        %没有找打index，证明这个数据应该放于reserve——array
        wine_reserved_array(reserveIndex,:) = wine_array(i,:);
        reserveIndex = reserveIndex + 1;
    else
        wine_missing_data(array_missingIndex,:) = wine_array(i,2:end);
        array_missingIndex = array_missingIndex + 1;
    end
end

%得到两个矩阵，reserved——array和missing——data，首先对reserved——array进行高斯分布参数估计
wine_reserved_data = wine_reserved_array(:,2:end);
attributeNumber = size(wine_reserved_data,2);
wine_reserved_label = wine_reserved_array(:,1);

reservedLabelProbability = tabulate(wine_reserved_label);
%当前每个分类的样本个数
reserved_flag_1 = reservedLabelProbability(1,2);
reserved_flag_2 = reservedLabelProbability(2,2);
reserved_flag_3 = reservedLabelProbability(3,2);

reserved_Probability_1 = zeros(2,attributeNumber);
reserved_Probability_2 = zeros(2,attributeNumber);
reserved_Probability_3 = zeros(2,attributeNumber);

reserved_classification1_array = wine_reserved_array(1:reserved_flag_1,:);
reserved_classification2_array = wine_reserved_array((reserved_flag_1+1):(reserved_flag_1+reserved_flag_2),:);
reserved_classification3_array = wine_reserved_array((reserved_flag_1+ reserved_flag_2+1): ...
    (reserved_flag_1+reserved_flag_2+reserved_flag_3),:);

for i=1:3
    for j=1:attributeNumber
       if i==1
           data=wine_reserved_data(1:reserved_flag_1,j);
           [mu,sigma] = normfit(data);
           reserved_Probability_1(1,j)= mu;
           reserved_Probability_1(2,j) = sigma;
       end

       if i==2
           data=wine_reserved_data((reserved_flag_1+1):(reserved_flag_1+reserved_flag_2),j);
           [mu,sigma] = normfit(data);
           reserved_Probability_2(1,j)= mu;
           reserved_Probability_2(2,j) = sigma;
       end

       if i==3
           data=wine_reserved_data((reserved_flag_1+ reserved_flag_2+1) : (reserved_flag_1+reserved_flag_2+reserved_flag_3),j);
           [mu,sigma] = normfit(data);
           reserved_Probability_3(1,j)= mu;
           reserved_Probability_3(2,j) = sigma;
       end

    end
end

%得到了reserved_Probability_1,2,3，模型数据
%将缺失的数据分别带入模型，判断属于哪个模型便将它带入模型，更新模型参数
%
predictData = wine_missing_data;
for row = 1 : missingNumber
    resevedTotalNumber = reserved_flag_1 + reserved_flag_2 + reserved_flag_3;
    P_y1 = reserved_flag_1 / resevedTotalNumber / 100;
    P_y2 = reserved_flag_2 / resevedTotalNumber / 100;
    P_y3 = reserved_flag_3 / resevedTotalNumber / 100;


    P_xy1=1;
    P_xy2=1;
    P_xy3=1;
    
    for i=1:attributeNumber
        P_xy1=P_xy1 * normpdf(predictData(row,i),reserved_Probability_1(1,i),reserved_Probability_1(2,i));
        P_xy2=P_xy2 * normpdf(predictData(row,i),reserved_Probability_2(1,i),reserved_Probability_2(2,i));
        P_xy3=P_xy3 * normpdf(predictData(row,i),reserved_Probability_3(1,i),reserved_Probability_3(2,i));
    end
    
    P_xy1P_y1=P_xy1 * P_y1;
    P_xy2P_y2=P_xy2 * P_y2;
    P_xy3P_y3=P_xy3 * P_y3;
    result = [P_xy1P_y1,P_xy2P_y2,P_xy3P_y3];
    [B,index] = sort(result);
    if index(3) == 1
        disp("classification 1");
        reserved_flag_1 = reserved_flag_1 + 1;
        reserved_classification1_array(reserved_flag_1,1) = 1;
        reserved_classification1_array(reserved_flag_1,2:end) = predictData(row,:);
        for j = 1 : attributeNumber
            [mu,sigma] = normfit(reserved_classification1_array(:,j + 1));
            reserved_Probability_1(1,j)= mu;
            reserved_Probability_1(2,j) = sigma;
        end
    elseif index(3) == 2
        disp("classification 2");
        reserved_flag_2 = reserved_flag_2 + 1;
         reserved_classification2_array(reserved_flag_2,1) = 2;
        reserved_classification2_array(reserved_flag_2,2:end) = predictData(row,:);
        for j = 1 : attributeNumber
            [mu,sigma] = normfit(reserved_classification2_array(:,j + 1));
            reserved_Probability_2(1,j)= mu;
            reserved_Probability_2(2,j) = sigma;
        end
    else
        disp("classification 3");
        reserved_flag_3 = reserved_flag_3 + 1;
        reserved_classification3_array(reserved_flag_3,1) = 3;
        reserved_classification3_array(reserved_flag_3,2:end) = predictData(row,:);
        for j = 1 : attributeNumber
            [mu,sigma] = normfit(reserved_classification3_array(:,j + 1));
            reserved_Probability_3(1,j)= mu;
            reserved_Probability_3(2,j) = sigma;
        end
    end
end




















%使用测试集测试
%分别计算在y1，y2，y3下的概率，
%首先使用全集开始测试
predictData=wine_array;
rowNumber = size(predictData,1);
errorCount = 0;

wine_label = wine_array(:,1);
lableProbability = tabulate(wine_label);

P_y1 = lableProbability(1,3) / 100;
P_y2 = lableProbability(2,3) / 100;
P_y3 = lableProbability(3,3) / 100;


for row = 1 : rowNumber
    rightLabel = predictData(row,1);
    P_xy1=1;
    P_xy2=1;
    P_xy3=1;
    
    for i=1:attributeNumber
        P_xy1=P_xy1 * normpdf(predictData(row,i+1),reserved_Probability_1(1,i),reserved_Probability_1(2,i));
        P_xy2=P_xy2 * normpdf(predictData(row,i+1),reserved_Probability_2(1,i),reserved_Probability_2(2,i));
        P_xy3=P_xy3 * normpdf(predictData(row,i+1),reserved_Probability_3(1,i),reserved_Probability_3(2,i));
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

accurate = (rowNumber - errorCount) / rowNumber;
disp("accurate = " + accurate);
end