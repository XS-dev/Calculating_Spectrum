% 工具类
classdef Utils
    
    % 静态方法
    methods (Static)
        
        % 创建一维梯度矩阵
        function out = create1DGradientMat(mRows)
            % region
            % 判断输入是否为正数
            if mRows > 0
                % 判断输入行数是否为1
                if mRows == 1
                    % 返回0
                    out = 0;
                else
                    % 创建元素全为0的梯度矩阵
                    gradientMat = zeros(mRows);
                    % 循环修改各行元素
                    for index = 1:mRows
                        % 判断是否为第一行
                        if index == 1
                            % [-1, 1, 0, 0, ...]
                            gradientMat(index, index) = -1;
                            gradientMat(index, index + 1) = 1;
                        % 判断是否为最后一行
                        elseif index == mRows
                            % [0, 0, ..., -1, 1]
                            gradientMat(index, index - 1) = -1;
                            gradientMat(index, index) = 1;
                        % 中间行
                        else
                            % [0, ..., -0.5, 0, 0.5, 0, ...]
                            gradientMat(index, index - 1) = -0.5;
                            gradientMat(index, index + 1) = 0.5;
                        end
                    end
                    % 返回梯度矩阵
                    out = gradientMat;
                end
            else
                % 输入参数非正数，抛出异常
                throw(MException('MATLAB:wrongInput', 'not input positive value'));
            end
            % endregion
        end

        % 创建一维差分矩阵
        function out = create1DDiffMat(mRows)
            % region
            % 判断输入是否为正数
            if mRows > 0
                % 判断输入行数是否为1
                if mRows == 1
                    % 返回0
                    out = 0;
                else
                    % 创建元素全为0的梯度矩阵
                    gradientMat = zeros(mRows);
                    % 循环修改各行元素
                    for index = 1:mRows
                        % 判断是否为最后一行
                        if index == mRows
                            % [0, 0, ..., -1]
                            gradientMat(index, index) = -1;
                        % 中间行
                        else
                            % [0, ..., -1, 1, 0, ...]
                            gradientMat(index, index) = -1;
                            gradientMat(index, index + 1) = 1;
                        end
                    end
                    % 返回梯度矩阵
                    out = gradientMat;
                end
            else
                % 输入参数非正数，抛出异常
                throw(MException('MATLAB:wrongInput', 'not input positive value'));
            end
            % endregion
        end

        % 计算相对偏差
        function [mae, mape, mse, rmse] = calcRelativeDeviation(referenceMat, observedMat)
            % region
            % 计算平均绝对误差
            mae = mean(abs(observedMat - referenceMat));
            % 计算平均绝对百分比误差（真实值不能有0）
            mape = mean(abs((observedMat - referenceMat) ./ referenceMat));
            % 计算均方误差
            mse = mean((observedMat - referenceMat) .^ 2);
            % 计算均方根误差
            rmse = sqrt(mean((observedMat - referenceMat) .^ 2));
            % endregion
        end

    end

end