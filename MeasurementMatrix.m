% 测量矩阵类
classdef MeasurementMatrix < handle

    % 外部只读属性
    properties (SetAccess = private)
        % 波长范围
        wavelengthColVec
        % 矩阵值
        matrix
    end

    % 外部公开使用的接口函数
    methods (Access = public)

        % 构造函数MeasurementMatrix()
        function obj = MeasurementMatrix(inWavelengthVec, inMatrix)
            % region
            % 判断是否存在必要参数：波长向量inWavelengthVec
            if nargin >= 1
                % 判断输入类型
                if isWavelengthVec(inWavelengthVec)
                    % 判断是否为行向量
                    if size(inWavelengthVec, 1) == 1
                        % 对行向量进行转置
                        obj.wavelengthColVec = inWavelengthVec.';
                    else
                        obj.wavelengthColVec = inWavelengthVec;
                    end
                end
                % 判断是否输入矩阵值
                if nargin == 2
                    % 判断输入矩阵列数是否与波长列向量长度一致
                    if size(inMatrix, 2) == length(obj.wavelengthColVec)
                        obj.matrix = inMatrix;
                    else
                        % 矩阵列数与波长列向量个数不一致，抛出异常
                        throw(MException('MATLAB:inconsistentMatrixDims', 'matrix column number inconsistent with wavelength vector number'));
                    end
                end
            else
                % 缺少必要参数，抛出异常
                throw(MException('MATLAB:wrongInputNumber', 'miss required in arguments.'));
            end
            % endregion
        end

        % 获取文件中满足类属性定义的波长范围的光谱强度
        function out = getValidSpectralIntensityColVecFromFile(obj, inFilePath, inInstrumentModel)
            % region
            % 读取光谱强度-波长矩阵
            fileSpectralIntensityVersusWavelengthMat = readSpectralIntensityVersusWavelengthMat(inFilePath, inInstrumentModel);
            % 提取类属性定义的波长范围对应的光谱强度
            out = obj.extractValidSpectralIntensityColVec(fileSpectralIntensityVersusWavelengthMat);
            % endregion
        end

        % 从文件中设置矩阵值
        function setMatrixFromFile(obj, inDirPath, inInstrumentModel)
            % region
            % 根据设备型号从目录路径获取文件列表
            fileList = getFileList(inDirPath, inInstrumentModel);
            % 循环遍历文件列表
            for index = 1:length(fileList)
                % 输入列表元素为文件结构体，通过拼接字符串获得文件路径
                filePath = append(fileList(index).folder, '/', fileList(index).name);
                validSpectralIntensityColVec = obj.getValidSpectralIntensityColVecFromFile(filePath, inInstrumentModel);
                % 将光谱强度列向量转置成行向量后与对象的矩阵属性竖直合并
                obj.matrix = vertcat(obj.matrix, validSpectralIntensityColVec.');
            end
            % endregion
        end

        % 吸收值转化为透过率，T=1/(10^A)
        function absorption2Transmittance(obj)
            % region
            obj.matrix = 1 ./ (10 .^ obj.matrix);
            % endregion
        end

        % 透过率转化为吸收值，A=lg(1/T)
        function transmittance2Absorption(obj)
            % region
            obj.matrix = log10(1 ./ obj.matrix);
            % endregion
        end

         % 根据光谱强度计算测量强度列向量
         function out = calcMeasuredIntensityColVec(obj, inSpectralIntensityVersusWavelengthMat)
            % region
            % 提取类属性定义的波长范围对应的光谱强度
            validSpectralIntensityColVec = obj.extractValidSpectralIntensityColVec(inSpectralIntensityVersusWavelengthMat);
            % 计算测量强度并返回
            out = obj.matrix * validSpectralIntensityColVec;
%             plot(out);
%             hold on;
%             set(gca, 'FontSize', 20);
%             xlabel('Wavelength (nm)');
%             ylabel('Intensity (a.u.)');
            % endregion
        end

        % 从文件中根据光谱强度计算测量强度列向量
        function out = calcMeasuredIntensityColVecFromFile(obj, inFilePath, inInstrumentModel)
            % region
            % 读取光谱强度-波长矩阵
            fileSpectralIntensityVersusWavelengthMat = readSpectralIntensityVersusWavelengthMat(inFilePath, inInstrumentModel);
            % 计算测量强度并返回
            out = obj.calcMeasuredIntensityColVec(fileSpectralIntensityVersusWavelengthMat);
            % endregion
        end

        % 根据测量强度还原光谱强度
        function out = restoreSpectralIntensityColVec(obj, inMeasuredIntensityColVec, inAlgorithm, inAlgorithmParameterVec, x0)
            % region
            % 判断算法
            switch inAlgorithm
                case 'LS'
                    % 调用最小二乘法线性回归算法
                    out = obj.solveSpectralIntensityColVecByLS(inMeasuredIntensityColVec);
                case 'CVX'
                    %调用CVX包
                    out = obj.solveWithCVX(inMeasuredIntensityColVec);
                case 'Ling'
                    %调用Ling回归
                    out = obj.solveWithLing(inMeasuredIntensityColVec);
                case 'LASSO'
                    %调用Lasso回归
                    out = obj.solveWithLasso(inMeasuredIntensityColVec);
                     case 'ALM'
                    % 调用增广拉格朗日乘子算法
                    out = obj.solveSpectralIntensityColVecByALM(inMeasuredIntensityColVec, inAlgorithmParameterVec, x0);
                case 'OMP'
                    % 调用OMP算法
                    out = obj.solveSpectralIntensityColVecByOMP(inMeasuredIntensityColVec, inAlgorithmParameterVec);

                case 'ADMM'
                    % 调用ADMM算法
                    out = obj.solveSpectralIntensityColVecByADMM(inMeasuredIntensityColVec);
                  
            end
            % endregion
        end
    
    end

    % 仅类内部使用的、需要用到类属性的工具函数
    methods (Access = private)

        % 根据类属性定义的波长范围提取光谱强度
        function out = extractValidSpectralIntensityColVec(obj, inSpectralIntensityVersusWavelengthMat)
            % region
            % 读取波长列向量和光谱强度列向量
            inWavelengthColVec = inSpectralIntensityVersusWavelengthMat(:, 1);
            inSpectralIntensityColVec = inSpectralIntensityVersusWavelengthMat(:, 2);
            % 获取对象的波长属性元素在文件的波长向量中是否存在及索引
            [wavelengthIsExistColVec, indexColVec] = ismember(obj.wavelengthColVec, inWavelengthColVec);
            % 判断对象的波长属性是否全部存在于输入中
            if all(wavelengthIsExistColVec)
                % 返回对应的光谱强度
                out = inSpectralIntensityColVec(indexColVec);
            else
                % 不满足对象定义的波长范围，抛出异常
                throw(MException('MATLAB:lackWavelengthValue', 'file not contain all the wavelengths required'));
            end


            % endregion
        end

         % 增广拉格朗日乘子算法，求解：min(||g||_1) s.t. Gx=g, Ax=b => min L(x)
        function out = solveSpectralIntensityColVecByALM(obj, inMeasuredIntensityColVec, inAlgorithmParameterVec, x0)
            % region
            % 定义算法所需参数
            % 测量矩阵
            A = obj.matrix;
            % 测量值
            b = inMeasuredIntensityColVec;
            % 梯度计算矩阵
            % G = Utils.create1DGradientMat(length(obj.wavelengthColVec));
            G = Utils.create1DDiffMat(length(obj.wavelengthColVec));
            % 初始惩罚因子
            mu_1 = inAlgorithmParameterVec(1);
            mu_2 = inAlgorithmParameterVec(2);
            % 最大惩罚因子系数（后续考虑在代码中加入初始值）
            rho = inAlgorithmParameterVec(3);
            mu_1max = inAlgorithmParameterVec(4);
            mu_2max = inAlgorithmParameterVec(5);
            % 初始拉格朗日乘子
            y_1 = 0;
            y_2 = 0;
            % 收敛判断常数（后续考虑在代码中加入初始值）
            epsilon = inAlgorithmParameterVec(6);
            % 待求参数
            x = abs(sin(obj.wavelengthColVec ./ 30));
            % x = 0.4 .* ones(length(obj.wavelengthColVec), 1);
            % x = x0;
            % 迭代运算
            % % 初始化循环判断参数
            % deltax = 1;
            % 判断循环次数变量
            times = 0;
            % 循环判断是否收敛
            % 思考：如何判断向量是否收敛？x需不需要设置初始值？
            while 1
                times = times + 1;
                % % 存储旧x值fs
                % x_old = x;
                % 迭代g
                % 分段函数
                g = ((G * x + y_1 / mu_1) - 1 / mu_1) .* ((G * x + y_1 / mu_1) > (1 / mu_1)) + ((G * x + y_1 / mu_1) + 1 / mu_1) .* ((G * x + y_1 / mu_1) < (1 / -mu_1));
                % 迭代x
                x = (mu_1 .* (G' * G) + mu_2 .* (A' * A)) \ (mu_1 .* (G' * (g - y_1 / mu_1)) + mu_2 .* (A' * (b - y_2 / mu_2)));
                % x = inv(mu_1 .* (G' * G) + mu_2 .* (A' * A)) * (mu_1 .* (G' * (g - y_1 / mu_1)) + mu_2 .* (A' * (b - y_2 / mu_2)));
                % 判断约束违反度是否满足要求
                % if all(abs(G * x - g) < epsilon) && all(abs(A * x - b) < epsilon) 
                if norm(G * x - g) + norm(A * x - b) < epsilon
                    disp(append('迭代次数：', num2str(times)));
                    out = x;
                    return;
                else
                    % 迭代y1
                    y_1 = mu_1 .* (G * x - g) + y_1;
                    % 迭代y2
                    y_2 = mu_2 .* (A * x - b) + y_2;
                    % 迭代μ1
                    mu_1 = rho * mu_1;
                    % mu_1 = min(rho * mu_1, mu_1max);
                    % 迭代μ2
                    mu_2 = rho * mu_2;
                    % mu_2 = min(rho * mu_2, mu_2max);
                end
                % % 计算迭代程度
                % deltax = x - x_old
            end
            % 返回迭代结果
            % out = x;
            % endregion
        end

        % OMP算法
        function [x] = solveSpectralIntensityColVecByOMP(obj, inMeasuredIntensityColVec, inAlgorithmParameterVec)
            index = [];
            k = 1;
            A = obj.matrix * dctmtx(length(obj.wavelengthColVec));
            b = inMeasuredIntensityColVec;
            [Am, An] = size(A);
            r = b;
            x = zeros(An, 1);
            cor = A' * r;
            sparsity = inAlgorithmParameterVec(1);
            thershold = inAlgorithmParameterVec(2);
            while norm(r) > thershold && norm(A' * r, Inf) > thershold
                [Rm, ind] = max(abs(cor));
                index = [index ind];
                P = A(:, index) * inv(A(:, index)' * A(:, index)) * A(:, index)';
                r = (eye(Am) - P) * b;
                cor = A' * r;
                k = k + 1;
            end
            xind = inv(A(:, index)' * A(:, index)) * A(:, index)' * b;
            x(index) = xind;
        end

        % 最小二乘法线性回归算法，求解min(||Ax-b||_2)^2 s.t. Ax=b
        function out = solveSpectralIntensityColVecByLS(obj, inMeasuredIntensityColVec)
            % region
            % 定义二次规划函数所需参数
            % H=2A'A
            H = 2 .* ((obj.matrix') * obj.matrix);
            % f=-2A'b
            f = -2 .* ((obj.matrix') * inMeasuredIntensityColVec);
            % 不等式约束条件：Ax<=b
            A = obj.matrix;
            b = inMeasuredIntensityColVec;
            % 等式约束条件：A_eqx=beq
            Aeq = obj.matrix;
            beq = inMeasuredIntensityColVec;
            % 二次规划函数
            out = quadprog(H, f, A, b, Aeq, beq);
            % endregion
        end

        % ADMM算法
        function [x] = solveSpectralIntensityColVecByADMM(obj, inMeasuredIntensityColVec, inAlgorithmParameterVec)
            index = [];
            k = 1;
            A = obj.matrix * dctmtx(length(obj.wavelengthColVec));
            b = inMeasuredIntensityColVec;
            [Am, An] = size(A);
            r = b;
            x = zeros(An, 1);
            cor = A' * r;
            sparsity = inAlgorithmParameterVec(1);
            thershold = inAlgorithmParameterVec(2);
            while norm(r) > thershold && norm(A' * r, Inf) > thershold
                [Rm, ind] = max(abs(cor));
                index = [index ind];
                P = A(:, index) * inv(A(:, index)' * A(:, index)) * A(:, index)';
                r = (eye(Am) - P) * b;
                cor = A' * r;
                k = k + 1;
            end
            xind = inv(A(:, index)' * A(:, index)) * A(:, index)' * b;
            x(index) = xind;
        end


        % 使用CVX包
        function out = solveWithCVX(obj, inMeasuredIntensityColVec)
            A = obj.matrix;
            b = inMeasuredIntensityColVec;
            C = obj.matrix;
            d = inMeasuredIntensityColVec;
            n = size(obj.wavelengthColVec);
            cvx_begin
                variable x(n)
                minimize( norm( A * x - b, 2 ) )
                subject to
                    C * x == d
            cvx_end
            out  = x;
        end

        % 使用岭回归
        function out = solveWithLing(obj, inMeasuredIntensityColVec)
%             n = size(obj.wavelengthColVec);
            x=obj.matrix;    
            y=inMeasuredIntensityColVec;       
            k1=[0.1:0.1:10];

            temp = 100;
            wucha = zeros(length(k1));
            for k=1:length(k1)
                %B = ridge(y,X,k,scaled)
                %k为岭参数，scaled为标准化的范围
                B = ridge(y,x,k1(k),0);
                if(k == 1)
                    out  = B(2:end);
                end
                
                A=B;
                yn= A(1)+x*A(2:end);
                wucha(k)=sum(abs(y-yn)./y)/length(y);
                if(temp >= wucha(k))
                    temp = wucha(k);
                     index = k1(k);
                    out  = B(2:end);
                end
            end
            plot(k1,wucha);
            plot(out);
        end


         % 使用LASSO回归
         function out = solveWithLasso(obj, inMeasuredIntensityColVec)
             x=obj.matrix;
             y=inMeasuredIntensityColVec;
             x1=[ones(size(x,1),1),x];
             lamda=0.0001;
%              beta=ones(size(x1,2),1);%回归系数
%              % 坐标轴下降法求解
%              epochs=1000;%迭代次数
%              learnrate=0.001;%学习率

%              for j=1:epochs
%                  beta1=beta;%记录上一轮的bata
%                  for i=1:length(beta)
%                      for n=1:epochs
%                          %找到让损失函数收敛的点
%                          yn=x1*beta;
%                          J=x1(:,i)'*(yn-y)/length(y)+lamda*sign(beta(i));
%                          beta(i)=beta(i)-J*learnrate;
%                          if (abs(J)<1e-3)
%                              break
%                          end
%                      end
%                  end
%                  if(sum(abs(beta1-beta)<1e-3)==sum(length(beta)))
%                      %      if((abs(beta1-beta)<1e-3))
%                      break
%                  end
%              end
%             y_p=x1*beta;
%             wucha=sum(abs(y_p-y)./y)/length(y);
%             out = x1;

            %使用MATLAB自带的LASSO函数
            [B,FitInfo] = lasso(x,y, 'Lambda',lamda);
            y_p1=x*B+FitInfo.Intercept;
            wucha1=sum(abs(y_p1-y)./y)/length(y);
            out = B;

            plot(wucha1);
            plot(out);
         end
    end

end

% 仅类内部使用的、不需要用到类属性的工具函数
% 判断是否为波长向量
function out = isWavelengthVec(inWavelengthVec)
    % region
    % 判断输入是否为向量
    if isvector(inWavelengthVec)
        % 判断输入是否为正
        if all(inWavelengthVec > 0)
            % 返回true
            out = true;
        else
            % 不是正向量，抛出异常
            throw(MException('MATLAB:wrongInput', 'not input positive value'));
        end
    else
        % 不是向量，抛出异常
        throw(MException('MATLAB:wrongInput', 'not input a vector'));
    end
    % endregion
end

% 获取文件列表
function out = getFileList(inDirPath, inInstrumentModel)
    % region
    % 判断路径尾部是否存在'/', 不存在则添加
    if ~endsWith(inDirPath, '/')
        inDirPath = append(inDirPath, '/');
    end
    % 判断仪器类型
    switch inInstrumentModel
        case 'UV3600'
            out = dir(append(inDirPath, '*.txt'));
    end
    % endregion
end

% 从文件列表中读取波长和光谱强度
function out = readSpectralIntensityVersusWavelengthMat(inFilePath, inInstrumentModel)
    % region
    % 读取文件输入
    % 打开文件
    fileId = fopen(inFilePath, 'r');
    % 判断仪器类型
    switch inInstrumentModel
        case 'UV3600'
            % 读取txt数据，返回1*2的元胞数组，其中第一列为波长向量，第二列为测量值向量
            fileCell = textscan(fileId, '%f %f', 'Delimiter', ',', 'HeaderLines', 2);
    end
    % 关闭文件
    fclose(fileId);
    % 将元胞数组转化为矩阵后导出
    out = cell2mat(fileCell);
    % endregion
end
