% 入口函数
function main()
    % 准备初始参数
    % 定义波长范围
    wavelength = 250:1:500;
    % 定义测量数据目录路径
    dirPath = './raw_data/absorption/CDS/';
    % 定义仪器型号
    instrumentModel = 'UV3600';
    % 定义原始数据文件路径
%     filePath = './raw_data/absorption/22.4.7/22.txt';
%     filePath = './raw_data/absorption/CDS/20.txt';
    filePath = './Generate/Quad0/400/Quad1.txt';

    % 初始化测量矩阵
    % 实例化测量矩阵对象
    measurementMatrixObject = MeasurementMatrix(wavelength);
    % 设置测量矩阵值
    measurementMatrixObject.setMatrixFromFile(dirPath, instrumentModel);
    % 将吸收值转化为通过率
    measurementMatrixObject.absorption2Transmittance();
    % plot(wavelength', measurementMatrixObject.matrix);
    
    % 准备待计算数据
    % 获取有效原始数据
    validSpectralIntensityColVec = measurementMatrixObject.getValidSpectralIntensityColVecFromFile(filePath, instrumentModel);

    % 获取计算得到的测量值
    measuredIntensityColVec = measurementMatrixObject.calcMeasuredIntensityColVecFromFile(filePath, instrumentModel);
    
    % 还原
    % 获取还原光谱强度
    reconstructedSpectralIntensityColVec1 = measurementMatrixObject.restoreSpectralIntensityColVec(measuredIntensityColVec, 'LS');
    reconstructedSpectralIntensityColVec2 = measurementMatrixObject.restoreSpectralIntensityColVec(measuredIntensityColVec, 'ALM', [10, 10, 1.01, 100000, 100000, 1e-5], validSpectralIntensityColVec);
    reconstructedSpectralIntensityColVec3 = inv(dctmtx(length(wavelength))) * measurementMatrixObject.restoreSpectralIntensityColVec(measuredIntensityColVec, 'OMP', [20, 1e-6]);
    reconstructedSpectralIntensityColVec4 = measurementMatrixObject.restoreSpectralIntensityColVec(measuredIntensityColVec, 'CVX');
    reconstructedSpectralIntensityColVec5 = measurementMatrixObject.restoreSpectralIntensityColVec(measuredIntensityColVec, 'Ling');
    reconstructedSpectralIntensityColVec6 = measurementMatrixObject.restoreSpectralIntensityColVec(measuredIntensityColVec, 'LASSO');
    % disp(reconstructedSpectralIntensityColVec);
    
    % 展示相对偏差
    [mae, mape, ~, rmse] = Utils.calcRelativeDeviation(validSpectralIntensityColVec, reconstructedSpectralIntensityColVec6);
    disp(append('平均绝对误差：', num2str(mae)));
    disp(append('平均绝对百分比误差：', num2str(mape)));
    disp(append('均方根误差：', num2str(rmse)));
  
%     % 展示还原效果
    plot(wavelength', validSpectralIntensityColVec, '-', ...
         wavelength', reconstructedSpectralIntensityColVec1, '.', ...
         wavelength', reconstructedSpectralIntensityColVec2, 'o',...
         wavelength', reconstructedSpectralIntensityColVec3, '+',...
         wavelength', reconstructedSpectralIntensityColVec4, '*' ,...
         wavelength', reconstructedSpectralIntensityColVec5, 'x' ,...
         'LineWidth', 3, 'MarkerSize', 12);


    hold on;
    set(gca, 'FontSize', 20);
    xlabel('Wavelength (nm)');
    ylabel('Intensity (a.u.)');
    legend('original','LS','ALM','OMP','CVX','Ling');
%     legend('original curve', 'Gradient', 'Diff', 'location', 'Best');
%     legend('boxoff');
end