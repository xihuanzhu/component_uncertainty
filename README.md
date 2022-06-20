1.本项目是关于块级不确定计算的研究

2.有效文件说明：

    helper文件夹功能为将提供数据转换成项目所需数据格式，
    
    data_revise.ipynb检查数据是否正常
    
    extract_metrics.py用于提取接近像素级小块的metric
    
    global_defs.py用于全局定义，包括训练数据，验证数据，以及其他指标文件目录
    
    metaseg_eval.py为主要训练文件
    
    metaseg_io.py为各种文件的读取和保存
    
    metaseg_plot.py用于画各种曲线（如回归曲线）
    
    metrics.pyx计算块级（原始分割块）的指标
    
    expe6文件夹存储指标：
    
        pixel_train_measure:像素级小块训练集指标
        
        components：利用分割mask生成用于区分分割块的数组
        
        metrics:利用分割mask提取的指标，并用于回归实验
        
        pixel_val_measure:像素级小块验证集指标
        
        train_measure:块级（原始分割块）训练集指标
        
        val_measure:块级（原始分割块）验证集指标

3.数据（body）：

    训练集：/raid/zhuxihuan/data/body/expe6/train_data/
    
    验证集：/raid/zhuxihuan/data/body/expe6/val_data/
    
    测试集：/raid/zhuxihuan/data/body/expe6/val_data/

4.流程：

    1）数据转换：用helper中transform文件转换数据格式为特定数据格式
    
    2）运行python metrics_setup.py build_ext --inplace
    
    3）计算指标：
        训练指标：
            1.修改global_defs.py中METASEG_READ_DATA_PATH和METASEG_MY_IO_PATH(前者为输入目录，后者为输出目录)，并设置NUM_IMAGES为训练集图片数量，
            设置COMPUTE_METRICS为True，ANALYZE_METRICS为False
            
            2.python metaseg_eval.py --NUM_CORES=10 --NUM_LASSO_LAMBDAS=50
            
        验证指标：类似训练指标，修改文件目录即可
        
    4）计算像素级小块指标，必须基于3）生成的conponent，这一步是对3）的改进，非必须步骤
        1.修改global_defs.py中METASEG_MY_PIXEL_IO_PATH
        
        2.修改metaseg_eval.py,将compute_metrics_i调用的地方换成compute_pixel_metrics_i
        
        3.后续和3）步骤一样
        
    5）训练
        1.修改global_defs.py，设置COMPUTE_METRICS为False，ANALYZE_METRICS为True
        
        2.python metaseg_eval.py --NUM_CORES=10 --NUM_LASSO_LAMBDAS=50



