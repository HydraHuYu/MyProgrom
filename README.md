# MyProgrom

针对与轴承故障诊断的代码示例

论文参考：基于卷积神经网络的轴承故障诊断算法研究_张伟

代码参考：https://github.com/AaronCosmos/wdcnn_bearning_fault_diagnosis

复现代码为wdcnn

环境为tensorflow 1.14 keras 2.2.5 

使用adaconda集成环境

1730-1979 为cwru数据集原始mat文件

data1730-data1979 为数据切割后保存的csv文件

log文件夹为数据实时图和上次退出模型以及最优模型

init_data.py 为数据准备文件 包含画图 自动保存最优秀模型 以及 上次退出时的模型 

main.py  为wdcnn的主程序 

contrast.py 为不同负载下的比较


