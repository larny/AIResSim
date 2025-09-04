# Jupyter Notebook 使用指南
## 储层仿真机器学习工作流

### 📖 概述
`reservoir_ml_workflow.ipynb` 是一个完整的储层仿真机器学习工作流notebook，展示了从数据导入到模型训练的全过程。

### 🚀 快速开始

#### 1. 环境准备
确保您已安装以下Python包：
```bash
pip install torch torchvision torchaudio
pip install matplotlib numpy pandas seaborn
pip install torch-geometric  # 可选，用于高级图神经网络
```

#### 2. 数据准备
确保HM文件夹包含以下文件：
- `HM.INIT` - 初始压力和饱和度
- `HM_PERM_I.GSG`, `HM_PERM_J.GSG`, `HM_PERM_K.GSG` - 渗透率
- `HM_POROSITY.GSG` - 孔隙度
- `HM.GSG` - 网格坐标
- `HM_WELL_CONNECTIONS.ixf` - 井射孔数据
- `HM_PRED_FM.ixf` - 井控制条件

#### 3. 运行Notebook
在Jupyter环境中打开 `reservoir_ml_workflow.ipynb` 并按顺序执行每个cell。

### 📊 工作流步骤详解

#### Cell 1: 库导入
- 导入PyTorch、matplotlib、numpy等必要库
- 导入自定义储层仿真模块
- 设置绘图参数和设备配置

#### Cell 2: 数据初始化  
- 设置案例名称和数据路径
- 初始化OptimizedReservoirSimulationWorkflow
- 配置模型超参数

#### Cell 3: 数据加载
- 加载储层二进制数据文件
- 提取8个关键通道的特征
- 构建ACTNUM优化图结构
- 显示数据统计信息

#### Cell 4: 静态属性可视化
- 将活跃网格数据转换为3D场
- 可视化压力、饱和度分布
- 在图上标记井位置
- 支持任意Z层的2D可视化

#### Cell 5: 图数据准备
- 转换为PyTorch张量格式
- 准备节点特征、边索引、边属性
- 显示图结构统计信息
- 检查特征分布和归一化

#### Cell 6: 模型定义
- 定义ReservoirGNN类（图神经网络）
- 包含编码器、GCN层、解码器
- 支持残差连接和批归一化

#### Cell 7: 模型训练
- 初始化GNN模型和优化器
- 执行训练循环（20个epoch）
- 可视化训练损失曲线
- 显示模型参数统计

#### Cell 8: 预测和可视化
- 使用训练好的模型进行预测
- 可视化预测结果vs原始数据
- 计算和显示绝对误差、相对误差
- 生成误差分布图

#### Cell 9: 耦合仿真
- 运行完整的GNN-FNO耦合仿真
- 可视化井性能时间序列
- 分析总产量、井底流压趋势
- 显示压力场演化

#### Cell 10: 多层分析
- 展示多个Z层的属性分布
- 对比不同层位的压力、饱和度
- 分析预测结果的空间分布

#### Cell 11: 量化分析
- 计算MAE、RMSE、MAPE、R²指标
- 生成预测精度散点图
- 对比不同属性的预测误差
- 分析井性能预测结果

#### Cell 12: 结果总结
- 保存训练好的模型
- 生成综合分析报告
- 总结技术成就和建议

### 🔧 自定义使用

#### 修改案例数据
在Cell 2中修改：
```python
CASE_NAME = "YOUR_CASE"  # 您的案例名
DATA_DIR = "/path/to/your/data"  # 您的数据路径
```

#### 调整模型参数
在Cell 2中修改config：
```python
workflow.config.update({
    'gnn_hidden_dim': 128,  # 增加隐藏层维度
    'gnn_num_layers': 6,    # 增加网络深度
    'learning_rate': 0.0005  # 调整学习率
})
```

#### 选择可视化层位
在Cell 4和10中修改：
```python
middle_layer = 5  # 选择第6层
layers_to_plot = [0, 3, 6, 9]  # 自定义层位
```

#### 扩展训练
在Cell 7中增加训练轮数：
```python
num_epochs = 100  # 增加到100轮
```

### 📊 输出文件

工作流执行后会生成以下文件：
- `/workspace/trained_models/reservoir_gnn_model.pth` - 训练好的模型
- `/workspace/final_workflow_summary.txt` - 综合分析报告
- 各种可视化图片（如果保存）

### 🎯 关键功能

#### 2D可视化功能
- 支持任意Z层的属性分布可视化
- 自动标记井位置
- 支持压力、饱和度、孔隙度、渗透率

#### 3D分析功能  
- 多层属性分布对比
- 空间误差分布分析
- 活跃网格映射

#### 误差量化功能
- 绝对误差和相对误差计算
- MAE、RMSE、MAPE、R²指标
- 空间和时间误差趋势分析

#### 井性能分析
- 单井日产油、日产液、日产水
- 井底流压时间序列
- 总产量和含水率分析

### 💡 使用技巧

1. **逐步执行**: 建议按顺序执行每个cell，观察输出结果
2. **参数调优**: 可以修改模型参数重新训练对比效果
3. **层位选择**: 根据储层特征选择感兴趣的Z层进行分析
4. **误差分析**: 重点关注误差分布图，识别模型预测的薄弱环节
5. **井性能**: 分析井产量趋势，评估开发策略

### ⚠️ 注意事项

1. **内存使用**: 大规模储层可能需要大量内存，建议使用GPU
2. **数据格式**: 确保二进制文件格式正确
3. **ACTNUM**: 活跃网格映射对模型性能影响很大
4. **收敛性**: 监控训练损失，确保模型收敛
5. **物理合理性**: 检查预测结果是否符合储层物理规律

### 🔗 相关文件

- `data_parser.py` - 数据解析模块
- `optimized_coupling_workflow.py` - 主工作流
- `actnum_handler.py` - 活跃网格处理
- `visualization.py` - 可视化工具
- `quantitative_analysis.py` - 量化分析工具

### 📞 技术支持

如有问题，请检查：
1. 数据文件是否完整
2. Python环境是否正确安装
3. 模型参数是否合理
4. 内存和GPU资源是否充足