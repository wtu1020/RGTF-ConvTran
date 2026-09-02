# RGTF-ConvTran模型结构

## Rhythm-aware ConvTran分支

1. 时间卷积提取局部ECG变化；
2. 空间卷积融合输入导联；
3. tAPE为ECG tokens提供时间绝对位置编码；
4. RR/HRV编码器将节律特征映射为rhythm token；
5. rhythm token与ECG tokens共同进入eRPE Transformer；
6. 汇总rhythm token、平均池化、最大池化和token attention pooling；
7. 线性投影得到时域表示。

## Frequency CNN分支

1. 使用Hann窗进行STFT；
2. 计算对数幅度谱并进行谱内标准化；
3. 使用三层Conv-BatchNorm-GELU-MaxPool提取频域表示；
4. 使用SE模块进行通道重标定；
5. 全局池化并投影为频域表示。

## Gated fusion

门控网络根据时域表示和频域表示产生两个softmax权重，然后对两个分支进行加权融合。融合结果输入主分类器。模型还保留时域和频域辅助分类头，但仓库不提供与辅助头相关的损失定义或训练设置。

## 结构参数

模型构造函数公开以下结构参数：

- `in_channels`
- `seq_len`
- `hrv_dim`
- `num_classes`
- `d_model`
- `out_dim`
- `temporal_filters`
- `temporal_kernel`
- `temporal_stride`
- `transformer_layers`
- `heads`
- `ffn_dim`
- `dropout`
- `n_fft`
- `hop_length`
- `freq_cnn_channels`
- `use_se`
- `aux_heads`
- `erpe_post_softmax`
- `classifier_dropout`

其中输入相关参数必须由调用者提供，仓库不保存任何数据集的具体输入配置。

