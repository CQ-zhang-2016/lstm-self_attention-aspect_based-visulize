# lstm-self_attention-aspect_based-visulize
加上注意力机制的lstm，在500w评论集上训练得到，精度0.61。此外通过句法分析训练方向学习模型，可视化表示

您可能需要安装斯坦佛分词包（用于句法分析）
	https://stanfordnlp.github.io/CoreNLP/
您可能需要安装pyqt5（用于可视化）
	清华镜像源
您可能需要安装data（用于支持代码运行）
	

其中data包含有model.pth为已经在500w评论集上训练好的lstm+self_attention+regression_loss模型，精度0.61
其中data包含有embedding_matrix3为在600w数据集中训练的词嵌入矩阵
其中data包含有90000条评论用于方向标签的提取和打分（采用句法分析和本体库检索）
另外data包含有视频教学和演示
运行run.py文件实现可视化，内置90000条评论进行方向标签提取和打分，搜索框内输入新评论按下搜索进行预测得分和关键词提取，并可以检索相关评论
