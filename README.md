# 介绍
本项目包含中文情绪持有者抽取任务的数据集以及相关代码。

`./data`目录下的`train.json`、`dev.json`以及`test.json`分别为训练集，验证集和测试集。
`data_with_features.json`为在数据集基础上处理得到的用于本研究所提出模型的数据。
`label_map.json`为序列标注的标注方案。`pos_map.json`为本研究基于[LAC](https://github.com/baidu/lac)的词性类别所设定的词性类别及编号。

# 环境

本项目基于PaddlePaddle 2.6.1与PaddleNLP 2.6.1开发。

运行代码时建议使用英伟达显卡进行加速，建议根据[PaddlePaddle官网](https://www.paddlepaddle.org.cn/install/quick)信息安装合适的PaddlePaddle版本。


安装PaddlePaddle后，删除`requirements.txt`中的`paddlepaddle==2.6.1`，随后执行如下代码即可：

```bash
pip install -r requirements.txt
```


# 运行代码

由于本项目目前仅提供了`.ipynb`的运行文件，所以建议先安装`jupyterlab`，然后在jupyter lab中直接运行`train_EHE.ipynb`。

