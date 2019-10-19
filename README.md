
基于tensorflow实现的用TextCNN方法做中文情感分析(二分类)

text_cnn.py 定义了textCNN 模型网络结构

model.py 定义了训练代码

data.py 定义了数据预处理操作

data_set 存放了测试数据集合

neg.xls 是负面情感文本  
pos.xls 是正面情感文本  
[数据来源](http://spaces.ac.cn/usr/uploads/2015/08/646864264.zip)

├── data_set   
│   ├── neg.xls   
│   └── pos.xls   

+ 中文情感(二分类)分析训练：  
`python3 model.py --positive_data_file data_set/pos.xls --negative_data_file data_set/neg.xls` 

+ 中文情感分析加载模型预测：  
`python3 predict.py 这本书我感觉很不错`  
[(1, 0.86357594)]
`这本书我感觉很不错`模型预测的情感标签(1:正面；0:负面)及概率是：[(1, 0.86357594)]

+ 利用docker镜像进行情感预测：    
docker run --rm -it gswyhq/sentiment-analysis-textcnn python3 predict.py 这本书写得很好

[致谢](https://mp.weixin.qq.com/s/vq4-4M46okms5_sxw9PBQA)
