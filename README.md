# Traffic-sign recognition
A simple sample of Traffic Sign Recognition.

项目基于Tensorflow进行实现。

#### 文件说明：
---
*    input_data.py:  图片的输入
*    traffic_sign_cnn.py:  用cnn进行训练分类
*    testDemo.py:  用于测试已经训练出来的模型，输入单个图片输出结果，并分类到文件夹


#### 数据集说明：
---
*    使用的是比利时的交通标志数据集，可以网上自己找，里面有62个分类。

#### 网络说明：
---
*    CNN网络包含两个卷积层，两个全连接层。识别率大概在95% 左右




另外，训练开始前需要先在项目目录下新建文件夹./log/train/,用来保存模型参数,数据集的目录结构大概是./data/train/00001（标签）/图片
