import os

# 定义参数配置
class Config(object):
    def __init__(self):
        # 文件路径定义
        self.data_path = './data/data101014'
        self.test = os.path.join(self.data_path, 'test.csv')
        self.train = os.path.join(self.data_path, 'train.csv')
        self.edges = os.path.join(self.data_path, 'edges.csv')
        self.feat = os.path.join(self.data_path, 'feat.npy')
        self.model_path = './model'
        self.result_path = './result'

        # 超参数定义
        self.lr = 1e-3
        self.epoch = 8000
        self.num_class = 35
        self.in_features = 100