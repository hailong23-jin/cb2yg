import paddle.nn as nn
from  pgl.nn import GCNConv
from config import Config


class GCNConfig(Config):
    def __init__(self):
        super(GCNConfig, self).__init__()
        # 定义模型名
        self.model_name = 'GCN'

        # 定义模型层数
        self.num_layers = 2

        # 定义每一层的输出向量维度
        self.out_features_per_layer = [256, self.num_class]
        
        # dropout
        self.dropout = 0.2


class GCN(nn.Layer):
    """Implement of GCN
    """

    def __init__(self, in_features, out_features_per_layer, num_layers=1, dropout=0.5):
        super(GCN, self).__init__()
        self.gcns = nn.LayerList()

        assert num_layers == len(out_features_per_layer)

        self.gcns = nn.LayerList()
        for i in range(num_layers - 1):
            self.gcns.append(
                GCNConv(
                    in_features if i == 0 else out_features_per_layer[i-1],
                    out_features_per_layer[i],
                    activation="relu", 
                    norm=True))
            self.gcns.append(nn.Dropout(dropout))

        self.gcns.append(GCNConv(out_features_per_layer[-2], out_features_per_layer[-1]))

    def forward(self, graph, feature):
        for m in self.gcns:
            if isinstance(m, nn.Dropout):
                feature = m(feature)
            else:
                feature = m(graph, feature)
        return feature

























































