import paddle.nn as nn
import paddle.nn.functional as F
import pgl

from config import Config


# 继承自Config类
# Config类中定义了基础的参数配置
class GCNIIConfig(Config):
    def __init__(self):
        super(GCNIIConfig, self).__init__()
        self.model_name = 'GCNII'
        self.num_layers = 3
        self.hidden_size=64
        self.dropout=0.6
        self.lambda_l=0.5
        self.alpha=0.1
        self.k_hop=64


class GCNII(nn.Layer):
    """Implement of GCNII
    """

    def __init__(self, in_features, num_class, num_layers=1, hidden_size=64, dropout=0.6, lambda_l=0.5, alpha=0.1, k_hop=64):
        super(GCNII, self).__init__()
        self.mlps = nn.LayerList()
        self.mlps.append(nn.Linear(in_features, hidden_size))
        
        self.drop_fn = nn.Dropout(dropout)
        for _ in range(num_layers - 1):
            self.mlps.append(nn.Linear(hidden_size, hidden_size))

        self.output = nn.Linear(hidden_size, num_class)
        self.gcnii = pgl.nn.GCNII(
            hidden_size=hidden_size,
            activation="relu",
            lambda_l=lambda_l,
            alpha=alpha,
            k_hop=k_hop,
            dropout=dropout)

    def forward(self, graph, feature):
        for m in self.mlps:
            feature = m(feature)
            feature = F.relu(feature)
            feature = self.drop_fn(feature)
        feature = self.gcnii(graph, feature)
        feature = self.output(feature)
        return feature
