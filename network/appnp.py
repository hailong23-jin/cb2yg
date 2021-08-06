import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from config import Config


class APPNPConfig(Config):
    def __init__(self):
        super(APPNPConfig, self).__init__()
        # 定义模型参数
        self.model_name = 'APPNP'
        self.num_layers = 3
        self.hidden_size = 64
        self.dropout = 0.2
        self.k_hop = 10
        self.alpha = 0.1


class APPNP(nn.Layer):
    """Implement of APPNP 
    """

    def __init__(self, in_features, hidden_size, num_class, num_layers, dropout=0.5, k_hop=10, alpha=0.1):
        super(APPNP, self).__init__()
        self.mlps = nn.LayerList()
        self.mlps.append(nn.Linear(in_features, hidden_size))

        self.drop_fn = nn.Dropout(dropout)
        for _ in range(num_layers - 1):
            self.mlps.append(nn.Linear(hidden_size, hidden_size))

        self.output = nn.Linear(hidden_size, num_class)

        self.appnp = pgl.nn.APPNP(alpha=alpha, k_hop=k_hop)

    def forward(self, graph, feature):
        for m in self.mlps:
            feature = self.drop_fn(feature)
            feature = m(feature)
            feature = F.relu(feature)
        feature = self.drop_fn(feature)
        feature = self.output(feature)
        feature = self.appnp(graph, feature)
        return feature