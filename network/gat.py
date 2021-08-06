import paddle.nn as nn
from pgl.nn import GATConv
from config import Config


# 继承自Config类
# Config类中定义了基础的参数配置
class GATConfig(Config):
    def __init__(self):
        super(GATConfig, self).__init__()
        # 模型名字，模型保存和结果保存均使用这个前缀
        self.model_name = 'ResGAT'

        # 表示搭建几层网络
        self.num_layers = 3

        # 每一层的输出特征维度
        self.out_features_per_layer = [64, 64, self.num_class]

        # 每一层中注意力头的个数
        self.num_heads_per_layer = [8, 8, 8]

        # dropout 
        self.feat_drop = 0.3
        self.attn_drop = 0.3


# 定义GAT神经网络架构
class GAT(nn.Layer):
    """Implement of GAT
    """

    def __init__(self, in_features, out_features_per_layer, num_heads_per_layer, num_layers=3, feat_drop=0.2, attn_drop=0.6): 
        super(GAT, self).__init__()

        assert num_layers == len(num_heads_per_layer) == len(out_features_per_layer)
        
        # 小技巧，为了简化下面的代码操作
        out_features_per_layer = [in_features] + out_features_per_layer
        num_heads_per_layer = [1] + num_heads_per_layer

        self.gats = nn.LayerList()
        for i in range(num_layers):
            self.gats.append(
                GATConv(out_features_per_layer[i] * num_heads_per_layer[i], 
                        out_features_per_layer[i+1],
                        feat_drop,
                        attn_drop,
                        num_heads_per_layer[i+1],
                        concat=True if i < num_layers - 1 else False,  # 只有最后一层不使用concat方法
                        activation='elu' if i < num_layers - 1 else None
                )
            )

    def forward(self, graph, feature):
        for m in self.gats:
            feature = m(graph, feature)
        return feature


class ResGAT(nn.Layer):
    """Implement of ResGAT
    """

    def __init__(self, in_features, out_features_per_layer, num_heads_per_layer, num_layers=3, feat_drop=0.2, attn_drop=0.6):
        super(ResGAT, self).__init__()

        hidden_size = out_features_per_layer[0] * num_heads_per_layer[0]
        # 仅进行线性映射，不加偏置
        self.in_features_proj = nn.Linear(in_features, hidden_size, bias_attr=False)

        self.num_layers = num_layers

        self.ln = nn.LayerNorm(hidden_size)

        out_features_per_layer = [hidden_size] + out_features_per_layer
        num_heads_per_layer = [1] + num_heads_per_layer

        self.gats = nn.LayerList()
        for i in range(num_layers):
            self.gats.append(
                GATConv(out_features_per_layer[i] * num_heads_per_layer[i], 
                        out_features_per_layer[i+1],
                        feat_drop,
                        attn_drop,
                        num_heads_per_layer[i+1],
                        concat=True if i < num_layers - 1 else False,  # 只有最后一层不使用concat方法
                        activation='elu' if i < num_layers - 1 else None
                )
            )


    def forward(self, graph, feature):
        feature_proj = self.in_features_proj(feature)

        # out = None
        # for i, m in enumerate(self.gats):
        #     out = m(graph, feature_proj)
            # if i < self.num_layers - 1:
            #     out = out + feature_proj
            #     feature_proj = self.ln(out)
        out1 = self.gats[0](graph, feature_proj)
        feature_proj = out1 + feature_proj
        out2 = self.gats[1](graph, feature_proj)
        feature_proj = out2 + feature_proj
        out3 = self.gats[2](graph, feature_proj)
        
        return out3

    

