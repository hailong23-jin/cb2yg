import paddle.nn as nn
from pgl.nn import TransformerConv
from config import Config

class TransformerConfig(Config):
    def  __init__(self):
        # model_name
        self.model_name = 'Transformer'

        # model_layers
        self.num_layers = 3

        # out_features_per_layer
        self.out_features_per_layer = [64, 64, self.num_class]

        # num_heads_per_layer
        self.num_heads_per_layer = [8, 8, 8]

        self.feat_drop = 0.2
        self.attn_drop = 0.2



class Transformer(nn.Layer):
    """Implement of TransformerConv
    """

    def __init__(self, in_features, out_features_per_layer, num_layers, num_heads_per_layer, feat_drop=0.6, attn_drop=0.6):
        super(Transformer, self).__init__()

        assert num_layers == len(out_features_per_layer) == len(num_heads_per_layer)

        out_features_per_layer = [in_features] + out_features_per_layer
        num_heads_per_layer = [1] + num_heads_per_layer

        self.trans = nn.LayerList()
        for i in range(num_layers):
            self.trans.append(
                TransformerConv(
                    out_features_per_layer[i] * num_heads_per_layer[i],
                    out_features_per_layer[i+1],
                    num_heads_per_layer[i+1],
                    feat_drop,
                    attn_drop,
                    concat = True if i < num_layers - 1 else False,
                    skip_feat=False,
                    layer_norm=True if i < num_layers - 1 else False,
                    activation='relu' if i < num_layers - 1 else None,

                )
            )

    def forward(self, graph, feature):
        for m in self.trans:
            feature = m(graph, feature)
        return feature