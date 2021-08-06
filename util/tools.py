from network import *

# 计算准确率
def calc_accuracy(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    correct = sum(x==y for x, y in zip(y_pred, y_true))
    return correct / len(y_true)


def get_config_model(model_name):
    if model_name == 'GCN':
        config = GCNConfig()

        model = GCN(
            config.in_features, 
            config.out_features_per_layer, 
            config.num_layers, 
            config.dropout)
    elif model_name == 'GAT':
        config = GATConfig()

        model = GAT(
            config.in_features,
            config.out_features_per_layer, 
            config.num_heads_per_layer, 
            config.num_layers, 
            config.feat_drop, 
            config.attn_drop)
    elif model_name == 'ResGAT':
        config = GATConfig()

        model = ResGAT(
            config.in_features,
            config.out_features_per_layer, 
            config.num_heads_per_layer, 
            config.num_layers, 
            config.feat_drop, 
            config.attn_drop)
    elif model_name == 'Transformer':
        config = TransformerConfig()

        model = Transformer(
            config.in_features, 
            config.out_features_per_layer, 
            config.num_layers, 
            config.num_heads_per_layer, 
            config.feat_drop, 
            config.attn_drop
        )
    elif model_name == 'APPNP':
        config = APPNPConfig()

        model = APPNP(
            config.in_features, 
            config.hidden_size, 
            config.num_class, 
            config.num_layers,
            config.dropout, 
            config.k_hop, 
            config.alpha
        )
    elif model_name == 'GCNII':
        config = GCNIIConfig()

        model = GCNII(
            config.in_features, 
            config.num_class, 
            config.num_layers, 
            config.hidden_size, 
            config.dropout, 
            config.lambda_l, 
            config.alpha,
            config.k_hop
        )
    else:
        raise ValueError('model name is wrong! Please check ./util/tools.py file.')
    
    return config, model































