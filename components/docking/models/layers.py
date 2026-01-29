import torch
from torch import nn

# 定义常用的激活函数
ACTIVATIONS = {
    'relu': nn.ReLU,# ReLU激活函数
    'silu': nn.SiLU# SiLU激活函数（Sigmoid Linear Unit）
}

def FCBlock(in_dim, hidden_dim, out_dim, layers, dropout, activation='relu'):
    
    """
    构建一个全连接（FC）块，包含多个隐藏层、激活函数、Dropout
    in_dim: 输入维度
    hidden_dim: 隐藏层维度
    out_dim: 输出维度
    layers: 隐藏层的数量
    dropout: Dropout的比率
    activation: 激活函数类型，默认为'relu'
    """
    activation = ACTIVATIONS[activation]# 根据参数选择激活函数
    assert layers >= 2# 至少有2层
    sequential = [nn.Linear(in_dim, hidden_dim), activation(), nn.Dropout(dropout)]
    for i in range(layers - 2):
        sequential += [nn.Linear(hidden_dim, hidden_dim), activation(), nn.Dropout(dropout)]
    sequential += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*sequential)


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    """
    用于将边的距离通过高斯函数进行嵌入，用于边的特征表示
    """
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        # 根据给定的参数生成高斯分布的偏移量
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        """
        根据输入的距离值（dist），计算每个距离对应的高斯值
        dist: 输入的边的距离
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)# 将距离和偏移量相减
        return torch.exp(self.coeff * torch.pow(dist, 2)) # 返回高斯函数的结果



class AtomEncoder(torch.nn.Module):
    """
        配体原子编码器，输入原子的类别和特征信息，输出嵌入后的向量
    """
    def __init__(self, emb_dim, feature_dims, sigma_embed_dim, lm_embedding_dim=0):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
         # feature_dims是一个包含两个元素的元组，第一个元素是每个类别特征的长度，第二个是标量特征的数量
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.additional_features_dim = feature_dims[1] + sigma_embed_dim + lm_embedding_dim
        # 为每个类别特征创建一个嵌入层
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

         # 如果有额外的特征，创建一个线性层来处理这些特征
        if self.additional_features_dim > 0:
            self.additional_features_embedder = torch.nn.Linear(self.additional_features_dim + emb_dim, emb_dim)

    def forward(self, x):
        """
        前向传播函数，输入x包含类别特征和其他标量特征，输出嵌入后的向量
        x: 输入数据，包含类别特征和标量特征
        """
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features + self.additional_features_dim
         # 对每个类别特征进行嵌入并相加
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.additional_features_dim > 0:
            x_embedding = self.additional_features_embedder(torch.cat([x_embedding, x[:, self.num_categorical_features:]], axis=1))
        return x_embedding


class OldAtomEncoder(torch.nn.Module):

    """
    旧版的原子编码器，包含类别特征、标量特征以及语言模型嵌入
    """
    def __init__(self, emb_dim, feature_dims, sigma_embed_dim, lm_embedding_type= None):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        
        super(OldAtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1] + sigma_embed_dim
        self.lm_embedding_type = lm_embedding_type
        # 为每个类别特征创建嵌入层
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:# 如果有标量特征，创建一个线性层
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
        if self.lm_embedding_type is not None:# 如果指定了语言模型嵌入类型，创建语言模型嵌入层
            if self.lm_embedding_type == 'esm':
                self.lm_embedding_dim = 1280
            else: raise ValueError('LM Embedding type was not correctly determined. LM embedding type: ', self.lm_embedding_type)
            self.lm_embedding_layer = torch.nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)# 创建一个线性层用于处理语言模型嵌入

    def forward(self, x):
        """
        前向传播函数，输入x包含类别特征、标量特征以及语言模型嵌入，输出嵌入后的向量
        x: 输入数据，包含类别特征、标量特征以及语言模型嵌入
        """
        x_embedding = 0
        if self.lm_embedding_type is not None:
            # 如果使用语言模型嵌入，确保输入的维度正确
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim
        else:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        # 对每个类别特征进行嵌入并相加
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        # 如果有标量特征，进行线性变换
        if self.num_scalar_features > 0:
            x_embedding += self.linear(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
        # 如果有语言模型嵌入，进行拼接并通过线性层转换
        if self.lm_embedding_type is not None:
            x_embedding = self.lm_embedding_layer(torch.cat([x_embedding, x[:, -self.lm_embedding_dim:]], axis=1))
        return x_embedding
