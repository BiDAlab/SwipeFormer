
import torch.nn as nn
import torch
from torch.nn.init import *
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
    
class EncoderLayerSeMCA(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayerSeMCA, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x))
        return self.sublayer[1](x, self.feed_forward)
    
class EncoderLayer_BlockRecTransf(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer_BlockRecTransf, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, mask))
        block_rec_att, block_rec_att_state =  self.self_attn(x, x, mask)
        x = self.sublayer[0](x, lambda x:block_rec_att)
        return self.sublayer[1](x, self.feed_forward)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Transformer_THAT(nn.Module):
    def __init__(self, hidden_dim, N, H):
        super(Transformer_THAT, self).__init__()
        self. pos_encoding = PositionalEncoding(hidden_dim, 0.2)
        self.model = Encoder(
            EncoderLayer(hidden_dim, MultiHeadedAttention(H, hidden_dim),
                         PositionwiseFeedForward(hidden_dim, hidden_dim*4)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        x = self.pos_encoding(x)
        return self.model(x, mask)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe.requires_grad = False
        #pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #x = x + self.pe[:x.size(0), :]#Variable(self.pe[:x.size(0), :], requires_grad=False)
        x = x + self.pe[:x.size(1), :].unsqueeze(0).repeat(x.size(0), 1, 1) ## modified by Bing to adapt to batch
        return self.dropout(x)
    
    

class PositionalEncoding_for_BERT(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding_for_BERT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe.requires_grad = False
        #pe = torch.mul(pe, 0.2)
        #pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #x = x + self.pe[:x.size(0), :]#Variable(self.pe[:x.size(0), :], requires_grad=False)
        return self.pe[:x.size(0), :]

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embeddings = PositionalEncoding_for_BERT(hidden_dim, 0.1)
        self.token_type_embeddings = nn.Embedding(2, hidden_dim)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, token_type_ids=None):
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(inputs_embeds)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

def normal_pdf(pos, mu, sigma):
    a = pos - mu
    log_p = -1*torch.mul(a, a)/(2*sigma) - torch.log(sigma)/2
    return F.softmax(log_p, dim=1)



class Gaussian_Position(nn.Module):
    def __init__(self, d_model, total_size, K=10):
        super(Gaussian_Position, self).__init__()
        #self.embedding = get_pe(d_model, K).to('cuda')
        #self.register_buffer('pe', self.embedding)
        self.embedding = nn.Parameter(torch.zeros([K, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.positions = torch.tensor([i for i in range(total_size)], requires_grad=False).unsqueeze(1).repeat(1, K).to('cuda')
        s = 0.0
        interval = total_size / K
        mu = []
        for _ in range(K):
            mu.append(nn.Parameter(torch.tensor(s, dtype=torch.float), requires_grad=True))
            s = s + interval
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor([torch.tensor([50.0], dtype=torch.float, requires_grad=True) for _ in range(K)]).unsqueeze(0))

    def forward(self, x):
        M = normal_pdf(self.positions, self.mu, self.sigma)
        pos_enc = torch.matmul(M, self.embedding)
        #print(M)
        return x + pos_enc.unsqueeze(0).repeat(x.size(0), 1, 1)
    
class Gaussian_Position_2(nn.Module):
    def __init__(self, d_model, total_size, K=10):
        super(Gaussian_Position_2, self).__init__()
        #self.embedding = get_pe(d_model, K).to('cuda')
        #self.register_buffer('pe', self.embedding)
        self.embedding = nn.Parameter(torch.zeros([K, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.positions = torch.tensor([i for i in range(total_size)], requires_grad=False).unsqueeze(1).repeat(1, K).to('cuda')
        s = 0.0
        interval = total_size / K
        mu = []
        for _ in range(K):
            mu.append(nn.Parameter(torch.tensor(s, dtype=torch.float), requires_grad=True))
            s = s + interval
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor([torch.tensor([50.0], dtype=torch.float, requires_grad=True) for _ in range(K)]).unsqueeze(0))

    def forward(self, x):
        M = normal_pdf(self.positions, self.mu, self.sigma)
        pos_enc = torch.matmul(M, self.embedding)
        #print(M)
        return x + pos_enc.unsqueeze(0).repeat(x.size(0), 1, 1)
    
class Gaussian_Temporal_Position(nn.Module):
    def __init__(self, d_model, total_size, K=10):
        super(Gaussian_Temporal_Position, self).__init__()
        #self.embedding = get_pe(d_model, K).to('cuda')
        #self.register_buffer('pe', self.embedding)
        self.embedding = nn.Parameter(torch.zeros([K, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.positions = torch.tensor([i for i in range(total_size)], requires_grad=False).unsqueeze(1).repeat(1, K).to('cuda')
        s = 0.0
        interval = total_size / K
        mu = []
        for _ in range(K):
            mu.append(nn.Parameter(torch.tensor(s, dtype=torch.float), requires_grad=True))
            s = s + interval
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor([torch.tensor([50.0], dtype=torch.float, requires_grad=True) for _ in range(K)]).unsqueeze(0))

    def forward(self, x):
        M = normal_pdf(self.positions, self.mu, self.sigma)
        pos_enc = torch.matmul(M, self.embedding)
        #print(M)
        return x + pos_enc.unsqueeze(0).repeat(x.size(0), 1, 1)


class HAR_CNN(nn.Module):
    "Implements CNN equation."
    def __init__(self, d_model, d_ff, filters, dropout=0.1):
        super(HAR_CNN, self).__init__()
        self.kernel_num = int(d_ff)
        self.filter_sizes = filters
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(d_model)
        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv2d(
                                 in_channels=64,
                                 out_channels=64,
                                 kernel_size=filter_size,
                                 padding=int((filter_size-1)/2))
                             )
            self.encoders.append(self.__getattr__(enc_attr_name))

    def forward(self, x):
        enc_outs = []
        for encoder in self.encoders:
            f_map = encoder(x.transpose(-1, -2))
            enc_ = f_map
            #enc_ = F.relu(f_map)
            #k_h = enc_.size()[-1]
            #enc_ = F.max_pool1d(enc_, kernel_size=k_h)
            #enc_ = enc_.squeeze(dim=-1)
            enc_ = F.relu(self.dropout(self.bn(enc_)))
            enc_outs.append(enc_.unsqueeze(dim=1))
        re = torch.div(torch.sum(torch.cat(enc_outs, 1), dim=1), 3)
        encoding = re
        #encoding = self.dropout(torch.cat(enc_outs, 1))
        #q_re = F.relu(encoding)
        return encoding.transpose(-1, -2)
    
    

    
class HAR_LSTM(nn.Module):
    "Implements LSTM equation."
    def __init__(self, d_model, d_ff, filters, dropout=0.1):
        super(HAR_LSTM, self).__init__()
        self.kernel_num = int(d_ff)
        self.filter_sizes = filters
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(d_model)
        self.encoders = []
        layer_dim = 1
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.LSTM(d_model, d_model, layer_dim, batch_first=True))
            self.encoders.append(self.__getattr__(enc_attr_name))

    def forward(self, x):
        enc_outs = []
        for encoder in self.encoders:
            x_rnn, (hn, cn) = encoder(x)
            enc_ = x_rnn.transpose(-1, -2)
            enc_ = F.relu(self.dropout(self.bn(enc_)))
            enc_outs.append(enc_.unsqueeze(dim=1))
        re = torch.div(torch.sum(torch.cat(enc_outs, 1), dim=1), 3)
        encoding = re
        return encoding.transpose(-1, -2)
    
     
class HAR_BLSTM(nn.Module):
    "Implements BLSTM equation."
    def __init__(self, d_model, d_ff, filters, dropout=0.1):
        super(HAR_BLSTM, self).__init__()
        self.kernel_num = int(d_ff)
        self.filter_sizes = filters
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(d_model)
        self.encoders = []
        self.w_2 = nn.Linear(d_ff*2, d_model)
        layer_dim = 1
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.LSTM(d_model, d_model, layer_dim, batch_first=True, bidirectional=True))
            self.encoders.append(self.__getattr__(enc_attr_name))

    def forward(self, x):
        enc_outs = []
        for encoder in self.encoders:
            x_rnn, (hn, cn) = encoder(x)
            x_rnn = self.w_2(x_rnn)
            enc_ = x_rnn.transpose(-1, -2)
            enc_ = F.relu(self.dropout(self.bn(enc_)))
            enc_outs.append(enc_.unsqueeze(dim=1))
        re = torch.div(torch.sum(torch.cat(enc_outs, 1), dim=1), 3)
        encoding = re
        return encoding.transpose(-1, -2)
    
class HAR_CNN_residualBlockChanged(nn.Module):
    "Implements CNN equation."
    def __init__(self, d_model, d_ff, filters, dropout=0.1):
        super(HAR_CNN_residualBlockChanged, self).__init__()
        self.kernel_num = int(d_ff)
        self.filter_sizes = filters
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(d_model)
        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(
                                 in_channels=d_model,
                                 out_channels=self.kernel_num,
                                 kernel_size=filter_size,
                                 padding=int((filter_size-1)/2))
                             )
            self.encoders.append(self.__getattr__(enc_attr_name))

    def forward(self, x):
        enc_outs = []
        for encoder in self.encoders:
            f_map = encoder(x.transpose(-1, -2))
            enc_ = f_map
            #enc_ = F.relu(f_map)
            #k_h = enc_.size()[-1]
            #enc_ = F.max_pool1d(enc_, kernel_size=k_h)
            #enc_ = enc_.squeeze(dim=-1)
            enc_ = self.bn(F.elu(self.dropout(enc_)))
            enc_outs.append(enc_.unsqueeze(dim=1))
        re = torch.div(torch.sum(torch.cat(enc_outs, 1), dim=1), 3)
        encoding = re
        #encoding = self.dropout(torch.cat(enc_outs, 1))
        #q_re = F.relu(encoding)
        return encoding.transpose(-1, -2)
    
    

class HAR_LSTM_6chan(nn.Module):
    "Implements CNN equation."
    def __init__(self, d_model, d_ff, filters, dropout=0.1):
        super(HAR_LSTM_6chan, self).__init__()
        self.kernel_num = int(d_ff)
        self.filter_sizes = filters
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(d_model)
        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             nn.LSTM(3, 3, 2, dropout=dropout, batch_first=True)
                             )
            self.encoders.append(self.__getattr__(enc_attr_name))

        self.drop = nn.Dropout2d(0.5)
        
    def forward(self, x):
        enc_outs = []
        for encoder in self.encoders:
            x_acc = x[:,  0:3, :]
            x_gyr = x[:,  3:6, :] 
            f_map_acc, _ = encoder(x_acc.transpose(-1, -2))
            # f_map_acc = f_map_acc.transpose(-1, -2)
            f_map_gyr, _ = encoder(x_gyr.transpose(-1, -2))
            # f_map_gyr = f_map_gyr.transpose(-1, -2)
            f_map = torch.cat([f_map_acc, f_map_gyr], dim=2)
            # f_map = f_map.transpose(-1, -2)
            enc_ = f_map
            # template = torch.cat(
            #     [embedded_acc_1, embedded_acc_2, embedded_acc_3, embedded_gyr_1, embedded_gyr_2, embedded_gyr_3], dim=1)
            # template = self.drop(template)
            # out = self.fc(template)
            # enc_ = out

            enc_ = F.relu(self.dropout(self.bn(enc_)))
            enc_outs.append(enc_.unsqueeze(dim=1))
        re = torch.div(torch.sum(torch.cat(enc_outs, 1), dim=1), 3)
        encoding = re
        #encoding = self.dropout(torch.cat(enc_outs, 1))
        #q_re = F.relu(encoding)
        return encoding.transpose(-1, -2)
    
class HARTransformer(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(HARTransformer, self).__init__()
        #self.pos_encoding = get_pe(hidden_dim)
        self.model = Encoder(
            EncoderLayer(hidden_dim, MultiHeadedAttention(H, hidden_dim),
                         HAR_CNN(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        #x = self.pos_encoding(x)
        return self.model(x, mask)

class SeMCA(nn.Module):
    def __init__(self, channel= 64, img_size=6, d=80,spe_head_Sidelength=5,padsize=15,share_head=9,spe_head=9):
        super().__init__()  
        self.d = d
        self.pad1 = self.d//2
        self.img_size = img_size
        self.spe_head_Sidelength = spe_head_Sidelength#spe_head_Sidelength*==spe_head
        self.padsize = padsize
        self.share_head = share_head
        self.spe_head = spe_head
        self.pad_space = padsize-img_size
        self.pad_spe_start = self.pad_space//2
        self.v_Pad = nn.ConstantPad3d((self.pad_space,0,self.pad_space,0,self.pad1,self.pad1), 0)
        self.kq_Pad = nn.ZeroPad2d(padding=(0,self.pad_space,0,self.pad_space))
        self.Unfold = nn.Unfold(kernel_size=(int((self.share_head)**0.5),\
            int((self.share_head)**0.5)), dilation=int((self.spe_head)**0.5), padding=0, stride=1)
        
        self.convd_key =nn.Conv1d(in_channels=self.d,
            out_channels=self.d, kernel_size=self.d,\
                stride=1, padding=self.d//2, groups=self.d)
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*self.spe_head*self.share_head,self.spe_head*self.share_head, 1,groups=self.spe_head,bias=0),
            nn.BatchNorm2d(self.spe_head*self.share_head),
            nn.ReLU(),
            nn.Conv2d(self.spe_head*self.share_head,self.spe_head*self.d,1,groups=self.spe_head)
        )
        self.convd_value = nn.Conv2d(channel,channel,3,padding=1,groups=channel)


    def forward(self, x):
        x = x.view(x.size()[0], x.size()[1], x.size()[2], 1)
        bs,c,h,w = x.shape
        k = x.reshape(bs,c,h*w).permute(0,2,1)
        k1 = self.convd_key(k).permute(0,2,1).reshape(bs,c,h,w)
        k1_pad,q_pad = self.kq_Pad(k1),self.kq_Pad(x)
        key_pad = self.Unfold(k1_pad).permute(0,2,1)
        q_pad = self.Unfold(q_pad).permute(0,2,1)
        key_q = torch.stack((q_pad,key_pad),dim=2).reshape(bs,2*self.spe_head,self.share_head,c)
        key_q = key_q.reshape(bs,2*self.spe_head*self.share_head,c).unsqueeze(-1) 
        attention = self.attention_embed(key_q)#[32, 225, 64, 1])
        attention = attention.view(bs,self.spe_head,self.d,c).repeat(1,self.share_head,1,1).permute(0,3,1,2)
        # [32, 225, 64, 1]--[32, 25, 9, 64]--[32, 225, 9, 64]--[32, 64, 225, 9]
        attention = attention.reshape(bs,c,self.padsize//self.spe_head_Sidelength,\
            self.padsize//self.spe_head_Sidelength,self.spe_head_Sidelength,self.spe_head_Sidelength,self.d)
        attention = attention.transpose(3,4).reshape(bs,c,self.padsize,self.padsize,self.d)
        attention = F.softmax(attention,dim=-1).permute(0,1,4,2,3)
        v = self.convd_value(x)
        pad_v = self.v_Pad(v)
        v = pad_v.reshape(bs,-1,self.padsize*self.padsize)
        v = v.unfold(1, self.d, 1).permute(0,1,3,2).reshape(bs,-1,self.d,self.padsize,self.padsize)
        temp_out = torch.sum(v*attention,dim=-3)
        out = temp_out+k1_pad
        out = out[:,:,self.pad_spe_start:h+self.pad_spe_start,self.pad_spe_start:w+self.pad_spe_start]
        return out

class SaMCA(nn.Module):

    def __init__(self, dim=6, kernel_size=3, spa_head=1):
        super().__init__()
        self.dim=dim
        self.spa_head = spa_head
        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
            )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
            )
        factor=2
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor, 1,bias=False,groups=self.spa_head),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,self.spa_head,1,groups=self.spa_head)
            )
        
    def forward(self, x):
        x = x.view(x.size()[0], x.size()[1], x.size()[2], 1)        
        bs,c,h,w=x.shape
        k1=self.key_embed(x) 
        v = self.value_embed(x)  
        v=x.view(bs,c,-1) 
        k_q = torch.stack((k1,x),2).reshape(bs,c*2,h,w)
        att=self.attention_embed(k_q).reshape(bs,self.spa_head,h*w)
        k2=F.softmax(att,dim=-1).unsqueeze(-2)
        k2 = k2.repeat(1,1,c//self.spa_head,1).reshape(bs,c,-1)
        k2 = k2*v
        k2=k2.view(bs,c,h,w)
        x = k1+k2
        return x.view(x.size()[0], x.size()[1], x.size()[2])



class HARTransformerSeMCA(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(HARTransformerSeMCA, self).__init__()
        #self.pos_encoding = get_pe(hidden_dim)
        self.model = Encoder(
            EncoderLayerSeMCA(hidden_dim, SeMCA(),
                         HAR_CNN(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        #x = self.pos_encoding(x)
        return self.model(x, mask)
    
    

class HARTransformerSaMCA(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(HARTransformerSaMCA, self).__init__()
        #self.pos_encoding = get_pe(hidden_dim)
        self.model = Encoder(
            EncoderLayerSeMCA(hidden_dim, SaMCA(),
                         HAR_CNN(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        #x = self.pos_encoding(x)
        return self.model(x, mask)
    
    
    
class HARTransformerLSTM(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(HARTransformerLSTM, self).__init__()
        #self.pos_encoding = get_pe(hidden_dim)
        self.model = Encoder(
            EncoderLayer(hidden_dim, MultiHeadedAttention(H, hidden_dim),
                         HAR_LSTM(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        #x = self.pos_encoding(x)
        return self.model(x, mask)
    
    
class HARTransformerBLSTM(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(HARTransformerBLSTM, self).__init__()
        #self.pos_encoding = get_pe(hidden_dim)
        self.model = Encoder(
            EncoderLayer(hidden_dim, MultiHeadedAttention(H, hidden_dim),
                         HAR_BLSTM(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        #x = self.pos_encoding(x)
        return self.model(x, mask)
class HAR_BRTransformer(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(HAR_BRTransformer, self).__init__()
        #self.pos_encoding = get_pe(hidden_dim)
        self.model = Encoder(
            EncoderLayer_BlockRecTransf(hidden_dim, BlockRecurrentAttention(hidden_dim, hidden_dim),
                         HAR_CNN(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        #x = self.pos_encoding(x)
        return self.model(x, mask)
    
class TransformerAutoformer(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(TransformerAutoformer, self).__init__()
        self.pos_encoding = PositionalEncoding(hidden_dim, 0.2)
        self.model = Encoder(
            EncoderLayer(hidden_dim,
                    AttentionLayer(
                        FullAttention(False,factor=5, attention_dropout=0.1,
                                      output_attention=False), d_model=hidden_dim, n_heads=H),
                         PositionwiseFeedForward(hidden_dim, hidden_dim*4)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        x = self.pos_encoding(x)
        return self.model(x, mask)
 
    
class TransformerFrequency(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(TransformerAutoformer, self).__init__()
        self.pos_encoding = PositionalEncoding(hidden_dim, 0.2)
        self.model = Encoder(
            EncoderLayer(hidden_dim,
                    AttentionLayer(
                        FullAttention(False,factor=5, attention_dropout=0.1,
                                      output_attention=False), d_model=hidden_dim, n_heads=H),
                         PositionwiseFeedForward(hidden_dim, hidden_dim*4)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        x = self.pos_encoding(x)
        return self.model(x, mask)
     
class InformerAutoformer(nn.Module):
    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(InformerAutoformer, self).__init__()
        self.pos_encoding = PositionalEncoding(hidden_dim, 0.2)
        self.model = Encoder(
            EncoderLayer(hidden_dim,
                    AttentionLayer(
                        ProbAttention(False,factor=5, attention_dropout=0.1,
                                      output_attention=False), d_model=hidden_dim, n_heads=H),
                         PositionwiseFeedForward(hidden_dim, hidden_dim*4)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        x = self.pos_encoding(x)
        return self.model(x, mask)


class AutoformerAutoformer(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(AutoformerAutoformer, self).__init__()
        self.pos_encoding = PositionalEncoding(hidden_dim, 0.2)
        self.model = Encoder(
            EncoderLayer(hidden_dim,
                        AutoCorrelationLayer(
                            AutoCorrelation(False,factor=1, attention_dropout=0.1,
                                      output_attention=False), d_model=hidden_dim, n_heads=H),
                         PositionwiseFeedForward(hidden_dim, hidden_dim*4)
                         , 0.1),
            N
        )


    def forward(self, x, mask=None):
        x = self.pos_encoding(x)
        return self.model(x, mask)   

    
class BlockRecTransformer(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(BlockRecTransformer, self).__init__()
        self.pos_encoding = PositionalEncoding(hidden_dim, 0.2)
        self.model = Encoder(
            EncoderLayer_BlockRecTransf(hidden_dim, BlockRecurrentAttention(hidden_dim, hidden_dim),
                          PositionwiseFeedForward(hidden_dim, hidden_dim*4)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        x = self.pos_encoding(x)
        return self.model(x, mask)
    
class HARTransformerAutoformer(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(HARTransformerAutoformer, self).__init__()
        #self.pos_encoding = get_pe(hidden_dim)
        self.model = Encoder(
            EncoderLayer(hidden_dim,
                    AttentionLayer(
                        FullAttention(False,factor=5, attention_dropout=0.1,
                                      output_attention=False), d_model=hidden_dim, n_heads=H),
                         HAR_CNN(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        #x = self.pos_encoding(x)
        return self.model(x, mask)  

class HARInformerAutoformer(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(HARInformerAutoformer, self).__init__()
        #self.pos_encoding = get_pe(hidden_dim)
        self.model = Encoder(
            EncoderLayer(hidden_dim,
                    AttentionLayer(
                        ProbAttention(False,factor=1, attention_dropout=0.1,
                                      output_attention=False), d_model=hidden_dim, n_heads=H),
                         HAR_CNN(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        #x = self.pos_encoding(x)
        return self.model(x, mask)  

class HARAutoformerAutoformer(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(HARAutoformerAutoformer, self).__init__()
        #self.pos_encoding = get_pe(hidden_dim)
        self.model = Encoder(
            EncoderLayer(hidden_dim,
                        AutoCorrelationLayer(
                            AutoCorrelation(False,factor=1, attention_dropout=0.1,
                                      output_attention=False), d_model=hidden_dim, n_heads=H),
                         HAR_CNN(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        #x = self.pos_encoding(x)
        return self.model(x, mask)  

class HARAutoformerAutoformerRB(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(HARAutoformerAutoformerRB, self).__init__()
        #self.pos_encoding = get_pe(hidden_dim)
        self.model = Encoder(
            EncoderLayer(hidden_dim,
                        AutoCorrelationLayer(
                            AutoCorrelation(False,factor=1, attention_dropout=0.1,
                                      output_attention=False), d_model=hidden_dim, n_heads=H),
                         HAR_CNN_residualBlockChanged(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        #x = self.pos_encoding(x)
        return self.model(x, mask)  
    
    
class HAR_BRTransformerRB(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(HAR_BRTransformerRB, self).__init__()
        #self.pos_encoding = get_pe(hidden_dim)
        self.model = Encoder(
            EncoderLayer_BlockRecTransf(hidden_dim, BlockRecurrentAttention(hidden_dim, hidden_dim),
                         HAR_CNN_residualBlockChanged(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        #x = self.pos_encoding(x)
        return self.model(x, mask)
    

    
class HARTransformer_2chan(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(HARTransformer_2chan, self).__init__()
        #self.pos_encoding = get_pe(hidden_dim)
        self.model = Encoder(
            EncoderLayer(hidden_dim, MultiHeadedAttention(H, hidden_dim),
                         HAR_CNN_2chan(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )
    def forward(self, x, mask=None):
        #x = self.pos_encoding(x)
        return self.model(x, mask)

class HARTransformerV(nn.Module):

    def __init__(self, hidden_dim, N, H, total_size, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(HARTransformerV, self).__init__()
        self. pos_encoding = PositionalEncoding(hidden_dim, 0.2)
        self.model = Encoder(
            EncoderLayer(hidden_dim, MultiHeadedAttention(H, hidden_dim),
                         HAR_CNN(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )


class HARTransformerVAF(nn.Module):
    def __init__(self, configs, filters=[1, 3, 5]):
        super(HARTransformerVAF, self).__init__()
        
        self.hidden_dim = configs.dimension
        self.N = configs.vlayers
        self.H = configs.vheads
        self.model = Encoder(
           EncoderLayer(self.hidden_dim, 
                           AttentionLayer(
                               FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=configs.output_attention), configs.dimension, configs.vheads),
                         HAR_CNN(self.hidden_dim, self.hidden_dim, filters)
                         , 0.1),
            self.N
        )


    def forward(self, x, mask=None):

        return self.model(x, mask)  

class HARTransformerHAF(nn.Module):
    def __init__(self, configs, filters=[1, 3, 5]):
        super(HARTransformerHAF, self).__init__()
        
        self.hidden_dim = configs.d_model
        self.N = configs.hlayers
        self.H = configs.hheads
        self.model = Encoder(
           EncoderLayer(self.hidden_dim, 
                           AttentionLayer(
                               FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=configs.output_attention), configs.d_model, configs.hheads),
                         HAR_CNN(self.hidden_dim, self.hidden_dim, filters)
                         , 0.1),
            self.N
        )


    def forward(self, x, mask=None):

        return self.model(x, mask)  

    
class HARInformer(nn.Module):
    def __init__(self, configs, filters=[1, 3, 5]):
        super(HARInformer, self).__init__()
        
        self.hidden_dim = configs.d_model
        self.N = configs.hlayers
        self.H = configs.d_model/10
        # self.enc_embedding = DataEmbedding(configs)
        # self. pos_encoding = PositionalEncoding(self.hidden_dim, 0.1)
        self.model = Encoder(
           EncoderLayer(self.hidden_dim, 
                           AttentionLayer(
                               FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                         HAR_CNN(self.hidden_dim, self.hidden_dim, filters)
                         , 0.1),
            self.N
        )


    def forward(self, x, mask=None):

        return self.model(x, mask)  
    
class HARInformerV(nn.Module):
    def __init__(self, configs, filters=[1, 3, 5]):
        super(HARInformerV, self).__init__()
        
        self.hidden_dim = configs.dimension
        self.N = configs.vlayers
        self.H = configs.dimension
        # self.enc_embedding = DataEmbedding(configs)
        # self. pos_encoding = PositionalEncoding(self.hidden_dim, 0.1)
        self.model = Encoder(
           EncoderLayer(self.hidden_dim, 
                           AttentionLayer(
                               ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=configs.output_attention), configs.dimension, configs.vheads),
                         HAR_CNN(self.hidden_dim, self.hidden_dim, filters)
                         , 0.1),
            self.N
        )


    def forward(self, x, mask=None):

        return self.model(x, mask)  

class HARInformerH(nn.Module):
    def __init__(self, configs, filters=[1, 3, 5]):
        super(HARInformerH, self).__init__()
        
        self.hidden_dim = configs.d_model
        self.N = configs.hlayers
        self.H = configs.d_model/10
        # self.enc_embedding = DataEmbedding(configs)
        # self. pos_encoding = PositionalEncoding(self.hidden_dim, 0.1)
        self.model = Encoder(
           EncoderLayer(self.hidden_dim, 
                           AttentionLayer(
                               ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=configs.output_attention), configs.d_model, configs.hheads),
                         HAR_CNN(self.hidden_dim, self.hidden_dim, filters)
                         , 0.1),
            self.N
        )


    def forward(self, x, mask=None):

        return self.model(x, mask)  
    
class HARAutoformerV(nn.Module):
    def __init__(self, configs, filters=[1, 3, 5]):
        super(HARAutoformerV, self).__init__()        
        self.hidden_dim = configs.dimension
        self.N = configs.vlayers
        self.H = configs.dimension
        self.model = Encoder(
           EncoderLayer(self.hidden_dim, 
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention),
                            configs.dimension, configs.vheads),
                         HAR_CNN(self.hidden_dim, self.hidden_dim, filters)
                         , 0.1),
            self.N
        )


    def forward(self, x, mask=None):

        return self.model(x, mask)  

class HARAutoformerH(nn.Module):
    def __init__(self, configs, filters=[1, 3, 5]):
        super(HARAutoformerH, self).__init__()        
        self.hidden_dim = configs.d_model
        self.N = configs.hlayers
        self.H = configs.d_model/10
        self.model = Encoder(
           EncoderLayer(self.hidden_dim, 
                            AutoCorrelationLayer(
                                AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                                output_attention=configs.output_attention),
                                configs.d_model, configs.hheads),
                         HAR_CNN(self.hidden_dim, self.hidden_dim, filters)
                         , 0.1),
            self.N
        )


    def forward(self, x, mask=None):

        return self.model(x, mask)   
    

class FEDformer(nn.Module):
    def __init__(self, configs, filters=[1, 3, 5]):
        super(FEDformer, self).__init__()        
        self.hidden_dim = configs.d_model
        self.N = configs.hlayers
        self.H = configs.d_model/10
        self.model = Encoder(
           EncoderLayer(self.hidden_dim, 
                            AutoCorrelationLayer(
                                AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                                output_attention=configs.output_attention),
                                configs.d_model, configs.hheads),
                         HAR_CNN(self.hidden_dim, self.hidden_dim, filters)
                         , 0.1),
            self.N
        )


    def forward(self, x, mask=None):

        return self.model(x, mask)   
 
    


class Swipe_CNN(nn.Module):
    "Implements CNN equation."
    def __init__(self, d_model, d_ff, filters, dropout=0.1):
        super(Swipe_CNN, self).__init__()
        self.kernel_num = int(d_ff)
        self.filter_sizes = filters
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(d_model)
        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(
                                 in_channels=d_model,
                                 out_channels=self.kernel_num,
                                 kernel_size=filter_size,
                                 padding=int((filter_size-1)/2))
                             )
            self.encoders.append(self.__getattr__(enc_attr_name))

    def forward(self, x):
        enc_outs = []
        for encoder in self.encoders:
            f_map = encoder(x.transpose(-1, -2))
            enc_ = f_map
            #enc_ = F.relu(f_map)
            #k_h = enc_.size()[-1]
            #enc_ = F.max_pool1d(enc_, kernel_size=k_h)
            #enc_ = enc_.squeeze(dim=-1)
            enc_ = F.relu(self.dropout(self.bn(enc_)))
            enc_outs.append(enc_.unsqueeze(dim=1))
        re = torch.div(torch.sum(torch.cat(enc_outs, 1), dim=1), 3)
        encoding = re
        #encoding = self.dropout(torch.cat(enc_outs, 1))
        #q_re = F.relu(encoding)
        return encoding.transpose(-1, -2)
    
class SwipeTransformer_arch(nn.Module):

    def __init__(self, hidden_dim, N, H, filters=[1, 3, 5]): #filters=[1, 5, 7]
        super(SwipeTransformer_arch, self).__init__()
        self.model = Encoder(
            EncoderLayer(hidden_dim, MultiHeadedAttention(H, hidden_dim),
                         Swipe_CNN(hidden_dim, hidden_dim, filters)
                         , 0.1),
            N
        )

    def forward(self, x, mask=None):
        return self.model(x, mask)
    
    
class Swipe_LSTM_Acien(nn.Module):
    def __init__(self, d_model, channels, output_dim, dropout=0.5):
        super(Swipe_LSTM_Acien, self).__init__()
        layer_dim = 1
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(d_model*channels)
        self.lstm1 = torch.nn.LSTM(d_model*channels, output_dim, layer_dim, batch_first = True)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.lstm2 = torch.nn.LSTM(output_dim, output_dim, layer_dim, batch_first = True)
        
    def forward(self, x, channels):
        x = self.bn1(x)
        x.unsqueeze_(-1)
        x = x.transpose(-1, -2)
        x, (hn, cn) = self.lstm1(x)
        x = x.transpose(-1, -2)
        x = self.dropout(x)
        x = self.bn2(x)
        x = x.transpose(-1, -2)
        x, (hn, cn) = self.lstm2(x)
        x.squeeze(1)
        return x
