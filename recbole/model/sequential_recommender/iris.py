# -*- coding: utf-8 -*-
# @Time    : 2022/02/22 19:32
# @Author  : Peilin Zhou, Yueqi Xie
# @Email   : zhoupl@pku.edu.cn
r"""
WSSR
################################################

Reference:
    Yueqi Xie and Peilin Zhou et al. "Decouple Side Information Fusion for Sequential Recommendation"
    Submited to SIGIR 2022.
"""

import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.IRISlayers import FeatureSeqEmbLayer, IRISTransformerEncoder, TransformerEncoder, VanillaAttention
from recbole.model.loss import BPRLoss
import copy
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

class IRIS(SequentialRecommender):
    """
    DIF-SR moves the side information from the input to the attention layer and decouples the attention calculation of
    various side information and item representation
    """

    def __init__(self, config, dataset):
        super(IRIS, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer

        self.attribute_hidden_size = config['attribute_hidden_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.fusion_type = config['fusion_type']
        self.combine_type = config['combine_type']

        self.lamdas = config['lamdas']
        self.attribute_predictor = config['attribute_predictor']

        

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.feature_embed_layer_list = nn.ModuleList(
            [copy.deepcopy(FeatureSeqEmbLayer(dataset,self.attribute_hidden_size[_],[self.selected_features[_]],self.pooling_mode,self.device)) for _
             in range(len(self.selected_features))])

        self.trm_encoder = IRISTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            attribute_hidden_size=self.attribute_hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            fusion_type=self.fusion_type,
            max_len=self.max_seq_length
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
            self.attribute_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        self.other_parameter_name = ['feature_embed_layer_list']

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64

        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1) # input seq length
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # triu: triupper matrix  (torch.uint8)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):  
        
        device = item_seq.device   
        
        max_len = self.max_seq_length

        item_emb = self.item_embedding(item_seq)
        
        # position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
        # concatenate item_emb 
        input_emb = item_emb 
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)      
        
        seq_outputs = []        

        # append feature_embs  
        for feature_embed_layer in self.feature_embed_layer_list:
            feature_table = []

            sparse_embedding, dense_embedding = feature_embed_layer(None, item_seq)
            sparse_embedding = sparse_embedding['item']
            dense_embedding = dense_embedding['item']           

            # concat the sparse embedding and float embedding
            if sparse_embedding is not None:
                feature_emb = sparse_embedding
            if dense_embedding is not None:
                feature_emb = dense_embedding

            extended_attention_mask = self.get_attention_mask(item_seq)
            trm_output = self.trm_encoder(input_emb, feature_emb, position_embedding, extended_attention_mask, output_all_encoded_layers=True)
            output = trm_output[-1]
            seq_output = self.gather_indexes(output, item_seq_len - 1)

            seq_outputs.append(seq_output)


        # ################################################################## 전체 합
        # # result = torch.sum(torch.stack(seq_outputs), dim=0) 

        # ################################################################## 전체 평균
        # # result = torch.mean(torch.stack(seq_outputs), dim=0)

        # ################################################################## 중앙값
        # # result,_ = torch.sort(torch.stack(seq_outputs), 0)
        # # result = result[len(result)//2]
        
        # ################################################################## min
        # # result = torch.min(torch.stack(seq_outputs), dim=0).values

        # ################################################################## max
        # # result = torch.max(torch.stack(seq_outputs), dim=0).values

        # ################################################################## gating
        # self.combine_layer = nn.Linear(self.hidden_size*(self.num_feature_field), self.hidden_size).to(device)
        # self.gate_sigmoid = torch.nn.Sigmoid()
        # seq_concat = torch.cat(seq_outputs, dim=-1)
        # gate_values = self.gate_sigmoid(self.combine_layer(seq_concat))
        # combined_result = torch.zeros_like(seq_outputs[0])
        # # 각 임베딩에 대해 가중치를 적용하여 합산
        # for i, embedding in enumerate(seq_outputs):
        #     weight = gate_values[:, :, i:i+1]  # 적절한 슬라이싱으로 가중치를 조정
        #     combined_result += weight * embedding
        # result = combined_result
        # ################################################################ concat    
        # # result = torch.cat(seq_outputs, dim=-1) #[2,1024,256] -> [1024, 512]
        # # result = self.combine_layer(result)
       
        #combine_type = 'sum', 'mean', 'median', 'min', 'max', 'concat', 'gate'
        if self.combine_type == 'sum':
            result = torch.sum(torch.stack(seq_outputs), dim=0)
        elif self.combine_type == 'mean':
            result = torch.mean(torch.stack(seq_outputs), dim=0)
        elif self.combine_type == 'median':
            result,_ = torch.sort(torch.stack(seq_outputs), 0)
            result = result[len(result)//2]
        elif self.combine_type == 'min':
            result = torch.min(torch.stack(seq_outputs), dim=0).values
        elif self.combine_type == 'max':
            result = torch.max(torch.stack(seq_outputs), dim=0).values
        elif self.combine_type == 'concat':
            self.combine_layer = nn.Linear(self.hidden_size*(self.num_feature_field), self.hidden_size).to(device)
            result = torch.cat(seq_outputs, dim=-1) #[2,1024,256] -> [1024, 512]
            result = self.combine_layer(result)
        elif self.combine_type == 'gate':
            self.combine_layer = nn.Linear(self.hidden_size*(self.num_feature_field), self.hidden_size).to(device)
            self.gate_sigmoid = torch.nn.Sigmoid()
            seq_concat = torch.cat(seq_outputs, dim=-1)
            gate_values = self.gate_sigmoid(self.combine_layer(seq_concat))
            combined_result = torch.zeros_like(seq_outputs[0])
            # 각 임베딩에 대해 가중치를 적용하여 합산
            for i, embedding in enumerate(seq_outputs):
                weight = gate_values[:, :, i:i+1]  # 적절한 슬라이싱으로 가중치를 조정
                combined_result += weight * embedding
            result = combined_result
        return result

########################################################################################################################

    def calculate_loss(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)# seq representation from model
        pos_items = interaction[self.POS_ITEM_ID]# positive items

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)

            # 각 항목에 대한 모델의 점수가 계산됨
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B] 
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)# BPR loss function
            return loss

        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) #  sequence output with the transposed item embeddings to get the most simliar predicted item
            loss = self.loss_fct(logits, pos_items) # Cross-Entropy Loss
            
            return loss        

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores