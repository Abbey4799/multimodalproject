import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F


class SentenceBERT(nn.Module):
    """
    Siamese 结构，两个句子分别输入BERT, [CLS] sen_a [SEP], [CLS] sen_b [SEP]
    """

    def __init__(self, config):
        super(SentenceBERT, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.bert_config = self.bert.config
        self.fc = nn.Linear(self.bert_config.hidden_size * 3, config.num_classes)

    def forward(self,
                sen_a_input_ids,
                sen_a_token_type_ids,
                sen_a_attention_mask,
                sen_b_input_ids,
                sen_b_token_type_ids,
                sen_b_attention_mask,
                inference=False):
        sen_a_bert_outputs = self.bert(
            input_ids=sen_a_input_ids,
            token_type_ids=sen_a_token_type_ids,
            attention_mask=sen_a_attention_mask
        )
        sen_b_bert_outputs = self.bert(
            input_ids=sen_b_input_ids,
            token_type_ids=sen_b_token_type_ids,
            attention_mask=sen_b_attention_mask
        )
        sen_a_bert_output, sen_b_bert_output = sen_a_bert_outputs[0], sen_b_bert_outputs[0]
        # (batch_size, seq_len, hidden_size)

        sen_a_len, sen_b_len = (sen_a_attention_mask != 0).sum(dim=1, keepdim=True), (sen_b_attention_mask != 0).sum(dim=1, keepdim=True)

        sen_a_pooling, sen_b_pooling = sen_a_bert_output.sum(dim=1) / sen_a_len, sen_b_bert_output.sum(dim=1) / sen_b_len
        # (batch_size, hidden_size)

        if inference:
            return sen_a_pooling, sen_b_pooling

        hidden = torch.cat([sen_a_pooling, sen_b_pooling, torch.abs(sen_a_pooling - sen_b_pooling)], dim=1)

        return self.fc(hidden)


class RegressionSentenceBERT(nn.Module):
    """
    Siamese 结构，两个句子分别输入BERT, [CLS] sen_a [SEP], [CLS] sen_b [SEP]
    """

    def __init__(self, config):
        super(RegressionSentenceBERT, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.bert_config = self.bert.config
        self.similarity = nn.CosineSimilarity(dim=1)

    def forward(self,
                sen_a_input_ids,
                sen_a_token_type_ids,
                sen_a_attention_mask,
                sen_b_input_ids,
                sen_b_token_type_ids,
                sen_b_attention_mask, inference=True):
        sen_a_bert_outputs = self.bert(
            input_ids=sen_a_input_ids,
            token_type_ids=sen_a_token_type_ids,
            attention_mask=sen_a_attention_mask
        )
        sen_b_bert_outputs = self.bert(
            input_ids=sen_b_input_ids,
            token_type_ids=sen_b_token_type_ids,
            attention_mask=sen_b_attention_mask
        )
        sen_a_bert_output, sen_b_bert_output = sen_a_bert_outputs[0], sen_b_bert_outputs[0]
        # (batch_size, seq_len, hidden_size)

        sen_a_len, sen_b_len = (sen_a_attention_mask != 0).sum(dim=1, keepdim=True), (sen_b_attention_mask != 0).sum(dim=1, keepdim=True)

        sen_a_pooling, sen_b_pooling = sen_a_bert_output.sum(dim=1) / sen_a_len, sen_b_bert_output.sum(dim=1) / sen_b_len
        # (batch_size, hidden_size)

        return self.similarity(sen_b_pooling, sen_b_pooling)


class BertClassifier(nn.Module):
    """
    两个句子拼在一起输入BERT: [CLS] sen_a [SEP] sen_b [SEP]
    """

    def __init__(self, config):
        super(BertClassifier, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.bert_config = self.bert.config
        self.fc = nn.Linear(self.bert_config.hidden_size, config.num_classes)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        pooler_output = outputs.pooler_output
        return self.fc(pooler_output)
