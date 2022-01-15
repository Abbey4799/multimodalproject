from torch.utils.data import Dataset, DataLoader
from config import Config
import torch
from preprocess import read_sen_pairs, read_news_pairs
import random
from utils import sample_negative


# Deprecated
class SentencePairDataset(Dataset):
    """
    两个句子分别输入BERT, [CLS] sen_a [SEP], [CLS] sen_b [SEP]
    """

    def __init__(self, sen_a_list, sen_b_list, config):
        super(SentencePairDataset, self).__init__()
        self.config = config
        self.sen_a_list = sen_a_list
        self.sen_b_list = sen_b_list

    def __len__(self):
        return len(self.sen_a_list)

    def __getitem__(self, index):
        return {
            'sen_a': self.sen_a_list[index],
            'sen_b': self.sen_b_list[index],
            'label': torch.tensor(1, dtype=torch.long)
        }

    def collate_fn(self, data):
        indices = list(range(len(data)))
        sen_a_list = [item['sen_a'] for item in data]
        sen_b_list = [item['sen_b'] for item in data]
        labels = [item['label'] for item in data]
        if len(indices) > 1:
            neg_a_list, neg_b_list = [], []
            for i, sen_a in enumerate(sen_a_list):
                j = sample_negative(indices, i)
                neg_a_list.append(sen_a_list[i])
                neg_b_list.append(sen_b_list[j])
            sen_a_list.extend(neg_a_list)
            sen_b_list.extend(neg_b_list)
            labels.extend([0] * len(data))
            indices = list(range(2 * len(data)))
            random.shuffle(indices)
        sen_a_encoded = self.config.tokenizer(sen_a_list, padding=True, max_length=512, truncation=True,
                                              return_tensors='pt')
        sen_b_encoded = self.config.tokenizer(sen_b_list, padding=True, max_length=512, truncation=True,
                                              return_tensors='pt')
        return {
            'sen_a_input_ids': sen_a_encoded['input_ids'][indices],
            'sen_a_token_type_ids': sen_a_encoded['token_type_ids'][indices],
            'sen_a_attention_mask': sen_a_encoded['attention_mask'][indices],
            'sen_b_input_ids': sen_b_encoded['input_ids'][indices],
            'sen_b_token_type_ids': sen_b_encoded['token_type_ids'][indices],
            'sen_b_attention_mask': sen_b_encoded['attention_mask'][indices],
            'label': torch.tensor(labels, dtype=torch.long)[indices]
        }

    def similarity_search_collate_fn(self, data):
        sen_a_list = [item['sen_a'] for item in data]
        sen_b_list = [item['sen_b'] for item in data]
        sen_a_encoded = self.config.tokenizer(sen_a_list, padding=True, max_length=512, truncation=True,
                                              return_tensors='pt')
        sen_b_encoded = self.config.tokenizer(sen_b_list, padding=True, max_length=512, truncation=True,
                                              return_tensors='pt')
        indices = list(range(len(data)))
        random.shuffle(indices)
        label = torch.arange(len(data)).long()
        return {
            'sen_a_input_ids': sen_a_encoded['input_ids'][indices],
            'sen_a_token_type_ids': sen_a_encoded['token_type_ids'][indices],
            'sen_a_attention_mask': sen_a_encoded['attention_mask'][indices],
            'sen_b_input_ids': sen_b_encoded['input_ids'],
            'sen_b_token_type_ids': sen_b_encoded['token_type_ids'],
            'sen_b_attention_mask': sen_b_encoded['attention_mask'],
            'label': label[indices]
        }


class SingleBertDataset(Dataset):
    """
    两个句子拼在一起输入BERT: [CLS] sen_a [SEP] sen_b [SEP]
    """

    def __init__(self, sen_a_list, sen_b_list, labels, config):
        super(SingleBertDataset, self).__init__()
        self.config = config
        self.sen_a_list = sen_a_list
        self.sen_b_list = sen_b_list
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sen_a = self.sen_a_list[index]
        sen_b = self.sen_b_list[index]
        label = self.labels[index]

        encoded = self.config.tokenizer(sen_a, sen_b, padding='max_length',
                                        max_length=self.config.max_len, return_tensors='pt')
        for k, v in encoded.items():
            encoded[k] = v.squeeze()

        return {
            **encoded,
            'label': label
        }


if __name__ == '__main__':
    config = Config()
    titles, contents = read_news_pairs(config.train_path, sample=True)

    # print(config.bert_tokenizer(sen_a_list[0], sen_b_list[0], padding='max_length', max_length=config.max_len, return_tensors='pt'))
    dataset = SentencePairDataset(titles, contents, config)
    print(dataset[0]['sen_a'])
    print(dataset[0]['sen_b'])
    data_loader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate_fn, shuffle=True)
    print(dataset[0])
    batch = next(iter(data_loader))
    print(batch['label'])
    print(batch['sen_a_input_ids'].size())
    print(batch['sen_a_token_type_ids'].size())
    print(batch['sen_a_attention_mask'].size())
    print(batch['sen_b_input_ids'].size())
    print(batch['sen_b_token_type_ids'].size())
    print(batch['sen_b_attention_mask'].size())
    print(batch['label'].size())
    # print(dataset[0].keys())
    # print((dataset[0]['attention_mask'] != 0).sum(dim=0))
