import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from preprocess import read_text_aspect_pairs
from model import SentenceBERT
import json
from tqdm import tqdm

config = Config()
model = SentenceBERT(config).to(config.device)
model.load_state_dict(torch.load(config.best_model_name))


def get_best_aspects(file):
    data = read_text_aspect_pairs(file)
    model.eval()
    cnt = 0
    with torch.no_grad():
        for i, instance in tqdm(enumerate(data), total=len(data)):
            instance['id'] = i
            text_encoded = config.tokenizer(instance['text'], padding=True,
                                            truncation=True, max_length=config.max_len, return_tensors='pt')
            text_encoded = {k: v.to(config.device) for k, v in text_encoded.items()}
            for pair in instance['pairs']:
                aspects_encoded = config.tokenizer(pair['aspects'], padding=True,
                                                   truncation=True, max_length=config.max_len, return_tensors='pt')
                aspects_encoded = {k: v.to(config.device) for k, v in aspects_encoded.items()}
                text_pooling, aspects_pooling = model(
                    sen_a_input_ids=text_encoded['input_ids'],
                    sen_a_token_type_ids=text_encoded['token_type_ids'],
                    sen_a_attention_mask=text_encoded['attention_mask'],
                    sen_b_input_ids=aspects_encoded['input_ids'],
                    sen_b_token_type_ids=aspects_encoded['token_type_ids'],
                    sen_b_attention_mask=aspects_encoded['attention_mask'],
                    inference=True
                )
                sim = F.cosine_similarity(text_pooling, aspects_pooling, dim=1)
                values, indices = torch.topk(sim, min(sim.size(0), 3))
                if values[0] >= 0.69:
                    cnt += 1
                    pair['best_aspect'] = [pair['aspects'][idx] for idx in indices]
                    pair['best_aspect_id'] = indices.cpu().tolist()
                    pair['sim'] = values.cpu().tolist()
    # json.dump(data, open('data/text_aspects_pair_labeled_top3_7_new.json', 'w+'), ensure_ascii=False, indent=4)
    print('#best_aspect: ', cnt)


if __name__ == '__main__':
    get_best_aspects('data/text_aspects_pair.json')

