import pandas as pd
import json
import random


def read_sen_pairs(file):
    data = pd.read_csv(file, sep='\t')
    sen_a_list = [str(sen) for sen in data['sen_a']]
    sen_b_list = [str(sen) for sen in data['sen_b']]
    labels = data['label'].tolist()
    print(len(max(sen_a_list, key=len)))
    print(len(max(sen_b_list, key=len)))
    print(len(sen_a_list))
    return sen_a_list, sen_b_list, labels


def read_news_pairs(file, sample=False):
    data = json.load(open(file, 'rb'))
    if sample:
        data = random.sample(data, 150000)
    data = [item for item in data if len(item['content']) <= 512 - 2]
    titles = [item['title'] for item in data]
    max_title_len = max([len(t) for t in titles])
    contents = [item['content'] for item in data]
    max_content_len = max([len(c) for c in contents])
    print('max_title_len: ', max_title_len)
    print('max_content_len: ', max_content_len)
    print("LEN: ", len(contents))
    return titles, contents


def save_text_aspect_pairs(file):
    data = []
    cnt = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            item = {}
            line = json.loads(line, encoding='utf-8')
            item['text'] = line['text']
            item['pairs'] = []
            for k, v in line['mention_img_url'].items():
                aspects = [list(d.keys())[0] for d in v if len(list(d.keys())[0]) > 0]
                if len(aspects) > 0:
                    item['pairs'].append({
                        'mention': k,
                        'aspects': aspects
                    })
            cnt += len(item['pairs'])
            data.append(item)
    print(cnt)
    json.dump(data, open('data/text_aspects_pair.json', 'w+'), ensure_ascii=False, indent=4)


def read_text_aspect_pairs(file):
    return json.load(open(file, 'r+', encoding='utf-8'))


def calculate_average_aspects(file):
    data = read_text_aspect_pairs(file)
    num_mentions, num_aspects = 0, 0
    for instance in data:
        num_mentions += len(instance['pairs'])
        for pair in instance['pairs']:
            num_aspects += len(pair['aspects'])
    print(num_aspects / num_mentions)


def calculate_statistic(file):
    data = read_text_aspect_pairs(file)
    num_texts, num_aspects, num_mentions = 0, 0, 0
    num_texts = len(data)
    for instance in data:
        num_mentions += len(instance['pairs'])
        for pair in instance['pairs']:
            if 'best_aspect' in pair:
                num_aspects += 1
    print("#Text: ", num_texts)
    print("#Mention: ", num_mentions)
    print("#Aspect: ", num_aspects)


if __name__ == '__main__':
    calculate_statistic('data/text_aspects_pair_labeled_top3_55_new.json')
    # read_news_pairs('data/train_data.json', sample=True)
    # read_news_pairs('data/test_data.json')
    # save_text_aspect_pairs('data/data_3000.json')
