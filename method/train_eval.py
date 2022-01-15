from tqdm import tqdm
import torch
import torch.nn.functional as F


def train(data_loader, model, loss, optimizer, device, scheduler=None):
    model.train()
    final_loss = 0.0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        label = data.pop('label')
        outputs = model(**data)
        ls = loss(outputs, label)

        optimizer.zero_grad()
        ls.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        final_loss += ls.item()

    return final_loss / len(data_loader)


def evaluate(data_loader, model, loss, device):
    model.eval()
    final_loss = 0.0
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)

            label = data.pop('label')
            outputs = model(**data)
            ls = loss(outputs, label)
            final_loss += ls.item()

    return final_loss / len(data_loader)


# 训练任务评估
def accuracy(data_loader, model, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)

            label = data.pop('label')
            outputs = model(**data)

            predicts = outputs.argmax(dim=1)
            total += predicts.size(0)
            correct += (predicts == label).sum().item()

    return correct / total


# 余弦相似度计算评估
def similarity_accuracy(data_loader, model, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)

            label = data.pop('label')
            outputs = model(**data, inference=True)
            # print(outputs)
            # print(label)
            outputs = (outputs > 0.5).long()
            assert outputs.size() == label.size()
            correct += (outputs == label).sum().item()
            total += outputs.size(0)
    return correct / total


def similarity_search_accuracy(data_loader, model, device):
    model.eval()
    correct, total = 0, 0
    all_predicts, all_labels = [], []
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)

            label = data.pop('label')
            sen_a_pooling, sen_b_pooling = model(**data, inference=True)
            outputs = F.cosine_similarity(sen_a_pooling.unsqueeze(1), sen_b_pooling.unsqueeze(0), dim=-1)
            predict = outputs.argmax(dim=-1)
            all_predicts.extend(predict.cpu().tolist())
            all_labels.extend(label.cpu().tolist())
            correct += (predict == label).sum().item()
            total += label.size(0)
    return correct / total, all_predicts, all_labels
