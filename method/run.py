import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from config import Config
from preprocess import read_sen_pairs, read_news_pairs
from model import SentenceBERT, BertClassifier, RegressionSentenceBERT
from dataset import SentencePairDataset, SingleBertDataset
from train_eval import train, evaluate, accuracy, similarity_accuracy, similarity_search_accuracy
import os, argparse, datetime
from pprint import pprint
from utils import seed_everything
from transformers import AdamW, get_linear_schedule_with_warmup


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    return parser.parse_args()


if __name__ == '__main__':
    seed_everything()
    args = set_args()
    print(args)
    config = Config(**vars(args))
    pprint(vars(config))
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=config.logging_file_name,
        filemode='a+',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    train_titles, train_contents = read_news_pairs(config.train_path, sample=True)
    test_titles, test_contents = read_news_pairs(config.test_path)

    # Single BERT Dataset
    # train_set = SingleBertDataset(train_sen_a_list, train_sen_b_list, train_labels, config)
    # test_set = SingleBertDataset(test_sen_a_list, test_sen_b_list, test_labels, config)

    # SentenceBERT
    train_set = SentencePairDataset(train_titles, train_contents, config)
    test_set = SentencePairDataset(test_titles, test_contents, config)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, collate_fn=train_set.collate_fn, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, collate_fn=test_set.collate_fn, shuffle=False)
    similarity_search_test_loader = DataLoader(test_set, batch_size=config.batch_size,
                                               collate_fn=test_set.similarity_search_collate_fn, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SentenceBERT
    model = SentenceBERT(config).to(device)

    loss = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * config.num_epochs)

    # logger.info("START")
    # best_model_name = ""
    # best_accuracy = 0.0
    # for epoch in range(config.num_epochs):
    #     train_loss = train(train_loader, model, loss, optimizer, device, scheduler)
    #     logger.info(f"Epoch: [{epoch + 1} / {config.num_epochs}]  | Train Loss: {train_loss}")
    #     acc = accuracy(test_loader, model, device)
    #     logger.info(f"Epoch: [{epoch + 1} / {config.num_epochs}] | Test ACC: {acc}")
    #     if acc > best_accuracy:
    #         best_accuracy = acc
    #         if best_model_name != '':
    #             os.remove(best_model_name)
    #         best_model_name = f"{config.save_path.split('.')[0]}_epoch_{epoch + 1}.pth"
    #         torch.save(model.state_dict(), best_model_name)
    #         logger.info(f"Model saved in {best_model_name} @Epoch {epoch + 1}")

    best_model_name = 'aspect_sbert_mean_8_5_epoch_2.pth'
    model.load_state_dict(torch.load(best_model_name))
    acc, all_predicts, all_labels = similarity_search_accuracy(similarity_search_test_loader, model, device)
    # print(f"Best Model '{best_model_name}' | Test ACC: {acc}")
    print(f"Best Model '{best_model_name}' | Test Similarity Search ACC: {acc}")
