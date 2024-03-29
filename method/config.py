from transformers import BertTokenizer
import os
import torch


class Config:
    def __init__(self, batch_size=16, num_epochs=5, lr=2e-5):
        self.current_path = os.path.dirname(__file__)
        self.train_path = os.path.join(self.current_path, 'data/train_data.json')
        self.test_path = os.path.join(self.current_path, 'data/test_data.json')
        self.model_name = os.path.join(self.current_path, 'chinese-macbert-base')
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sen_a_max_len = 100
        self.sen_b_max_len = 136
        self.max_len = 512
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_classes = 2
        self.dropout_rate = 0.3
        self.learning_rate = lr
        self.save_path = f"aspect_sbert_mean_{self.batch_size}_{self.num_epochs}.pth"
        self.logging_file_name = f'aspect_sbert_mean_logging_{self.batch_size}_{self.num_epochs}.log'
        self.best_model_name = 'aspect_sbert_mean_8_5_epoch_2.pth'


if __name__ == '__main__':
    config = Config()
    print(config.current_path)
