from paddlenlp.transformers import AutoTokenizer
from paddle.io import Dataset
import paddle

class SequenceTaggingDataset(Dataset):
    def __init__(self, data, bert_model_name, pos_tag_vocab, label_vocab, max_seq_len=128, tokenizer=None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.max_seq_len = max_seq_len
        self.pos_tag_vocab = pos_tag_vocab
        self.label_vocab = label_vocab
        self.convert_tag_to_ids(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.data[idx]
        pad_num = max(0, self.max_seq_len - len(item["input_ids"]))
        return {
            "input_ids": paddle.to_tensor(item["input_ids"][:self.max_seq_len] + [self.tokenizer.pad_token_id] * pad_num),
            "pos_ids": paddle.to_tensor(item["pos_ids"][:self.max_seq_len] + [self.pos_tag_vocab["o"]] * pad_num),
            "emotion_ids": paddle.to_tensor(item["emotion_ids"][:self.max_seq_len] + [0] * pad_num),
            "labels": paddle.to_tensor(item["labels"][:self.max_seq_len] + [self.label_vocab["O"]] * pad_num),
            "lengths": min(self.max_seq_len, len(item["input_ids"]))
        }

    def convert_tag_to_ids(self, data):
        self.data = []
        for item in data:
            self.data.append(
                {
                    "input_ids": self.tokenizer.convert_tokens_to_ids(item["input_tokens"]),
                    "pos_ids": [self.pos_tag_vocab[pt] for pt in item["pos_labels"]], 
                    "emotion_ids": item["emotion_labels"],
                }
            )
            self.data[-1]["labels"] = [self.label_vocab[pt] for pt in item["labels"]]
        self.length = len(self.data)