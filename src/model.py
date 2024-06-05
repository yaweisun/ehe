from paddlenlp.transformers import AutoModel
import paddle
from paddlenlp.layers import LinearChainCrf, LinearChainCrfLoss, ViterbiDecoder

class EHEModel(paddle.nn.Layer):
    def __init__(self, config):
        super(EHEModel, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.bert_model_name)
        hidden_size = self.bert.config["hidden_size"]
        self.dropout = paddle.nn.Dropout(config.dropout)
        if self.config.pos_emb:
            self.pos_embedding = paddle.nn.Embedding(config.pos_vocab_size, hidden_size)
        if self.config.emotion_emb:
            self.emotion_embedding = paddle.nn.Embedding(config.emotion_vocab_size, hidden_size)

        if self.config.bilstm:
            self.bilstm = paddle.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=1, direction='bidirectional')
            
        if self.config.crf:
            self.fc = paddle.nn.Linear(hidden_size, self.config.num_classes+2)  # 用于CRF的发射矩阵
            self.crf = LinearChainCrf(self.config.num_classes)
            self.crf_loss = LinearChainCrfLoss(self.crf)
            self.viterbi_decoder = ViterbiDecoder(self.crf.transitions)
        else:
            self.fc = paddle.nn.Linear(hidden_size, self.config.num_classes)  # 用于CRF的发射矩阵

    def forward(self, input_ids, pos_ids, emotion_ids, lengths, labels=None):
        # 处理input_ids
        bert_output, _ = self.bert(input_ids)
        bert_output = self.dropout(bert_output)
        # 处理pos_ids
        if self.config.pos_emb:
            pos_embedding = self.pos_embedding(pos_ids)
        # 处理emotion_ids
        if self.config.emotion_emb:
            emotion_embedding = self.emotion_embedding(emotion_ids)
        # 合并向量
        if self.config.pos_emb and self.config.emotion_emb:
            combined_output = bert_output + pos_embedding + emotion_embedding
        elif self.config.pos_emb:
            combined_output = bert_output + pos_embedding
        elif self.config.emotion_emb:
            combined_output = bert_output + emotion_embedding
        else:
            combined_output = bert_output
            
        # 应用BiLSTM
        if self.config.bilstm:
            lstm_output, _ = self.bilstm(combined_output)
            drpout_output = self.dropout(lstm_output)
            fc_output = self.fc(drpout_output)
        else:
            fc_output = self.fc(combined_output)
        
        if self.config.crf:
            # print(emissions)
            _, preds = self.viterbi_decoder(fc_output, lengths)
            if labels is not None:
                loss = self.crf_loss(fc_output, lengths, labels)
                return preds, loss.sum() / loss.shape[0]
            else:
                return preds
        else:
            preds = fc_output.argmax(axis=-1)
            if labels is not None:
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(fc_output.reshape((-1, self.config.num_classes)), labels.reshape((-1,)))
                return preds, loss
            else:
                return preds
            
