import torch
from transformers import BertForSequenceClassification


class SentimentAnalysisModel(torch.nn.Module):

    def __init__(self):
        super(SentimentAnalysisModel, self).__init__()
        self.bert = (BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased"
        ))
        self.dropout = torch.nn.Dropout(p=0.3)
        self.output = torch.nn.Linear(in_features=768, out_features=3)  # we have three possible classes

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.dropout(output[0][:, 0])
        output = self.output(output)
        return output
    