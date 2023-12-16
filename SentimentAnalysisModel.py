import torch
import transformers


class SentimentAnalysisModel(transformers.PreTrainedModel):

    def __init__(self, pretrained_model="ghislainehaha/headline-sentiment-analyzer"):
        config = transformers.DistilBertConfig.from_pretrained(pretrained_model)
        super(SentimentAnalysisModel, self).__init__(config=config)
        self.bert = transformers.DistilBertModel.from_pretrained(pretrained_model)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.output = torch.nn.Linear(in_features=768, out_features=3)  # we have three possible classes

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.dropout(output[0][:, 0])
        output = self.output(output)
        return output
    