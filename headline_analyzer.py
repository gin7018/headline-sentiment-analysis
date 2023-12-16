import torch
from transformers import BertTokenizerFast

import news_material_provider
from SentimentAnalysisModel import SentimentAnalysisModel

headline_analyzer = SentimentAnalysisModel()
headline_analyzer.eval()

tokenizer = (BertTokenizerFast.from_pretrained(
    pretrained_model_name_or_path="bert-base-uncased",
    use_fast=True
))

sentiment_labels = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


def get_sentiment(headline):
    tokenized_headline = tokenizer(headline,
                                   add_special_tokens=True,
                                   max_length=512,
                                   padding='max_length',
                                   return_tensors="pt",
                                   truncation=True)
    with torch.no_grad():
        output = headline_analyzer(
            tokenized_headline["input_ids"],
            attention_mask=tokenized_headline["attention_mask"]
        )
        predicted_sentiment = torch.argmax(output, dim=1).tolist()

        return sentiment_labels[predicted_sentiment[0]]


def gather_overall_topic_sentiment(query, time_span):
    headlines = news_material_provider.get_news_material(query, time_span)[:3]
    # headlines = ["life is so good!", "stock market crash oh no", "nothing much"]
    all_sentiments = []  # TODO
    for headline in headlines:
        sent = get_sentiment(headline)
        print(f"title: {headline[:40]}, sentiment: {sent}")

    # TODO consolidate sentiment to get overall sentiment
    return None


if __name__ == '__main__':
    gather_overall_topic_sentiment("tesla", "last week")
