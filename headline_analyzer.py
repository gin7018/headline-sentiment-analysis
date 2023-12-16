import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import news_material_provider

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model.eval()

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
    output = model(
        tokenized_headline["input_ids"],
        attention_mask=tokenized_headline["attention_mask"]
    )
    predicted_sentiment = torch.argmax(output.logits, dim=1).tolist()[0]

    if predicted_sentiment > 2: predicted_sentiment = 2
    return sentiment_labels[predicted_sentiment]


def gather_overall_topic_sentiment(query, time_span):
    headlines = news_material_provider.get_news_material(query, time_span)[:3]

    all_sentiments = []
    for headline in headlines:
        all_sentiments.append(get_sentiment(headline))
        print(f"title: {headline}, sentiment: {get_sentiment(headline)}")

    positives = all_sentiments.count("positive")
    negatives = all_sentiments.count("negative")
    neutrals = all_sentiments.count("neutral")

    overall_sentiment = torch.argmax(torch.tensor([negatives, neutrals, positives]), -1).item()
    return sentiment_labels[overall_sentiment]


if __name__ == '__main__':
    sent = gather_overall_topic_sentiment("tesla", "last week")
    print(f"overall sentiment: {sent}")
