import torch
from datasets import load_dataset
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizerFast

from training_ground.SentimentAnalysisModel import SentimentAnalysisModel

tokenizer = (BertTokenizerFast.from_pretrained(
    pretrained_model_name_or_path="bert-base-uncased"
))


def get_financial_data():
    dataset = load_dataset(
        "financial_phrasebank",
        "sentences_66agree",
        split="train",
    )
    return dataset.to_pandas()


def trainer():
    # loading the data into tensors, with 70-30 train test split
    df = get_financial_data()
    tokenized_sentences = tokenizer(df["sentence"].tolist(),
                                    add_special_tokens=True,
                                    max_length=512,
                                    padding='max_length',
                                    return_tensors="pt",
                                    truncation=True)
    sentiments = torch.tensor(df["label"].tolist())
    print("got the data tokenized")

    dataset = TensorDataset(
        tokenized_sentences["input_ids"],
        tokenized_sentences["attention_mask"],
        sentiments)
    train_set_size = int(len(dataset) * 0.7)
    validation_set_size = len(dataset) - train_set_size

    training_set, validation_set = random_split(dataset, [train_set_size, validation_set_size])

    batch_size = 16
    training_dataloader = DataLoader(
        training_set,
        sampler=RandomSampler(training_set),
        batch_size=batch_size
    )
    validation_dataloader = DataLoader(
        validation_set,
        sampler=SequentialSampler(validation_set),
        batch_size=batch_size
    )
    print("put the data into loader")

    # loading our model
    sentiment_model = SentimentAnalysisModel()
    print("loaded the model")

    epochs = 1
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(
        params=sentiment_model.parameters(),
        lr=learning_rate
    )
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        sentiment_model.train()
        training_progress_bar = tqdm(training_dataloader, desc=f"Epoch {epoch + 1} - Training")
        for batch in training_progress_bar:
            input_ids, attention_mask, target_sentiments = batch

            optimizer.zero_grad()  # reset the optimizer because the loss accumulates
            outputs = sentiment_model(input_ids, attention_mask)
            batch_loss = loss_function(outputs, target_sentiments)
            batch_loss.backward()  # gradients - the rate of change of the loss functions
            optimizer.step()  # gradient descent to find the weights which minimize the loss function

            training_progress_bar.set_postfix(loss=batch_loss.item())

        # testing how good our model is at classifying the sentences
        sentiment_model.eval()
        total_correct_classification = 0
        total_samples = 0
        validation_progress_bar = tqdm(validation_dataloader, desc=f"Epoch {epoch + 1} - Validating")
        for batch in validation_progress_bar:
            input_ids, attention_mask, target_sentiments = batch

            with torch.no_grad():
                outputs = sentiment_model(input_ids, attention_mask)

                sentiment_prediction = torch.argmax(outputs, 1)
                total_correct_classification += torch.sum(torch.eq(sentiment_prediction, target_sentiments))
                total_samples += target_sentiments.size(0)
        print(f"epoch: {epoch}, accuracy: {total_correct_classification / total_samples}")

    torch.save(sentiment_model, "sentiment_analysis_model.pt")


if __name__ == '__main__':
    trainer()
