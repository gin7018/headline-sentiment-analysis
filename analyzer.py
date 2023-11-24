import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from transformers import TFBertForSequenceClassification, BertTokenizer
from official.nlp import optimization


training_examples = [
    "Altman Is Back at OpenAI, But Questions Remain as to Why He Was Fired in First Place",
    "Falling House Prices Could Worsen Mortgage Crunch for Britons",
    "Barclays Looking at Cutting £1 Billion in Costs, Reuters Says",
    "Google stock up 20% after new AI release"
]

bert_model_name = "bert_en_uncased_L-12_H-768_A-12"
bert_preprocessor = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
bert_encoder_endpoint = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"


def build_sentiment_classifier():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    text_preprocessing_layer = hub.KerasLayer(bert_preprocessor, name="preprocessing")

    bert_encoder_inputs = text_preprocessing_layer(text_input)
    bert_model = hub.KerasLayer(bert_encoder_endpoint, trainable=True, name="BERT_encoder")

    bert_results = bert_model(bert_encoder_inputs)
    net = bert_results["pooled_output"]
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name="sentiment_classifier")

    return tf.keras.Model(text_input, net)


def preprocess_text_material(input_text):

    bert_preprocess_model = hub.KerasLayer(bert_preprocessor)
    preprocessed_text = bert_preprocess_model([input_text])

    print(f'Keys       : {list(preprocessed_text.keys())}')
    print(f'Shape      : {preprocessed_text["input_word_ids"].shape}')
    print(f'Word Ids   : {preprocessed_text["input_word_ids"][0, :12]}')
    print(f'Input Mask : {preprocessed_text["input_mask"][0, :12]}')
    print(f'Type Ids   : {preprocessed_text["input_type_ids"][0, :12]}')
    return preprocessed_text


def training_sentiment_classifier(training_data, validation_data, testing_data):

    simple_sentiment_analysis_model = build_sentiment_classifier()

    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    accuracy_metrics = tf.metrics.BinaryAccuracy()

    epochs = 5
    steps_per_epoch = tf.data.experimental.cardinality(training_examples).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw') # this uses the gradient descent
    simple_sentiment_analysis_model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=accuracy_metrics
    )

    simple_sentiment_analysis_model.fit(
        x=training_data,
        validation_data=validation_data,
        epochs=epochs
    )

    loss, accuracy = simple_sentiment_analysis_model.evaluate(testing_data)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    # save the model
    saving_model_path = "./sentiment_analysis_bert"
    simple_sentiment_analysis_model.save(saving_model_path, include_optimizer=False, overwrite=True)


# def process_text_material_through(text_input):
#     model_name = "bert-base-uncased"
#     bert_model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=3)
#     tokenizer = BertTokenizer.from_pretrained(model_name)

#     tokens = tokenizer.encode_plus(text_input, return_tensors="tf", max_length=128, truncation=True)
#     bert_results = bert_model(tokens)
#     logits = bert_results.logits

#     predicted_sentiment = tf.argmax(logits, axis=1).numpy()
#     print("predicted sentiments: ", predicted_sentiment)



def main():
    # text_embending_layer()
    # preprocessed = preprocess_text_material(training_examples[0])
    # process_text_material_through(training_examples[0])


if __name__ == '__main__':
    main()