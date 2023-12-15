# headline-sentiment-analysis

### Overview
Given a keyword and time period, 
- the analyzer calls the NewsApi to gather all news material regarding that keyword during that period
- gather all the news headlines and tokenize them using the `BertTokenizer`
- runs the tokens as input through the `SentimentAnalysisModel` to obtain the predicted sentiment of each
  headline which is then consolidated to get the overall sentiment of the keyword during the given time period

### `SentimentAnalysisModel`
this model was first trained on financial phrases because it was initially intended to analyze financial news,
but the use case was expanded to include all types of news from various sources. </br>
collab notebook training the model: https://colab.research.google.com/drive/1-HV8IoTcBC_MiZ0KV2TTHWuOjHZuGD1R?usp=sharing </br>
hugging face repo which has the model and all its parameter configurations: https://huggingface.co/ghislainehaha/headline-sentiment-analyzer

### Usage
this tool currently only runs on the command line but a react UI will be added soon to visualize the results
