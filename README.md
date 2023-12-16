# headline-sentiment-analysis

### Overview
Given a keyword and time period, 
- the analyzer calls the NewsApi to gather all news material regarding that keyword during that period
- gather all the news headlines and tokenize them using the Bert tokenizer
- runs the tokens as input through the `nlptown/bert-base-multilingual-uncased-sentiment` to obtain the predicted sentiment of each
  headline which is then consolidated to get the overall sentiment of the keyword during the given time period

### Usage
this tool currently only runs on the command line but a React UI will be added soon to visualize the results

#### sample run
```
> python headline_analyzer.py tesla "last week"

title: Tesla Is Suing Sweden, sentiment: negative
title: Tesla sues Sweden for blocking license plate deliveries during labor strike, sentiment: negative
title: Tesla Cybertruck will usher in a new ‘Powershare’ bidirectional charging feature, sentiment: positive

overall sentiment: negative
```