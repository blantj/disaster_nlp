# Disaster Messages NLP

## Introduction
My modeling goal was to build an nlp model that could classify whether tweets were related to disasters or not .  I downloaded the tweet data in csv form from Kaggle and read it into pandas, resulting in 7,163 datapoints.  I pre-processed the data using nltk and modeled it using Naive Bayes, Adaboost, XGBoost and Soft Voting Classifier models.  Naive bayes and the Soft Voting Classifier were the top performing models with test set f1 scores of .70, which compared favorably to the baseline f1 score of .43.

## Obtain Data
I downloaded the disaster tweet dataset in csv form from Kaggle. The raw dataset included 7,163 datapoints across 5 variables.

## Scrub Data

<a href="url"><img src="https://github.com/blantj/disaster_nlp/blob/main/images/df_info.png" align="middle" height="250" width="400" ></a>

A df.info() revealed that the only scrubbing needed on the disaster tweet dataset was the removal of 3 columns not needed for modeling. After removing these 3 columns, the dataset included text data and labels for 7,613 datapoints.

In order to pre-process the text data, I first tokenized the reviews. I next removed nltk’s standard list of stopwords as well as custom stopwords that were within the 25 most frequent tokens of the disaster and non-disaster classes. I finally stemmed the tokens using nltk’s snowball stemmer.

## Explore Data

<a href="url"><img src="https://github.com/blantj/disaster_nlp/blob/main/images/class_balance.png" align="middle" height="250" width="250" ></a>

Plotting the class distribution of disaster vs. non-disaster messages revealed that 57% of messages were of the non-disaster class, while 43% were of the disaster class. While this class imbalance was not enough to necessitate re-balancing the classes, I would need to use f1 score as my principal evaluation metric in order to ensure that my model was weighting performance across the two classes equally.

<a href="url"><img src="https://github.com/blantj/disaster_nlp/blob/main/images/most_freq_words.png" align="middle" height="250" width="500" ></a>

A plot of the 25 most frequent tokens across the disaster and non-disaster classes revealed several stopwords common to the two plots including ‘the’, ‘i’ and ‘a’. I added these custom stopwords to the nltk list of stopwords for removal. The disaster class also included several unique disaster related-tokens that would be good for nlp modeling, such as ‘bomb’, ‘flood’ and ‘crash’.

<a href="url"><img src="https://github.com/blantj/disaster_nlp/blob/main/images/token_word_cloud.png" align="middle" height="175" width="500" ></a>

After removing the additional stopwords, the re-plotted 25 most frequent tokens across the two classes showed that the additional stopwords had not had much of an impact on the 25 most frequent tokens. This was likely due to the fact that the stopwords were removed pre-stemming, while the most frequent tokens were calculated post-stemming. With more time I would like to plot the 25 most frequent tokens before stemming in order to determine the pre-stemming custom stopwords common to both classes that need to be removed.


## Model Data

<a href="url"><img src="https://github.com/blantj/disaster_nlp/blob/main/images/model_performance.png" align="middle" height="175" width="400" ></a>

All four of the models outperformed the Dummy Classifier test set f1-score of .43. Naive Bayes and the Soft Voting Classifier were the top performing models with test set f1 scores of .70. I selected Naive Bayes as my preferred model as it was less complex than the soft voting classifier and had like performance.

## Analyze Results

<a href="url"><img src="https://github.com/blantj/disaster_nlp/blob/main/images/nb_confusion_matrix.png" align="middle" height="250" width="250" ></a>

Looking at the Confusion Matrix for Naive Bayes modeling, the model performed better at identifying non-disaster messages than disaster messages with a sensitivity of .84 and a specificity of .65. This isn’t surprising as the non-disaster class was slightly larger, so it makes sense that the model would lean towards overclassifying datapoints as a member of the non-disaster class.


# Github Files
[Modeling.ipynb](https://github.com/blantj/mushroom_classification/blob/master/Modeling.ipynb) :  Poisonous mushroom classification modeling

# Sources
Kaggle: https://www.kaggle.com/c/nlp-getting-started
