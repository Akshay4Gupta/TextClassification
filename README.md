# TextClassification
In this problem, we will use the Naı̈ve Bayes algorithm for classification of tweets by different twitter users.
The dataset for this problem can be obtained from http://help.sentiment140.com/for-students. Given a user’s tweet, task is to predict
the sentiment (Positive, Negative or Neutral) of the tweet. Read the website for more details about the
dataset. The dataset contains separate training and test files containing 1.6 Million training samples and
498 test samples, respectively.

Q1) (a, c) Implement the Naı̈ve Bayes algorithm to classify each of the tweets into one of the given
categories. Report the accuracy over the training as well as the test set.

Time:	 12.84270167350769
----------------------------------Q1(a)-------------------------------------
accuracy over the training:  84.932875

confusionMatrix:	 negative 	 positive
negative(predicted)	 712613 	 153687
positive(predicted)	 87387 	 646313

Time:	 41.86461043357849
----------------------------------Q2(a)-------------------------------------
accuracy over the test:  80.77994428969359

confusionMatrix:	 negative 	 positive
negative(predicted)	 142 		 34
positive(predicted)	 35 		 148

Time:	 41.87290859222412
----------------------------------

Q2) Removing twitter handles and stemming and removing stop words

by removing only twitter handles and nothing else i.e. no tokenize or split is done 
Time:	 396.9018154144287
----------------------------------Q1(a)-------------------------------------
accuracy over the training:  79.7666591259305

confusionMatrix:	 negative 	 positive
negative(predicted)	 654398 	 179548
positive(predicted)	 143331 	 618500

Time:	 826.0535900592804
----------------------------------Q2(a)-------------------------------------
accuracy over the test:  82.17270194986072

confusionMatrix:	 negative 	 positive
negative(predicted)	 143 		 30
positive(predicted)	 34 		 152

Time:	 826.1673648357391
----------------------------------
