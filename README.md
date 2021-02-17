# Machine Learning 

## PS5

### Naive Bayes Model
In this project, I used a Naive Bayes classifier to build a prediction model for whether a review is fresh or rotten, depending on the text of the review.
I converted raw text fields into "bag of words" vectors, i.e. a data structure that tells you how many times a particular word appears in a blurb. Then I computed the likelihood and log-likelihood of my data as a way to assess the performance of my model. Then I cross-valid by tuning two most important hyperparameters to get the best model.

* The min_df keyword in CountVectorizer, which will ignore words which appear in fewer than min_df fraction of reviews. Words that appear only once or twice can lead to overfitting, since words which occur only a few times might correlate very well with Fresh/Rotten reviews by chance in the training dataset.

* The alpha keyword in the Bayesian classifier is a "smoothing parameter" -- increasing the value decreases the sensitivity to any single feature, and tends to pull prediction probabilities closer to 50%.

### Interpretation
I measured the impact of a word on freshness rating to get top 10 words best predict a fresh or rotten review. and found that Top 10 words that best predict fresh review are mostly positive amd top 10 words that best predict rotten review are mostly negative.

### Error Analysis
One of the best sources for inspiration when trying to improve a model is to look at examples where the model performs poorly.

The characteristics of misclassifications example are that the comments contain negative sentences and transitional conjunctions. This is because in Naive Bayes, we assume that words are independent of each other, but these examples directly violate this assumption, leading to wrong classification.
