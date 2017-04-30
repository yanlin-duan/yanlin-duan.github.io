---
title: "My Review Note for Applied Machine Learning (Second Half)"
excerpt: "Good luck with the final!."
excerpt_separator: "<!--more-->"
categories:
  - Machine Learning
tags:
  - Applied Machine Learning
  - Machine Learning
  - Learning Diary
mathjax: true
---
{% include toc %}

# Why this post

This semester I am taking Applied Machine Learning with [Andreas Mueller](http://amueller.github.io/). It's a great class focusing on the practical side of machine learning.

I received many positive feedbacks for my review note of the first half of the class. I am therefore motivated to continue working on a similar post for the second half. Again, I am posting my notes on my blog so it can benefit more people, whether he/she is in the class or not :)

# Acknowledgment

The texts of this note are largely inspired by:

- [Course material](https://amueller.github.io/applied_ml_spring_2017/) for COMS 4995 Applied Machine Learning.

The example codes in this note are modified based on:

- [Course material](https://amueller.github.io/applied_ml_spring_2017/) for COMS 4995 Applied Machine Learning.
- Supplemental material of *An Introduction to Machine Learning with Python* by Andreas C. Müller and Sarah Guido (O’Reilly). Copyright 2017 Sarah Guido and Andreas Müller, 978-1-449-36941-5.

Care has been taken to avoid copyrighted contents as much as possible, and give citation wherever is proper.

# Model Evaluation Metrics

## Classification

### Why do we need precision, recall and f-score

It is very natural to evaluate the performance of a model by looking at its **accuracy** -- meaning out of all testing data, how many we get *correct* (predict true when it should be true, and predict false when it should be false).

However, this measurement becomes less effective if the data is imbalanced -- meaning we have way more data in one class compared to others. For example -- if 99% of the data are labeled as 1, a model can simply `cheat` by always predicting 1 -- a naive, trivial but high accuracy model. In those cases, we need better measurements, and that's why we introduce precision and recall.

I have had a long time memorizing which one is which, and there are so many combinations between True Positive, False Positive, True Negative and False Negative so I almost always get confused. I found it's actually easier if we take a step back and first intuitively understand the word `precision` and `recall` (yes, the name is not a random one!)

When we say precision, we are talking about how precise you are. For example, if I am searching something on Google, I will say the precision is high when *out of everything Google returns to me*, I found most of them relevant to what I want to search for.

This means, I am measuring the proportion of correct search results (True positive) over everything Google predicts to be "what I want" (True positive and False positive).

For recall, I find it easier to understand it in an "ecology" context -- in ecology, there is a method called **"mark and recapture"** where a portion of the population of, say, an insect is captured, marked and released (we remember how many we marked). Later, another portion is captured. We are now interested in **out of all the marked insects**, how much do we recall.

Back to machine learning, this naturally translates to the proportion of *"recaptured and marked ones"* (true positive) over *all marked insects* (true positive + false negative, i.e. those mark insects we recapture + fail to recapture).

**Sanity Check time: if this review note is to predict topics covered in class, what should I include if I want to have a high precision? (Hint: count # of parameters in a neural network) How about a high recall?**

As you can see from the sanity check question, it is not hard for models to achieve a perfect recall or a perfect precision *alone*. Therefore, we would like to summarize them -- and that's what f-score, the harmonic mean of precision and recall, is doing.

### Other common tools

- Precision-Recall Curve. Area under it is the **average precision** (ignoring some technical differences). Ideal curve --upper right.

- Receiver operating characterisitics (ROC), which is FPR ($$FP/(FP+TN)$$)v.s. TPR (recall). AUC is the area under the curve, which does not depend on threshold selection. AUC is always 0.5 for random prediction, regardless of whether the class is balanced. The AUC can be interpreted as evaluating the ranking of positive samples. Ideal curve -- upper left.


### Multi-class Classification Metrics

- Confusion Matrix and classification report (note: support means  number of data points (ground truth) in that class)
- Micro-F1 (each data point to be equal weight) and Macro-F1 (each class to be equal weight)

## Regression

### Built-in standard metrics

- $$R^2$$: a standardized measure of degree of predictedness, or fit, in the sample. Easy to understand scale. 
- MSE: estimate the variance of residuals, or non-fit, in the population. Easy to relate to input
- Mean Absolute Error, Median Absolute Error, Mean Absolute Percentage Error etc.
- $$R^2$$ is still the most commonly used one.

## Sample code for choosing evaluation metrics in sklearn

TODO.