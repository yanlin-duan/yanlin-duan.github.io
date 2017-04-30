---
title: "My Review Note for Applied Machine Learning (Second Half)"
excerpt: "Good luck with the final!"
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

# Dimensionality Reduction

## Linear, Unsupervised Transformation -- PCA

PCA rotates the dataset so that the rotated features are statistically uncorrelated. It first finds the direction of maximum variance, and then finds the direction that has maximum variance but at the same time **is orthogonal** to the first direction (thus making those two rotated features not correlated), so on and so forth.

When to use: PCA is commonly used for linear dimension reduction (select up to first k principal components), visualization of high-dimensional data (draw first v.s. second principal components), regularization and feature extraction (for example, comparing distance in pixel space does not really make sense; maybe using PCA space will perform better)

Whitening: rescale the principal components to have the same scale; Same as using StandardScaler after perfoming PCA.

### Why PCA (in general) works

PCA finds uncorrelated components that maximizes the variance explained in the data. However, only when the data follows Gaussian distribution, zero correlation between components implies independence, as the first and second order statistics already captures all the information. This is not true for most of the other distributions. 

Therefore, PCA 'sort of' makes an implicit assumption that data is drawn from Gaussian, and works the best when representing multivariate normally distributed data.

### Important notes

- PCA, compared to histograms or other tools, is used because it can capture the interactions between features.
- Do scaling before performing PCA. Imagine one feature with very large scale. Without scaling, it’s guaranteed to be the first principal component!
- PCA is unsupervised, so it does not use any class information.
- PCA has no guarantee that the top k principal components are the dimensions that **contains most information**. High variance $$!=$$ most information!
- Max number of principal components min(n_samples, n_features).
- Sign of the principal components does not mean **anything**.


### Sample Code
TODO.


## Non-linear, unsupervised transformation - t-SNE

t-distributed stochastic neighbor embedding (t-SNE) is an algorithm in the category of manifold learning. The high level idea of t-SNE is that it will find a two-dimensional representation of the data such that if they are 'similar' in high-dimension, they will be 'closer' in the reduced 2D space. To put it in another way, it tries to preserve the neighborhood information.

How t-SNE works: it starts with a random embedding, and iteratively updates points to make close points close.

The usage for t-SNE is now more on data visualization.

### Note
- t-SNE does not support transforming new data, so no transform method in sklearn
- Axes do not correspond to anything in the input space, so merely for visualization purpose.
- To tune t-SNE, tune perplexity (low perplxity == only close neighbors) and early_exaggeration parameters, though the effects are usually minor.


## Linear, supervised transformation -- Linear Discriminant Analysis


![Comparison Between PCA and LDA](http://sebastianraschka.com/images/blog/2014/linear-discriminant-analysis/lda_1.png) [^1]

[^1]: Source: http://sebastianraschka.com/Articles/2014_python_lda.html#principal-component-analysis-vs-linear-discriminant-analysis

Linear Discriminant Analysis is a “supervised” generative model that computes the directions (“linear discriminants”) that will maximize the separation between multiple classes. LDA assumes data to be drawn from Gaussian distributions (just as PCA, but for each class). It further assumes that features are statistically independent, and identical covariance matrices for every class.

LDA can be used both as a classifier and a dimensionality reduction techinique. The advantage is that it is a supervised model, and there's no parameters to tune. It is also very fast since it only needs to compute means and invert covariance matrices (if number of features is way less than number of samples).

A variation is Quadratic Discriminant Analaysis, where basically each class will have separate covariance matrices.

