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

I received many positive feedbacks for my review note of the first half of the class. I am therefore motivated to continue working on a similar post for the second half. Again, I am posting my notes on my blog so it can benefit more people, no matter he/she is in the class or not :)

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

## Clustering (supervised evaluation)

When evaluating clustering with the ground truth, note that labels do not matter -- [0,1,0,1] is exactly the same as [1,0,1,0]. We should only look at **partition**.

### Why can't we use accuracy score

The problem in using accuracy in clustering problem is that it requires *exactly match* between the ground truth and the predicted label. However, the cluster labels themselves are meaningless -- as mentioned above, we should only care about partition, not labels!

### Contigency matrix
One tool we will use is contingency matrix. It is similar as confusion matrix, except that it does not have to square, and switching of rows/columns will not change the result.

### Rand Index, Adjusted Rand Index, Normalized Mutual Information and Adjusted Mutual Information

Rand index measures the similarity between two clustering. The formula is $$RI(C_1,C_2) = \frac{a+b}{{n}\choose{2}}$$, where $$a$$ is the number of pairs of points that are in the same set in both cluster $$C_1$$ and $$C_2$$, while $b$ is the number of pairs of points that are in different sets in $C_1$ and $C_2$. The denominator is just number of all possible pairs.

It can be intuitively understood if we view each pair as a data point, and treat this problem as using $$C_2$$ to predict $$C_1$$ (the ground truth). (Or the other way around, it's symmetric).

We count the number of true positive (they are in the same cluster in $$C_1$$, and $$C_2$$ predicts that they are also in a same cluster), plus the number of true negative (they are not in the same cluster in $$C_1$$, and $$C_2$$ predicts that they are also not in a same cluster) Sounds familiar now? Yes, it is just an analogy of accuracy!

**Sanity Check Question: What is R([0,1,0,1], [1,0,0,2])?**

Rand Index always ranges between 0 and 1. The bigger the better.

Adjusted Rand Index (ARI) is introduced to ensure to have a value close to 0.0 for random labeling independently of the number of clusters and samples and exactly 1.0 when the clusterings are identical (up to a permutation). **ARI penalizes too many clusters**. **ARI can become negative**.

Note: ARI requires the knowledge of ground truth. Therefore, **ARI is not a practical way to assess clustering algorithms like K-Means.**

Furthermore, we have normalized mutual information (which penalizes overpatitions via entropy) and adjusted mutual information (adjust for chance, so any two random partitions have expected AMI of 0).


## Clustering (unsupervised evaluation)

### Silhouette Score

Formula:

For each sample, calculate $$ s = \frac{b-1}{\max(a,b)}$$, where $$a$$ is mean distance to samples in same cluster, $$b$$ is the mean distance to samples in nearest cluster.

For whole clustering, we average s over all samples.

This scoring prefers compact clusters (like K-means).

Rationale: we want to maximize the difference between $$b$$ and $$a$$, so that the result is *decoupling and cohesion* (sounds like object-oriented programming hah?)

Cons: While compact clusters are good, compactness doesn’t allow for complex shapes.

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

# Working with imbalanced data

## Change threshold

```python
y_pred = lr.predict_proba(X_test)[:, 1] > .85 # change threshold to 0.85
```

**Sanity check question: for the above code, would you expect the precision of predicting positive (class 1) to increase or decrease? How about recall? How about support?**

## Sampling approaches

### Random undersampling

Drop data from the majority class randomly, until balanced.

Pros: very fast training, really good for large datasets

Cons: Loses data

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(replacement = False)
X_train_subsample, y_train_subsample = rus.fit_sample(X_train, y_train)
```

Use make_pipeline in imblearn:

```python
from imblearn.pipeline import make_pipeline as make_imb_pipeline
undersample_pipe = make_imb_pipeline(RandomUnderSampler(), LogisticRegressionCV())
scores = cross_val_score(undersample_pipe, X_train, y_train, cv=10)
```

### Random oversampling

Repeat data from the minority class randomly, until balanced.

Pros: more data (although many duplication)

Cons: MUCH SLOWER (and sometimes, the accuracy will get lower)

```python
from imblearn.pipeline import make_pipeline as make_imb_pipeline
oversample_pipe = make_imb_pipeline(RandomOverSampler(), LogisticRegressionCV())
scores = cross_val_score(oversample_pipe, X_train, y_train, cv=10)
```

### Class-weights

Instead of repeating samples, we can just re-weight the loss function. It has the same effect as over-sampling (though not random), but not as expensive and time consuming.

```python
from sklearn.linear_model import LogisticRegression
scores = cross_val_score(LogisticRegression(class_weight="balanced"), X_train, y_train, cv=5)
```

### Ensemble resampling

Random resampling for each model, and then ensemble them.

Pros: As cheap as undersampling, but much better results

Cons: Not easy to do right now with sklearn and imblearn

```python
# Code for Easy Ensemble
probs = [] 
for i in range(n_estimators):
	est = make_pipe(RandomUnderSampler(), DecisionTreeRegressor(random_state=i))
	est.fit(X_train, y_train)
	probs.append(est.predict_probab(X_test, y_test)) 
pred = np.argmax(np.mean(probs, axis=0), axis=1)
```

### Edited Nearest Neighbors

Remove all samples that are misclassified by KNN from training data (mode) or that have any point from other class as neighbor (all). Can be used to clean up outliers or boundary cases.

```python
from imblearn.under_sampling import EditedNearestNeighbours

# what? it's NearestNeighbours with u and n_neighbors without u 
# @.@ Great API design...
enn = EditedNearestNeighbours(n_neighbor=5) 
X_train_enn, y_train_enn = enn.fit_sample(X_train, y_train)

enn_mode = EditedNearestNeighbours(kind_sel = "mode", n_neighbor=3)
X_train_enn_mode, y_train = enn_mode.fit_sample(X_train, y_train)
```

### Condensed Nearest Neighbors

Iteratively adds points to the data that are misclassified by KNN. Contrast to Edited Nearest Neighbors,this resampling method focuses on the boundaries.

```python
from imblearn.under_sampling import CondensedNearestNeighbour
# CNN is not convolutional neural net XD
cnn_pipe = make_imb_pipeline(CondensedNearestNeighbour(), LogisticRegressionCV())
scores = cross_val_score(cnn_pipe, X_train, y_train, cv=10)
```

### Synthetic Minority Oversampling Technique (SMOTE)

Add synthetic (artificial) interpolated data to minority class.

Algorithm
- picking random neighbors from k neighbors.
- pick a point on the line between those two uniformly.
- repeat.

Pros: allows adding new interpolated samples, which works well in practice; There are many more advanced variants based on SMOTE.

Cons: leads to very large datasets (as it is doing oversampling), but can be mitigated by combining with undersampled data.

```python
from imblearn.over_sampling import SMOTE
smote_pipe = make_imb_pipeline(SMOTE(), LogisticRegressionCV())
scores = cross_val_score(smote_pipe, X_train, y_train, cv=10)
```

# Clustering and Mixture Model

## K-Means algorithm

Algorithm:
- Pick number of clusters k.
- Pick k random points as “cluster center”.
- While cluster centers change:
	– Assign each data point to it’s closest cluster center.
	- Recompute cluster centers as the mean of the assigned points.

Code:

```python
km = KMeans(n_clusters=5, random_state=42)
km.fit(X)
print(km.cluster_centers_.shape)
# km.labels_ is basically the predict
print(km.labels_shape)
print(km.predict(X).shape)
```

Note:
- Clusters are Voronoi-diagrams of centers, so always convex in space.
- Cluster boundaries are always in the middle of the centers.
- Cannot model covariance well.
- Cannot 'cluster' complicated shape (say two-moons dataset, which I usually refer to as the dataset where two bananas "interleaving" together).
- K-means performance relies on initialization. By default K-means in sklearn does 10 random restarts with different initializations.
- When dataset is large, consider using random, in particular for MiniBatchKMeans.
- k-means can also be used as fetaure extraction, where cluster membership is the new categorical feature and cluster distance is the continuous feature.

## Agglomerative clustering

Algorithm:
- Start with all points in their own cluster.
- Greedily merge the two most similar clusters until reaching number of samples required.

Merging criteria:
- Complete link (smallest maximum distance).
- Average linkage (smallest average distance between all pairs in the clusters.
- Single link (smallest minimum distance).
- Ward (smallest increase in with-in cluster variance, which normally leads to more equally sized clusters).

Pros:
- Can restrict to input “topology” given by any graph, for example neighborhood graph.
- Fast with sparse connectivity.
- Hierarchical clustering gives more holistic view, can help with picking the number of clusters.

Cons:
- Some linkage criteria may lead to very imbalanced cluster sized (depending on the scenario, it can be a benefit!).

Code:

```python
from sklearn.cluster import AgglomerativeClustering
for connectivity in (None, knn_graph):
	for linkage in ('ward', 'average', 'complete'):
		clustering = AgglomerativeClustering(linkage=linkage, connectivity=connectivity, n_clusters=10)
		clustering.fit(X)
```

## DBSCAN

Algorithm:

- Sample is "core sample" if more than min_samples is within epsilon ("dense region").
- Start with a core sample.
- Recursively walk neighbors that are core-samples and add to cluster.
- Also add samples within epsilon that are not core samples (but don’t recurse)
- If can’t reach any more points, pick another core sample, start new cluster.
- Remaining points are labeled outliers.

Pros:
- Can cluster well in complex custer shapes (two-moons would work!)
- Can detect outliers

Cons:
- Needs to adjust parameters (epsilon is hard to pick)


## Mixture Models

(Gaussian) Mixture Model is a generative model, where we assume that the data is formed in a generating process.

Assumptions:
– Data is mixture of small number of known distributions (in GMM, it's Gaussian). Each mixture component follows some other distribution (say, multinomial)
– Each mixture component distribution can be learned “simply”.
– Each point comes from one particular component.

EM algorithm:
- This is a non-convex optimization problem, so gradient descent won't work well.
- Instead, sometimes local minimum is good enough, and we can get there through Expectation Maximization algorithm (EM)

Code:
```python
from sklearn.mixture import GassuainMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
print(gmm.means_) # If X is of two dimension, returns 3 2D vectors
print(gmm.covariances_) # If X is of two dimension, returns 3 2x2 matrices
gmm.predict_proba(X) # For each data point, what is the probability of it being in each of the three classes?
print(gmm.score(X)) # Compute the per-sample average log-likelihood of the given data X.
print(gmm.score_samples(X)) # Compute the weighted log probabilities for each sample. Returns an array
```

Note:
- In high dimensions, covariance=”full” might not work.
- Initialization matters. Try restarting with different initializations.
- It allows partial_fit, meaning you can evaluate the probability of a new point under a fitted model.

## Bayesian Infinite Mixtures

Note:
- Bayesian treatment adds priors on mixture coefficients and Gaussians, and can unselect components if they do not contribute, so it is possibly more robust.
- Infinite mixtures replace Dirichlet prior over mixture coefficients by Dirichlet process, so it can automatically find number of components based on prior.
- Use variational inference (as opposed to gibbs sampling).
- Needs to specify upper limit of components.


## A zoo of clustering algorithm [^2]

![Plot cluster comparison](http://scikit-learn.org/dev/_images/sphx_glr_plot_cluster_comparison_001.png)

[^2]: Source: http://scikit-learn.org/dev/auto_examples/cluster/plot_cluster_comparison.html

## On picking the "correct" number of clusters

Sometimes, the right number of clusters does not even have a deterministic answer. So very likely manual check needs to be involved at the end.

But there are some tools that may be helpful:

### Silhouette Plots [^3]

![Silhouette plots](http://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_silhouette_analysis_005.png)

[^3]: Source: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

### Cluster Stability

The idea is that the configuration that yields the most consistent result among perturbations is best.

Idea:

- Draw bootstrap samples
- Do clustering
- Store clustering outcome using origial indices
- Compute averaged ARI


### Qualitative Evaluation (Fancy name for eyeballing)

Things to look at:
- Low-dimension visualization
- Individual points
- Clustering centers (if available)

### GridSearchCV (if doing feature extraction)

```python
km = KMeans(n_init = 1, init = "random")
pipe = make_pipeline(km, LogisticRegression())
param_grid = {'kmeans__n_clusters': [10, 50, 100, 200, 500]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X,y)
``` 

n_clusters: For preprocessing, larger is often better; for exploratory analysis: the one that tells you the most about the data is the best.