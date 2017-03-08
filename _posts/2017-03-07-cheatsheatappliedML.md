---
title: "My Cheatsheet for Applied Machine Learning Class"
excerpt: "Summarizes what we have learnt so far."
excerpt_separator: "<!--more-->"
categories:
  - Machine Learning
tags:
  - Applied Machine Learning
  - Machine Learning
  - Learning Diary
mathjax: true
---

# Why this post

This semester I am taking Applied Machine Learning with [Andreas Mueller](http://amueller.github.io/). It's a great class focusing on the practical side of machine learning.

As the midterm is coming, I am revising for what we have covered so far in the class, and think that preparing a cheatsheet would be an effective way to do so (though the exam is closed book). I am posting my notes here so it can benefit more people.

# Introduction to Machine Learning

## Type of machine learnings

- Supervised (function approximation + generalization; regression v.s. classification)
- Unsupervised (clustering, outlier detection)
- Reinforcement Learning (explore & learn from the environment)
- Others (semi-supervised, active learning, forecasting, etc.)

## Parametric and Non-parametric models
- Parametric model: Number of “parameters” (degrees of freedom) independent of data.
  - e.g.: Linear Regression, Logistic Regression, Nearest Shrunken Centroid
- Non-parametric model: Degrees of freedom increase with more data. Each training instance can be viewed as a "parameter" in the model, as you use them in the prediction.
  - e.g.: Random Forest, Nearest Neighbors

## Classification: From binary to multi-class
- One v.s. Rest (OvR) (standard): needs n binary classifiers; predict the class with highest score.
- One v.s. One (OvO): needs $$n \cdot (n-1) / 2$$ binary classifiers; predict by voting for highest positives

## How to formalize a machine learning problem in general

$$ \min_{f \in F} \sum_{i=1}^N{L(f(x_i),y_i) + \alpha R(f)} $$

We want to find the $$f$$ in function family $$F$$ that minimizes the error (risk, denoted by function $$L$$) on the training set, and at the same time keeps it simple (denoted by the regularized term $$R$$ and $$\alpha$$).

## Decomposing Generalization Error (Bottou et. al, picture from Applied ML course note aml-06, page 28 & 29)

![Decomposing Generalization Error]({{ site.url }}/assets/pics/decompose_error.png)
![Decomposing Generalization Error 2]({{ site.url }}/assets/pics/decompose_error_2.png)

## Difference between Machine Learning and Statistics

| ML        | Statistics           |
|:-------------:|:-------------:|
| Data First     | Model First |
| Prediction + Generalization[^1]   | Inference      |

[^1]: In principle we don’t care too much about performance on training data, but on new samples from the same distribution.

## Guideline Principles in Machine Learning
- Defining the goal, and measurement (metrics) of the goal
- Thinking about the context: baseline and benefit
- Communicating the result: how explainable is the model/result?
- Ethics
- Data Collection (More data? What is the cost?)

## The Machine Learning Workflow [^2]
![ML Workflow](https://www.mapr.com/ebooks/spark/images/mllib_rec_engine_image006.png)

[^2]: Source: https://www.mapr.com/ebooks/spark/08-recommendation-engine-spark.html

## Information Leakage

> Data Leakage is the creation of unexpected additional information in the training data, allowing a model or machine learning algorithm to make unrealistically good predictions. Leakage is a pervasive challenge in applied machine learning, causing models to over-represent their generalization error and often rendering them useless in the real world. It can caused by human or mechanical error, and can be intentional or unintentional in both cases.

Source: https://www.kaggle.com/wiki/Leakage

Common mistakes include:
- Keep features that are not available in new data
- Leaking of information from the future into the past
- Do preprocessing on the whole dataset (before train/test split)
- Test on test data sets multiple times


# Git

For git, I have found the following 2 YouTube videos very helpful:

[![Git For Ages 4 And Up](http://img.youtube.com/vi/1ffBJ4sVUb4/0.jpg)](http://www.youtube.com/watch?v=1ffBJ4sVUb4)

[![Learn Git in 20 Minutes](http://img.youtube.com/vi/Y9XZQO1n_7c/0.jpg)](http://www.youtube.com/watch?v=Y9XZQO1n_7c)

Below I summarized some key points about git:

- Create/Remove repository:
```bash
git init # use git to track current directory
rm .git # undo the above (your files are still there)
```

- Typical workflow:
```bash
git clone [url] # clone a remote repository
git branch newBranch # create a new branch
git checkout newBranch # say "Now I want to work on that branch"
# do your job...
git add this_file # add it to staging area
git commit # Take a snapshot of the state of the folder, with a commit message. It will be identified with an ID (hash value)
git push origin master # push from local -> remote
git pull origin master # pull from remote -> local
git merge A # merge branch A to current branch
```

- My favorite shortcuts/commands:
```bash
git checkout -b newBranch # create branch and checkout in one line
git add -A # update the indices for all files in the entire working tree
git commit -a # stage files that have been modified and deleted, but not new files you have not done git add with
git commit -m <msg> # use the given <msg> as the commit message.
git stash # saves your local modifications away and reverts the working directory to match the HEAD commit. Can be used before a git pull
```

- Other important ones (in lecture notes or used in Homework 1):
```bash
git reset --soft <commit> # moves HEAD to <commit>, takes the current branch with it
git reset --mixed <commit> # moves HEAD to <commit>, changes index to be at <commit>, but not working directory
git reset --hard <commit> # moved HEAD to <commit>, changes index and working tree to <commit>
git rebase -i <commit> # interactive rebase
git rebase --onto feature master~3 master # rebase everything from master~2 (master - 3 commits, excluding this one) up until the tip of master (included) to the tip of feature.
git reflog show # show reference logs that records when the tips of branches and other references were updated in the local repository.
git checkout HEAD@{3} # checkout to the commit where HEAD used to be three moves ago
git checkout feature this_file # merge the specific file (this_file) from feature to your current branch
git log # show git log
```

- The hardest part of git in my opinion is the **"polymorphism"** of git commands. As shown above, you can do git checkout on a branch, a commit, a commit + a file, and they all mean different things. (This motivates me to write a git tutorial in the future when I have time, where I will go through the common git commands in a different way as existing tutorials.)

- Difference (Relationship) between git and github: people new to git may be confused by those two. In one sentence: Git is a version control tool, and GitHub is an online project hosting platform using git.(Therefore, you may use git with or without Github.)

- Git add and staging area[^3]:

![Staging Area](https://git-scm.com/book/en/v2/images/areas.png)

[^3]: Source: https://git-scm.com/book/en/v2/Getting-Started-Git-Basics

- Fast-forward [^4] (Note that no new command is created):

![What is fast-forward](https://ariya.io/images/2013/09/merging.png)

[^4]: Source: https://ariya.io/2013/09/fast-forward-git-merge

- What is HEAD^ and HEAD~ [^5]:

![HEAD^ and HEAD~ in git]({{ site.url }}/assets/pics/git_1.png)

[^5]: Source: http://schacon.github.io/git/git-rev-parse#_specifying_revisions

- Github Pull Request:
  - Pull requests allow you to contribute to a repository which you don't have permission to write to. The general workflow is: fork -> clone to local -> add a feature branch -> make changes -> push.

  - To keep update with the upstream, you may also need to: add upstream as another remote -> pull from upstream -> work upon it -> push to your origin remote.

# Coding Guidelines:

## Good resources (and books that I really like):
- [Pep 8](https://www.python.org/dev/peps/pep-0008/)
- [*Refactoring: Improving the Design of Existing Code* by Kent Beck and Martin Fowler](https://www.amazon.com/Refactoring-Improving-Design-Existing-Code/dp/0201485672)
- [*Clean Code: A Handbook of Agile Software Craftsmanship* by Robert Cecil Martin](https://www.amazon.com/Refactoring-Improving-Design-Existing-Code/dp/0201485672)

# Python:

## Python Quick Intro:
- Powerful
- Simple Syntax
- Interpreted language: slow (many libraries written in C/Fortran/Cython)
- Python 2 v.s. Python 3 (main changes in: division, print, iterator, string; need something from python 3 in python 2? do `from __future__ import bla`)
- For good practices, always use explicit imports and standard naming
conventions. (Don't `from bla import *`!)

# Testing and Documentation:

## Different kinds of tests
- Unit Tests: a function is doing the right thing; Can be done with pytest
- Integration tests: functions together are doing the right thing; Can be done with TravisCI (continuous integration)
- Non-regression tests: bugs truly get removed

## Different ways of doing documentation:
- PEP 257 for docstrings and inline comments
- NumpyDoc format
- Various tools for generating documentation pages: SPhinx, ReadTheDocs

# Visualization -- Exploration and Communications

## Visual Channels: Try not to...
- Use 3D-volume to show information
- Use textures to show information
- Use hues for quantitative changes
- Use bad colormaps such as jet and rainbow. They vary non-linearly and non-monotonically in lightness, which can create edges in images where there are none. The varying lightness also makes grayscale print completely useless.

## Color maps:

| Sequential Colormaps        | Diverging Colormaps           | Quantitative Colormaps           | Miscellaneous Colormaps |
|:-------------:|:-------------:|:-------------:|:-------------:|
| Go from one hue/saturation to another (Lightness also changes)    | Grey/white (focus point) in the middle, different hues going in either direction | Use to show discrete values | **Don't use jet and rainbow!** (Andy will be disappointed if you do so @.@)|
|  Use to emphasize extremes | Use to show deviation from the neutral points      | Designed to have optimum contrast for a particular number of discrete values |  Use perceptual uniform colormaps|

## Matplotlib Quick Intro:
- `% matplotlib inline` v.s. `% matplotlib notebook` in Jupyter Notebook
-  Figure and Axes:
  - Create automatically by doing plot command
  - Create by `plt.figure()`
  - Create by `plt.subplots(n,m)`
  - Create by `plt.subplot(n, m, i)`, where i is 1-indexed, column-based position
- Two interfaces:
  - Stateful interface: applies to current figure and axes (e.g.: `plt.xlim`)
  - Object-oriented interface: explicitly use object (e.g.: `ax.set_xlim`)


## Important commands:
- Plot command `ax.plot(np.linspace(-4, 4, 100), sin, '--o', c = 'r', lw = 3)`
  - Use figsize to specify how large each plot is (otherwise it will be "squeezed")
  - Single variable x: plot it against its index; Two variables x and y: plot against each other
  - By default, it's line-plot. Use “o” to create a scatterplot
  - Can change the width, color, dashing and markers of the plot
- Scatter command: `ax.scatter(x, y, c=x-y, s=np.abs(np.random.normal(scale=20, size=50)), cmap='bwr', edgecolor='k')`
  - cmap is the colormap, bwr means blue-white-red
  - k is black
- Histogram: `ax.hist(np.random.normal(size=100), bins="auto")`
  - Use bins="auto" to heuristically choose number of bins
- Bar chart (vertical): `plt.bar(range(len(value)), value); plt.xticks(range(len(value)), labels, rotation=90)`
  - For bar chart, the length must be provided. This can be done using range and len.
- Bar chart (horizontal): `plt.barh(range(len(value)), value); plt.yticks(range(len(value)), labels, fontsize=10)`
- Heatmap: `ax[0, 1].imshow(arr, interpolation='bilinear')`
  - imshow essentially renders numpy arrays as images
- Hexgrids: `plt.hexbin(x, y, bins='log', extent=(-1, 1, -1, 1))`
  - hexbin is essentially a 2-D histogram with hexagonal cells. (which is used to show 2D density map)
  - It can be much more informative than a scatter plot
- TwinX: `ax2 = ax1.twinx()`
  - Show series in different scale much better

# Fight Against Overfitting

## Naive Way: No train/test Split

### Drawback
You never know how your model performs on new data, and you will cry.


## First Attempt: Train Test Split (by default 75%-25%)

### Code
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

### Drawback
If we use the test error rate to tune hyper-parameters, it will learn about noise in the test set, and this knowledge will not generalize to new data.

Key idea: **You should only touch your test data once**.

## Second Attempt: Three-fold split (add validation set)

### Code
```python
from sklearn.model_selection import train_test_split
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval)
```

### Pros
Fast and simple

### Cons
We lose a lot of data for evaluation, and the results depend on the particular sampling. (overfit on validation set)

## Third Attempt: K-fold Cross Validation + Train/Test split

### Idea
Split data into multiple folds and built multiple models. Each time test models on different (unused) fold.

### Code
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X_trainval, y_trainval, cv=10) # equiv to StratifiedKFold without shuffle
print(np.mean(scores), np.std(scores))
```
### Pros
- Each data point is in the test-set exactly once.
- Better data use (larger training sets)

### Cons
- It takes 5 or 10 times longer (you train 5/10 models)

## More CV strategies

### Code
```python
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
kfold = KFold(n_splits=10)
ss = ShuffleSplit(n_splits=30, train_size=.7, test_size=.3)
skfold = StratifiedKFold(n_splits=10)
```

### Explanation
- **Stratified K-Fold**: preserves the class frequencies in each fold to be the same as of the overall dataset
  - Especially helpful when data is imbalanced
- **Leave One Out**: Equivalent to `KFold(n_folds=n_samples)`, where we use n-1 samples to train and 1 to test.
  - Cons: high variance, and it takes a long time!
  - Solution: Repeated (Stratified) K-Fold + Shuffling: Reduces variance, so better!
- **ShuffleSplit**: Repeatedly and randomly pick training/test sets based on training/test set size for number of iterations times.
  - Pros: Especially good for subsample when data set is large
- **GroupKFold**: Patient example; where samples in the same group are highly correlated. New data essentially means new group. So we want to split data based on group.
- **TimeSeriesSplit**: Stock price example; Taking increasing chunks of data from the past and making predictions on the next chunk. Making sure you do not have access to the "future".

## Final Attempt 1: Use GridSearch CV that wraps up everything

### Code
```python
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

param_grid = {'n_neighbors':  np.arange(1, 20, 3)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=10)
grid.fit(X_train, y_train)
print(grid.best_score_, grid.best_params_) #grid also has grid.cv_results_ which has many useful statistics
```

### Note
- We still need to split our data into training and test set.
- If we do GridSearchCV on a pipeline, the param_grid's key should look like: `'svc__C:'`.

## Final Attempt 2: Use built-in CV for specific models

### Code
```
from sklearn.linear_model import RidgeCV
ridge = RidgeCV().fit(X_train, y_train)
print(ridge.score(X_test, y_test))
print(ridge.alpha_)
```
### Note
- Usually those CV are more efficient.
- Support: RidgeCV(), LarsCV(), LassoLarsCV(), ElasticNetCV(), LogisticRegressionCV().
- All have reasonable built-in parameter grids.
- For RidgeCV you can’t pick the “cv”!

# Preprocessing

## Scaling and Centering

### When to scale & centering
The following model examples are particularly sensitive on scale of features:
- KNN
- Linear Models

### When not to scale
The following model(s) is(are) not quite sensitive to scaling:
- Decision Tree

If data is sparse, do not center (make data dense). Only scale is fine.

### How to scale
- StandardScaler: subtract mean and divide by standard
deviation.
- MinMaxScaler: subtract minimum, divide by (max - min), resulting in range 0 and 1.
- Robust Scaler: uses median and quantiles, therefore robust to outliers. Similar to StandardScaler.
- Normalizer: only considers angle, not length. Helpful for histograms, not that often used.

### Code
```
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
for scaler in [StandardScaler(), RobustScaler(), MinMaxScaler(), Normalizer(norm='l2')]:
    X_ = scaler.fit_transform(X)
```

### Note
We should perform `scaler.fit` only on training data!

## Pipelines

Pipelines are used to solve the common need of linking preprocessing, models, etc. together and prevents information leakage.

### Code
```
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), Lasso())
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
print(pipe.steps)

# Or we can have pipeline with named steps
from sklearn.pipeline import Pipeline
pipe = Pipeline((("scaler", StandardScaler()),
                 ("regressor", KNeighborsRegressor)))

# Note how param_grid change when combining GridSearchCV with pipeline
from sklearn.model_selection import GridSearchCV
pipe = make_pipeline(StandardScaler(), SVC())
param_grid = {'svc__C': range(1, 20)}
grid = GridSearchCV(pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
score = grid.score(X_test, y_test)
```

## Feature Transformation

### Why do feature transformation
Linear models and neural networks, for example, perform better when the features are approximately normal distributed.

### Box-Cox Transformation
- Box-Cox minimizes skew, trying to create a more “Gaussian-looking” distribution.
- Box-Cox only works on positive features!

### Code
```
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
X_train_mm = MinMaxScaler().fit_transform(X_train) # Use MinMaxScaler to make all features positive
X_bc = []
for i in range(X_train_mm.shape[1]):
  X_bc.append(stats.boxcox(X_train_mm[:, i] + 1e-5))
```

## Discrete/Categorical Features

### Why it matters
It doesn't make sense to train the model (esp linear model) directly if the data set contains discrete features, where "0,1,2" means nothing but different category.

### Models that support discrete features
In theory, tree-based models do not care if you have categorical features. However, current scikit-learn implementation does not support discrete features in any of its models

### One-hot Encoding (Turn k categories to k dummy variables)

```
import pandas as pd
pd.get_dummies(df, columns=['boro'])

# alternatively, specified by astype
df = pd.DataFrame({'year_built': [2006, 1973, 1988, 1984, 2010, 1972],
                   'boro': ['Manhattan', 'Queens', 'Manhattan', 'Brooklyn', 'Brooklyn', 'Bronx']})
df.boro = df.boro.astype("category", categories=['Manhattan', 'Queens', 'Brooklyn', 'Bronx', 'Staten Island'])
pd.get_dummies(df)

# or, we can use one-hot encoder in scikit-learn
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({'year_built': [2006, 1973, 1988, 1984, 2010, 1972],
                   'boro': [0, 1, 0, 2, 2, 3]})
OneHotEncoder(categorical_features=[0]).fit_transform(df.values).toarray()
```

### Count-based Encoding
For high cardinality categorical features, instead of creating many dummy variables, we can create count-based new features based on it. For example, average response, likelihood, etc.


## Feature Engineering: Polynomial features

Sometimes we want to add features to make our model stronger. One way is to add interactive features, i.e. polynomial features.

### Code
```
from sklearn.preprocessing import PolynomialFeatures
poly_lr = make_pipeline(PolynomialFeatures(degree=3, include_bias=True, interaction_only=True), LinearRegression())
poly_lr.fit(X_train, y_train)
```

# Model: Neighbors

## KNN
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

## Nearest Centroid (find the mean of each class, and predict the one that is closet; resulting in a linear boundary)
```python
from sklearn.neighbors import NearestCentroid
nc = NearestCentroid()
nc.fit(X, y)
```

## Nearest Shrunken Centroid
```python
nc = NearestCentroid(shrink_threshold=threshold)
```

### Difference between Nearest Shrunken Centroid and Nearest Centroid [^6]
>  It "shrinks" each of the class centroids toward the overall centroid for all classes by an amount we call the threshold . This shrinkage consists of moving the centroid towards zero by threshold, setting it equal to zero if it hits zero. For example if threshold was 2.0, a centroid of 3.2 would be shrunk to 1.2, a centroid of -3.4 would be shrunk to -1.4, and a centroid of 1.2 would be shrunk to zero.

[^6]: Source: http://statweb.stanford.edu/~tibs/PAM/Rdist/howwork.html

# Model: Linear Regression

## Linear Regression (without regularization)

### Model
$$ \min_{w \in \mathbb{R}^d} \sum_{i=1}^N{||w^Tx_i - y_i||^2} $$

### Code
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
```

## Ridge (l2-norm regularization)

### Model
$$ \min_{w \in \mathbb{R}^d} \sum_{i=1}^N{||w^Tx_i - y_i||^2 + \alpha ||w||_2^2} $$

### Code
```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=10).fit(X_train, y_train) # takes alpha as a parameter
print(ridge.coef_) #can get coefficients this way
```

## Lasso (l1-norm regularization)

### Model
$$ \min_{w \in \mathbb{R}^d} \sum_{i=1}^N{||w^Tx_i - y_i||^2 + \alpha ||w||_1} $$

### Code
```python
from sklearn.linear_model import Lasso
lasso = Lasso(normalize=True, alpha=3, max_iter=1e6).fit(X_train, y_train)
```

### Note
Lasso can (sort of) do feature selection because many coefficients will be set to 0. This is particularly useful when feature space is large.

## Elastic Net (l1 + l2-norm regularization)

### Model
$$ \min_{w \in \mathbb{R}^d} \sum_{i=1}^N{||w^Tx_i - y_i||^2 + \alpha_1 ||w||_1 + \alpha_2 ||w||_2^2} $$

### Code

```python
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.6)
y_pred_test = enet.fit(X_train, y_train).predict(X_test)
```

## Random Sample Consensus (RANSAC)

### Idea
- Iteratively train a model and at the same time, detect outliers.
- It is non-deterministic in the sense that it produces a reasonable result only with a certain probability. The more iterations allowed, the high the probability.

### Code

```python
from sklearn.linear_model import RANSACRegressor
model_ransac = RANSACRegressor()

model_ransac.fit(X, y)
inlier_mask = model_ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
```

## Robust Regression (Huber Regressor)

### Idea
Minimizes what is called "Huber Loss", which makes sure that the loss function is not heavily affected by the outliers. At the same time, it will not completely ignore their influence.

### Code
```python
from sklearn.linear_model import HuberRegressor
huber = HuberRegressor(epsilon=1, max_iter=100, alpha=1).fit(X, y)
```

# Model: Linear Classification

## (Penalized) Logistic Regression

### Model
$$ \min_{w \in \mathbb{R}^d} - C\sum_{i=1}^N{\log(\exp(-y_iw^Tx_i) + 1)} + ||w||_1 $$
$$ \min_{w \in \mathbb{R}^d} - C\sum_{i=1}^N{\log(\exp(-y_iw^Tx_i) + 1)} + ||w||_2^2 $$

### Note
- The higher C, the less regularization. (inverse to $$\alpha$$)
- l2-norm version is smooth (differentiable)
- l1-norm version gives sparse solution / more compact model
- Logistic regression gives probability estimates
- In multi-class case, using OvR by default
- Solver: ‘liblinear’ for small datasets, ‘sag’ for large datasets and if you want speed; only ‘newton-cg’, ‘sag’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes; ‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty. More details [here](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Use Stochastic Average Gradient Descent solver for really large n_samples

### Code
```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg = LogisticRegression(multi_class="multinomial", solver="lbfgs").fit(X, y) # multi-class version
logreg.fit(X_train, y_train)
```

## (Soft margin) Linear SVM

### Model
$$ \min_{w \in \mathbb{R}^d} C\sum_{i=1}^N{\max(0, 1-y_iw^Tx)} + ||w||_1 $$
$$ \min_{w \in \mathbb{R}^d} C\sum_{i=1}^N{\max(0, 1-y_iw^Tx)} + ||w||_2^2 $$

### Note
- Both versions are strongly convex, but neither is smooth
- Only some points contribute (the support vectors). So the solution is naturally sparse
- There's no probability estimate. (Though we have `SVC(probability=True)`?)
- Use LinearSVC if we want a linear SVM instead of `SVC(kernel="linear")`
- Prefer `dual=False` when n_samples > n_features.

## Lars / LassoLars

### Model
It is Lasso model fit with Least Angle Regression a.k.a. Lars. More details [here](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html)

### Note
Use when n_features >> n_samples

# References and Copyright Notice

The notes are largely inspired by:

- [Course material](https://amueller.github.io/applied_ml_spring_2017/) for COMS 4995 Applied Machine Learning
- *Introduction to machine learning with python* by Mueller and Guido
- *Applied predictive modeling* by Kuhn, Johnson

Care has been taken to avoid copyrighted contents as much as possible (images, code snippets, etc), and give citation wherever is proper.
