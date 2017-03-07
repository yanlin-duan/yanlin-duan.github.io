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

## The Machine Learning Workflow
![alt text][ML Workflow]

[ML Workflow]: https://www.mapr.com/ebooks/spark/images/mllib_rec_engine_image006.png "The Machine Learning Workflow"

Source: https://www.mapr.com/ebooks/spark/08-recommendation-engine-spark.html



# References and Copyright Notice

The notes are largely inspired by:

- [Course material](https://amueller.github.io/applied_ml_spring_2017/) for COMS 4995 Applied Machine Learning
- *Introduction to machine learning with python* by Mueller and Guido
- *Applied predictive modeling* by Kuhn, Johnson

Care has been taken to avoid copyrighted contents as much as possible (images, code snippets, etc), and give citation wherever is proper.
