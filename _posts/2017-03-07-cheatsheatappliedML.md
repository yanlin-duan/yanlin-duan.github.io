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

## The Machine Learning Workflow [^2]
![ML Workflow](https://www.mapr.com/ebooks/spark/images/mllib_rec_engine_image006.png)

[^2]: Source: https://www.mapr.com/ebooks/spark/08-recommendation-engine-spark.html

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

- Other important ones (used in Homework 1):
```bash
git rebase --onto feature master~3 master # rebase everything from master~2 (master - 3 commits, excluding this one) up until the tip of master (included) to the tip of feature.
git reflog show # show reference logs that records when the tips of branches and other references were updated in the local repository.
git checkout HEAD@{3} # checkout to the commit where HEAD used to be three moves ago
git checkout feature this_file # merge the specific file (this_file) from feature to your current branch
git log # show git log
```

- The hardest part of git in my opinion is the **"polymorphism"** of git commands. As shown above, you can do git checkout on a branch, a commit, a commit + a file, and they all mean different things. (This motivates me to write a git tutorial in the future when I have time, where I will go through the common git commands in a different way as existing tutorials.)

- Difference (Relationship) between git and github: people new to git may be confused by those two. In one sentence: Git is a version control tool, and GitHub is an online project hosting platform using git.(Therefore, you may use git with or without Github.)

- Git add and staging area[^3]:

![Staging Area] (https://git-scm.com/book/en/v2/images/areas.png)

[^3]: Source: https://git-scm.com/book/en/v2/Getting-Started-Git-Basics

- Fast-forward [^4] (Note that no new command is created):

![What is fast-forward](https://ariya.io/images/2013/09/merging.png)

[^4]: Source: https://ariya.io/2013/09/fast-forward-git-merge

- What is HEAD^ and HEAD~ [^5]:

![HEAD^ and HEAD~ in git]({{ site.url }}/assets/pics/git_1.png)

[^5]: Source: http://schacon.github.io/git/git-rev-parse#_specifying_revisions

# References and Copyright Notice

The notes are largely inspired by:

- [Course material](https://amueller.github.io/applied_ml_spring_2017/) for COMS 4995 Applied Machine Learning
- *Introduction to machine learning with python* by Mueller and Guido
- *Applied predictive modeling* by Kuhn, Johnson

Care has been taken to avoid copyrighted contents as much as possible (images, code snippets, etc), and give citation wherever is proper.
