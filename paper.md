---
title: 'ADaPT-ML: A Data Programming Template for Machine Learning'
tags:
  - Python
  - Docker
  - MLOps
  - data programming
  - machine learning
  - model lifecycle
authors:
  - name: Andrea M. Whittaker^[first author] # note this makes a footnote saying 'first author'
    orcid: 0000-0001-8512-1810
    affiliation: 1
affiliations:
  - name: University of Alberta
    index: 1
date: 21 October 2021
bibliography: paper.bib
---

# Summary

Classification is a task that involves making a prediction about which class(es) a data point 
belongs to; this data point can be text, an image, audio, or can even be multimodal. This task can become intractable for many reasons, including:

* Insufficient training data to create a data-driven model; available training data may not be appropriate for the domain being studied, it may not be of the right type (e.g. only text but you want text and images), it may not have all of the categories you need, etc.

* Lack of available annotators with domain expertise, and/or resources such as time and money to label large amounts of data.

* Studying a phenomenon that changes rapidly, so what constitutes a class may change over time, making the available training data obsolete. 

ADaPT-ML is a multimodal-ready MLOps system that covers the data processing, data labelling, model design, model training and optimization, and endpoint deployment, with the particular ability to adapt to classification tasks that have the aforementioned challenges. ADaPT-ML is designed to accomplish this by:

* Using `Snorkel` [@Snorkel] as the data programming framework to create large, annotated, multimodal datasets that can easily adapt to changing classification needs for training data-driven models.

* Integrating `Label Studio` [@LabelStudio] for annotating multimodal data.

* Orchestrating the Labeling Function / Label Model / End Model development, testing, and monitoring using `MLflow` [@MLflow].

* Deploying all End Models using `FastAPI` [@FastAPI]


# Statement of Need

Often when studying natural phenomena by creating data-driven models, processing the data becomes the largest challenge. Without a framework to build upon and implement one's ideas, researchers are forced to hastily build inflexible programs from the ground up. When hypotheses need to be reworked or modelling a new aspect of the phenomena becomes necessary, even more time is spent on the program before finally being able to test out new ideas. This inherently causes problems, with additional problems arising such as including internal and external validation steps as an afterthought rather than a checkstop in the pipeline.

ADaPT-ML aims to be the flexible framework upon which researchers can implement their understanding of the phenomena under study. This software was created especially for any researcher with:

* Some programming experience or interest in learning how to write code based off of examples. 

* Access to large amounts of unlabeled data that is constantly changing, such as social media data. 

* Domain expertise or an intuition about how they would follow rules, heuristics, or use knowledge bases to annotate the unlabeled data. 

ADaPT-ML takes as much of the development work as possible out of creating novel models of phenomenon for which we have well-developed theories that have yet to be applied to big data.

# Related Work
An early version of this software supported the modelling of universal personal values and was complementary to the software architecture described in @Gutierrez2021. During the development of this software, `Snorkel` progressed into `Snorkel Flow` [@SnorkelFlow], a proprietary MLOps system that incorporates data cleaning, model training and deployment, and model evaluation and monitoring into its existing data programming framework.

# Acknowledgements
We acknowledge contributions from Mitacs and The Canadian Energy and Climate Nexus / Le Lien Canadien de L’Energie et du Climat for funding during the early stages of this project's development.

# References
