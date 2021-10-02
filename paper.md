---
title: 'ADaPT-ML: A Data Programming Template for Machine Learning'
tags:
  - Python
authors:
  - name: Andrea M. Whittaker^[first author] # note this makes a footnote saying 'first author'
    orcid: 0000-0001-8512-1810
    affiliation: 1
affiliations:
 - name: University of Alberta
   index: 1
date: 24 June 2021
bibliography: paper.bib
---

# Summary

Multiclass/multilabel classification is a task that involves making a prediction about which class(es) a data point 
belongs to; this data point can be text, an image, audio, or can even be multimodal. This task can become intractable for many reasons, including:
1. Insufficient training data to create a data-driven model; available training data may not be appropriate for the domain being studied, it may not be of the right type (e.g. only text but you want text and images), it may not have all of the categories you need, etc.
2. Lack of available annotators with domain expertise, and/or time and money to label large amounts of data.
3. Studying a phenomenon that changes rapidly, so what constitutes a class may change over time, making the available training data obsolete. 

ADaPT-ML is an MLOps system that covers the data processing, data labelling, model design, model training and optimization, and endpoint deployment, with the particular ability to adapt to classification tasks that have the aforementioned challenges. ADaPT-ML is designed to accomplish this by:
1. Using `Snorkel` [@Snorkel] as the data programming framework to create large, annotated, multimodal datasets that can easily adapt to changing classification needs for training data-driven models.
2. Integrating `Label Studio` for annotating multimodal data. Entire training and testing datasets can be annotated using this tool, or just a subset of the training data can be annotated to validate the Label Model and to account for class balance during Label Model training.
3. Orchestrating the Labeling Function / Label Model / End Model development, testing, and monitoring using `MLflow` [@MLflow]. It is easy to add new machine learning algorithms 
4. Deploying all End Models using `FastAPI` [@FastAPI]


# Statement of Need

`ADaPT-ML` is a response to the need for an MLOps system that incorporates data programming for multimodal data, and that can quickly incorporate new classification tasks into one system that works mostly without needing knowledge of whether the task is multiclass or multilabel, and with no hard-coded feature engineering. I created this software especially for any researcher with: some programming experience or interest in learning how to write code based off of examples; access to large amounts of unlabeled data that is constantly changing, such as social media data; and domain expertise or an intuition about how they would follow rules, heuristics, or use knowledge bases to annotate the unlabeled data. I have aimed to take as much of the development work as possible out of creating novel models of phenomenon that have well-developed theories and have yet to be applied to big data.

# Related Work
This software was developed in tandem with the software architecture described in [@Gutierrez2021], to be presented at CASCON x EVOKE 2021. During the development of this software, `Snorkel` progressed into `Snorkel Flow`, a proprietary MLOps system that incorporates data cleaning, model training and deployment, and model evaluation and monitoring into its existing data programming framework.
A description of how this software compares to other commonly-used packages in this research area.
Mentions (if applicable) of any ongoing research projects using the software or recent scholarly publications enabled by it.
A list of key references including a link to the software archive.

# Acknowledgements
We acknowledge contributions from Mitacs and The Canadian Energy and Climate Nexus / Le Lien Canadien de L’Energie et du Climat for funding.

# References


#Paper:

Summary: Has a clear description of the high-level functionality and purpose of the software for a diverse, non-specialist audience been provided?
A statement of need: Does the paper have a section titled ‘Statement of Need’ that clearly states what problems the software is designed to solve and who the target audience is?
State of the field: Do the authors describe how this software compares to other commonly-used packages?
Quality of writing: Is the paper well written (i.e., it does not require editing for structure, language, or writing quality)?
References: Is the list of references complete, and is everything cited appropriately that should be cited (e.g., papers, datasets, software)? Do references in the text use the proper citation syntax?