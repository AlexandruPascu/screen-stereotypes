# 2022: BERT stereotype classifier

The project's starting point (tag `v2022`): a notebook that fine-tunes BERT to label IMDb episodes as stereotype or
non-stereotype titles, using plots of titles found through IMDb stereotype keywords as positives. Built by Alexandru
Pascu and Stefano Li Pira in 2022, following Stefano's initiative.

Looking back in 2026, the labels mixed two things: keyword-found titles were mostly films with long synopses, episode
negatives had short plots, so the classifier partly learned text length and format. The 2026 follow-up at the top of
this repository measures stereotypes in ways that can be checked against the data instead.

Original description: this notebook contains complete code to fine-tune the transformers language model BERT to
perform categorization of movies and series into a label, especially the probability of them being a stereotype
title. We will use BERT to categorize titles from IMDb datasets in stereotypes and non-stereotypes (the categories can
be easily changed for other use cases).
