---
layout: post
title: "Delving into BERT Representations to Explain Probing Results"
categories: junk
author:
- Hosein Mohebbi
- Ali Modarressi

permalink: /blog/explain-probing-results
---

This is a post for the paper [Exploring the Role of BERT Token Representations to Explain Sentence Probing Results]("https://arxiv.org/pdf/2104.01477.pdf").


we carried out an extensive gradient-based attribution analysis to to explain probing performance results from the viewpoint of token representations, and found that:
* while most of the positional information is diminished through layers of BERT, sentence-ending tokens are partially responsible for carrying this knowledge to higher layers in the model.
* BERT tends to encode verb tense and noun number information in the ##s token and that it can clearly distinguish
the two usages of the token by separating them into distinct subspaces in the higher layers. 
* abnormalities can be captured by specific token representations, e.g., in two consecutive swapped tokens or a coordinator between two swapped clauses.