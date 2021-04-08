---
layout: post
title: "Delving into BERT Representations to Explain Sentence Probing Results"
categories: junk
author:
- Hosein Mohebbi
- Ali Modarressi

permalink: /blog/explain-probing-results
---

<img align="right" src="/resources/posts/tsne.png" width="375" height="180" >

This is a post for the paper [Exploring the Role of BERT Token Representations to Explain Sentence Probing Results](https://arxiv.org/pdf/2104.01477.pdf).

we carry out an extensive gradient-based attribution analysis to explain probing performance results from the viewpoint of token representations. Based on a set of probing tasks we show that:
* while most of the positional information is diminished through layers of BERT, sentence-ending tokens are partially responsible for carrying this knowledge to higher layers in the model.
* BERT tends to encode verb tense and noun number information in the $$\texttt{##s}$$ token and that it can clearly distinguish
the two usages of the token by separating them into distinct subspaces in the higher layers. 
* abnormalities can be captured by specific token representations, e.g., in two consecutive swapped tokens or a coordinator between two swapped clauses.

---

## What's Wrong with Standard Probing?
Probing is one of the popular analysis methods, often used for investigating the encoded knowledge in language models. This is typically carried out by training a set of diagnostic classifiers that predict a specific linguistic property based on the representa-
tions obtained from different layers. 

Recent works in probing language models demonstrate that initial layers are responsible for encoding low-level linguistic information, such as part of speech and positional information, whereas intermediate layers are better at syntactic phenomena, such as syntactic tree depth or subject-verb agreement, while in general semantic information is spread across the entire model. 
Despite elucidating the type of knowledge encoded in various layers, these studies do not go further to investigate the reasons behind the layer-wise behavior and the role played by token representations. Analyzing the shortcomings of pre-trained language models requires a scrutiny beyond the mere performance in a given probing task.

In this paper, we extend the layer-wise analysis to the token level in search for distinct and meaningful subspaces in BERT’s representation space that can explain the performance trends in various probing tasks.

## Gradient-based Attribution Method
> Saliency Meets Probes

We leveraged a gradient-based attribution method in order to enable an in-depth analysis of layer-wise representations with the objective of explaining probing performances. Specifically, we are interested in computing the attribution of each input token to the output labels. This is usually referred to as the saliency score of an input token to classifier’s decision. 
Note that using attention weights for this purpose can be misleading given that raw attention weights do not necessarily correspond to the importance of individual token representations.


## Probing Explanation
In what follows in this section, we use the attribution method to find those tokens that play the central role in different surface, syntactic and semantic probing tasks. Based on these tokens we then investigate the reasons behind performance variations across layers.

### Sentence Length
In this surface-level task we probe the representation of a given sentence in order to estimate its size, i.e., the number of words (not tokens) in it. To this end, we used [SentEval](https://github.com/facebookresearch/SentEval/tree/master/data/probing)’s SentLen dataset, but changed the formulation from the original classification objective to a regression one which allows a better generalization due to its fine-grained setting. 
The diagnostic classifier receives average-pooled representation of a sentence as input and outputs a continuous number as an estimate for the input length.

Rounding the regressed estimates and comparing them with the gold labels in the test set, we can observe a significant performance drop from 0.91 accuracy in the first layer to 0.44 in the last layer.
Given that the ability to encode the exact length of input sentences is not necessarily a critical feature, and this decay is not surprising due to the existing previous studies, we do not focus on layer-wise performance results and instead discuss the reason behind the performance variations across layers. To this end, we calculated the absolute saliency scores for each input token in order to find those tokens that played pivotal role while estimating sentence length.

<br>
> Sentence ending tokens retain positional information.

<img align="right" src="/resources/posts/SentLen5LayerAttrib_1.png" width="376">
<br>
This figure shows tokens that most contributed to the probing results across different layers according to the attribution analysis. Clearly, finalizing tokens (e.g. “[SEP]” and “.”) are the main contributors in the higher layers. 

<br><br>
<img align="right" src="/resources/posts/PosEmbedding4TSNE_wCbar.png" width="380" >
Here are t-SNE plots of the representations of four se-
lected high frequency tokens (“[SEP]”, “.” full stop,
_“the”_, _“and”_) in different sentences. 
Colors indicate the corresponding token’s position in the sentence (darker colors means higher position index). Finalizing tokens (e.g., “[SEP]”, “.”) preserve distinct patterns in final layers, indicating their role in encoding positional information, while other (high frequency) tokens exhibit no such behavior.
Clearly, positioning information is lost throughout layers in BERT; however, finalizing tokens partially retain this information, as visible from distinct pattern in higher layers.

<br>
### Verb Tense and Noun Number
This analysis inspects BERT representations for grammatical number and tense information. For this experiment we used the Tense and ObjNum tasks: the former checks whether the main-clause verb is labeled as present or past, whereas the latter classifies the object according to its number, i.e., singular or plural. On both tasks, BERT preserves a consistently high performance (> 0.82 accuracy) across all layers.

> Articles and ending tokens (e.g., $$\texttt{##s}$$ and $$\texttt{##ed}$$ ) are key playmakers.

<img align="right" src="/resources/posts/Number_Tense_5LayerAttrib_4.png" width="362" height="330">
Attribution analysis, illustrated here, reveals that article words (e.g., _“a”_ and _“an”_) and the ending $$\texttt{##s}$$ token, which makes out-of-vocab plural words (or third person present verbs), are among the most attributed tokens in the ObjNum task. This shows that these tokens are mainly responsible for encoding object’s number information across layers. 

As for the Tense task, this Figure shows a consistently high influence from verb ending tokens (e.g., $$\texttt{##ed}$$  and $$\texttt{##s}$$) across layers which is in line with performance trends for this task and highlights the role of these tokens in preserving verb tense information.

<br>

> $$\texttt{##s}$$ — Plural or Present?

The $$\texttt{##s}$$ token proved influential in both __tense__ and __number__ tasks.
The token can make a verb into its simple present tense (e.g., _read_ → _reads_) or transform a singular noun into its plural form (e.g., _book_ → _books_). We further investigated the representation space to check if BERT can distinguish this nuance. Results are shown here: 

<img align="center" src="/resources/posts/4Layer_S.png">

Colors indicate whether the token occurred in present- or past-labeled sentence in the Tense task. For the sake of comparison, we also include two present verbs without the $$\texttt{##s}$$ token (i.e., does and works) and two irregular plural nouns (i.e., _men_ and _children_), in rounded boxes. 
After the initial layers, BERT recognizes and separates these two forms into two distinct clusters (while BERT’s tokenizer made no distinction among different usages). The distinction between the two different usages of the token (as well as the tense information) is clearly encoded in higher layer contextualized representations.



### Inversion Abnormalities
For this set of experiments, we opted for [SentEval](https://github.com/facebookresearch/SentEval/tree/master/data/probing)’s Bi-gram Shift and Coordination Inversion tasks which respectively probe model’s ability in detecting syntactic and semantic abnormalities. The goal of this analysis was to to investigate if BERT encodes inversion abnormality in a given sentence into specific token representations.


#### Token-level inversion
Bi-gram Shift (__BShift__) checks the ability of a model to identify whether two adjacent words within a given sentence have been inverted. Probing results shows that the higher half layers of BERT can properly distinguish this peculiarity. Similarly to the previous experiments, we leveraged the gradient attribution method to figure out those tokens that were most effective in detecting the inverted sentences. Given that the dataset does not specify the inverted tokens, we reconstructed the inverted examples by randomly swapping two consecutive tokens in the original sentences of the test set, excluding the be ginning of the sentences and punctuation marks.

<img align="right" src="/resources/posts/bshift_heatmap.png" width="360">
Our attribution analysis shows that swapping two consecutive words in a sentence results in a significant boost in the attribution scores of the inverted tokens. As an example, the subsequent figure depicts attribution scores of each token in a randomly sampled sentence from the test set across different layers. The classifier distinctively focuses on the token representations for the shifted words, while no such patterns exists for the original sentence.

To verify if this observation holds true for other instances in the test set, we carried out the following experiment.
For each given sequence $$X$$ of $$n$$ tokens, we defined a boolean mask $$M =[m_1 , m_2 , ..., m_n]$$ which denotes the position of the
inversion according to the following condition:
$$\begin{equation}
m_i = 
  \begin{cases}
    1, & x_i \in V \\
    0, & \textrm{otherwise} \
  \end{cases}
\end{equation}$$
where $$V$$ is the set of all tokens in the shifted bi-gram ($$|V|\ge2$$, given BERT's sub-word tokenization).
<img align="right" src="/resources/posts/bshift_corr.png" width="255">

Then we computed the Spearman's rank correlation coefficient of the attribution scores with $$M$$ for all examples in the test set.
We observe that in altered sentences the correlation significantly grows over the first few layers which indicates model's increased sensitivity to the shifted tokens.

<img align="right" src="/resources/posts/bshift_sim.png" width="365">
We hypothesize that BERT implicitly encodes abnormalities in the representation of shifted tokens. To investigate this, we computed the cosine distance of each token to itself in the original and shifted sentences. This figure shows layer-wise statistics for both shifted and non-shifted tokens. Distances between the shifted token representations aligns well with the performance trend for this probing task (also shown in the figure).

To investigate the root cause of this, we took a step further and analyzed the building blocks of these representations, i.e., the self-attention mechanism (read [the paper](https://arxiv.org/pdf/2104.01477.pdf) for details).

#### Phrasal-level inversion
The Coordination Inversion (__CoordInv__) task is a binary classification that contains sentences with two coordinated clausal conjoints (and only one coordinating conjunction). In half of the sentences the clauses’ order is inverted and the goal is to detect malformed sentences at phrasal level.

BERT’s performance on this task increases through layers and then slightly decreases in the last three layers. We observed that the attribution scores for _“but”_ and _“and”_ coordinators to be among the highest and that these scores notably increase through layers. We hypothesize that BERT might implicitly encodes phrasal level abnormalities in specific token representations.

<img align="center" src="/resources/posts/coord_bar_full.png">

<br>
> Odd Coordinator Representation

To verify our hypothesis, we filtered the test set to ensure all sentences contain either a _“but”_ or an _“and”_ coordinator, since no sentence appears with both labels in the dataset. Then, we reconstructed the original examples by inverting the order of the two clauses in the inverted instances.

<img align="right" src="/resources/posts/but_and_sim_saliency_1.png" width="380">
Feeding this to BERT, we extracted token representations and computed the cosine distance between the representations of each token in the original and inverted sentences. The subsequent figure shows these distances, as well as the normalized saliency score for coordinators (averaged on all examples in each layer), and layer-wise performance for the CoordInv probing task. 

Surprisingly, all these curves exhibit a similar trend. As we can see, when the order of the clauses are inverted, the representations of the coordinators _“but”_ or _“and”_ play a pivotal role in making sentence representations distinct from one another while there is nearly no change in the representation of other words. This observation implies that BERT somehow encodes oddity in the coordinator representations (corroborating part of the findings of our previous analysis of BShift task in the previous part).

## Conclusion

We provided an analysis on the representation space of BERT in search for distinct and meaningful subspaces that can explain probing results. Based on a set of probing tasks and with the help of attribution methods we showed that BERT tends to encode meaningful knowledge in specific token representations (which are often ignored in standard classification setups), allowing the model to detect syntactic and semantic abnormalities, and to distinctively separate grammatical number and tense subspaces.

Our approach in using a simple diagnostic classifier and incorporating attribution methods provides a novel way of extracting qualitative results based on multi-class classification probes. This analysis method could be easily applied to probing various deep pre-trained models on various sentence level tasks. We hope this method will spur future probing studies in other evaluation scenarios. Future work might explore to investigate how these subspaces are evolved or transformed during fine-tuning and whether being beneficial at inference time to various downstream tasks or to check whether these behaviors are affected by different training objectives or tokenization strategies.