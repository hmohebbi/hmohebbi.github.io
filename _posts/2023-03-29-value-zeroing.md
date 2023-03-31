---
layout: post
title: "Why Value Zeroing?"
categories: junk
author:
- Hosein Mohebbi

permalink: /blog/value-zeroing

---

<i>This post serves as a side note on __Value Zeroing__, an interpretability method for quantifying context mixing in Transformers. 
It is based on our recent research [paper](https://arxiv.org/abs/2301.12971) in which we show that the token importance scores obtained through Value Zeroing offer better interpretations compared to previous analysis methods in terms of plausibility, faithfulness, and agreement with probing.</i>

<i>For the definition and evaluation of Value Zeroing, please refer to the paper. Here, I aim to share a few thoughts and ignite a discussion on why Value Zeroing is deemed to be a promising analysis method for dissecting Transformers.</i>

<a class="my-button" href="https://arxiv.org/pdf/2301.12971.pdf">📃 Paper</a>
<a class="my-button" href="https://github.com/hmohebbi/ValueZeroing">☕ Code</a>
<a class="my-button" href="https://huggingface.co/spaces/amsterdamNLP/value-zeroing">🤗Demo</a>

---

### Intro: Context Mixing
Transformers [(Vaswani et al., 2017)](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) are excelling in the field of Artificial Intelligence, they are always ahead of the game, dominating every area they venture into, such as [language](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html), [speech](https://arxiv.org/abs/2212.04356), [vision](https://arxiv.org/abs/2010.11929), and [biology](https://www.nature.com/articles/s41586-021-03819-2). Their ability to utilize pairwise interactions between input tokens at each timestep (which we call <i>context mixing</i>), has made them a prime choice of architecture for learning contextualized representations. 

While enjoying the Waltz of Transformers in real-world applications, we should understand their inner dynamics too. To better understand these models, we require more than a one-dimensional colorful heatmap array over input tokens to display their importance for the final model's decision.
We need to be able to quantify the context-mixing interactions between all input pairs, which is something that Transformers are made for. Specifically, we require a matrix that can tell us to what extent each token uses information from other tokens within the sentence to form its contextualized representation at each layer. This is precisely where __Value Zeroing__ comes into play.


### Why Value Zeroing?
>🔨 __Customized for Transformers__ 

In my humble view, it is crucial to develop or customize analysis methods tailored to the modeling architectures of the target models, based on their true mathematical foundation. 
We need the right tool for the job! Merely employing interpretability approaches that worked for earlier generations of deep learning architectures without any adaptation may not lead to reliable findings.
To illustrate, looking at the attention pattern has been a commonly used approach to gain insight into the information mixing process in attention-based models. It makes sense to rely on these weights when analyzing those attention layers that act as a weighted average over the representations obtained in various time steps in recurrent neural models. However, in the case of Transformers, self-attention weights comprise only a small part of the model, while there are also other components in a Transformer layer that can exert a significant impact on information mixing in the output representations.

__Value Zeroing__ is a simple yet effective and intuitive analysis method, which is designed based on the mathematical principles of Transformers, without any underlying assumptions or posing confounding variables (such as involving an extra fine-tuned classifier).

<br>
>🌍 __Considering all the components inside of a Transformer layer__

Previous research has shown that Transformers do not solely depend on the weights assigned by the self-attention mechanism, as altering or removing these weights may lead to the same, and sometimes even better model performance on downstream tasks [(Hassid et al., 2022)](https://aclanthology.org/2022.findings-emnlp.101/). Additionally, by looking at their patterns, it has been observed that these weights tend to concentrate on uninformative tokens within the sentence [(Voita et al., 2018](https://aclanthology.org/P18-1117/); [Clark et al., 2019)](https://aclanthology.org/W19-4828/).

To address this, some post-processing interpretability techniques have been proposed to expand the scope of analysis by incorporating other components in a Transformer encoder layer. An exemplary instance of this is [Kobayashi (2020)](https://aclanthology.org/2020.emnlp-main.574/)'s insightful work, which highlights the importance of considering the impact of value vectors as well. It is possible that the model may assign a high attention weight to a token with a small norm (they showed that this usually happens for highly frequent and less
informative words). This implies a higher attention might not necessarily lead to a higher contribution to the model's decision.

Good news is __Value Zeroing__ is being computed from a Transformer layer output, which means it incorporates all the components inside a Transformer layer by design, such as multi-head attention, residual connections, layer normalization, and also
position-wise feed-forward networks, resulting in a more reliable
context mixing score.

<br>
>❄ __Keeping the information flow intact during analysis__

Unlike generic perturbation approaches, Value Zeroing does not remove a token representation from the input of an encoder or a decoder layer. We argue that ablating an input token representation cannot be a reliable basis for understanding context mixing process since any changes in the input vectors will mathematically lead to changes in the query and key vectors, resulting in a shift in the attention distribution. Consequently, there will be a discrepancy between the alternative attention weights that we analyze and those we initially had for the original context. 

[Madsen (2021)](https://aclanthology.org/2022.findings-emnlp.125/)'s work provides a cautionary example of the dangers of this approach. They show that for the input example, "The movie is great. I really liked it.", although the model mostly attends to the word 'great' to predict it as a positive sentiment, the model's confidence remains the same when we replace the word 'great' with a '[MASK]' token. Therefore, one might consider an analysis method unfaithful if it strongly highlights the word 'great'. However, the fact here is that the model's confidence does not drop due to the redundancy of positive cues in the sentence, causing the model's attention to shift significantly towards the word 'liked' to compensate after removing the word 'great'.

In contrast, __Value Zeroing__ only nullifies the value vector of a specific token representation and leaves the key and query vectors (and thus the pattern of attention flow) intact. In this way, the token representation can also maintain its identity within the Transformer layer, but it does not contribute to forming other token representations.

<br>
>🧐 __Providing interpretation for both layer-specific and input attribution__

Our experiments suggest that __Value Zeroing__ offers better interpretation compared to previous analysis methods, not only for analyzing a specific single Transformer layer, but also for analyzing the entire model when scores are aggregated using the rollout method [(Abnar and Zuidema,
2020)](https://aclanthology.org/2020.acl-main.385/).


### A Qualitative Example
<img align="center" src="/resources/posts/vz.png">

Here's a graph I really like, showing Value Zeroing scores for [RoBERTa](https://arxiv.org/abs/1907.11692) for the sentence <i>"Either you win the game or you \<mask\> the game"</i>, showing a very interesting pattern that was caught by one of my supervisors:
* In the first two layers, the highest values are around the diagonal (mixing information w/ immediate neighbors), and an exchange of information between 'either' and 'or'.
* Then, in layers 4-7, we see a mixing of information between equivalent words in the two sub-sentences ('you-you,' 'win-\<mask\>,' 'the-the,' 'game-game').
* In layers 9-11, there are notable vertical lines, as if all the information is concentrated in the nodes for 'win' and 'or.'
* In the final layer, there are only high values on the diagonal (hardly any mixing).

  
🤗 Try your own examples in our online [demo](https://huggingface.co/spaces/amsterdamNLP/value-zeroing).
<br>

### Your thoughts and comments

<script src="https://utteranc.es/client.js"
        repo="hmohebbi/hmohebbi.github.io"
        issue-term="title"
        theme="github-light"
        crossorigin="anonymous"
        async>
</script>
