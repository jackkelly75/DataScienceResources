### This sections describes some good resources I have used to learn about transformers and thier applications to tabular data


#### Begginings
The Transformer was first introduced in 2017 by Vaswani et al in thier paper Attention Is All You Need. (link to this once annotated). (this paper has an intro to scaled dot product attention


Good blog post on how they work - https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853#:~:text=In%20the%20Transformer%2C%20the%20Attention,independently%20through%20a%20separate%20Head.

Line by line in python on how to manually make one  -- https://blog.varunajayasiri.com/ml/transformer.html

Another good blog post - https://towardsdatascience.com/transformers-for-tabular-data-part-3-piecewise-linear-periodic-encodings-1fc49c4bd7bc


BERT paper describes an encoder-style large language model and maked language modeling for prediction tasks (BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding)





In 2020 a transformer for classification of tabular data was proposed in TabTransformer: Tabular Data Modeling Using Contextual Embeddings (make notes on this paper)

Embedding numeric data - https://arxiv.org/abs/2203.05556



The paper Revisiting Deep Learning Models for Tabular Data goes on to describe Feature Tokeniser TabTransformers (FT-TabTransformer) and benchmarks it against other tabular deep learning approaches and importantly against XGBoost and CatBoost (Gradient-boosted decision trees algorithms that are considered gold-standard in tabular data analysis). The also have supplied thier work in thier github repo [https://github.com/Yura52/rtdl]() - link once annotated.





There are github repos that contain info on applying tabular transformers ([https://github.com/aruberts/TabTransformerTF]()). 

The wiki is surprisingly good at explaining them [https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)]()


Extension of Tabular transformers - good selection of parameters (what works best as default and how to most efficiently tune them)
-alteration of the MLP that feeds out data - currently is on end to provide the information but could be improved
-ease of use - package/open source code that allows for ensembling and tuning of model efficiently
