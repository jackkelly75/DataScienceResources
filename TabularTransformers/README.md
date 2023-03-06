### This sections describes some good resources I have used to learn about transformers and thier applications to tabular data




#### Beginnings
Transformers have become increasingly popular in recent years for their ability to classify large amounts of data with high accuracy. The Transformer was first introduced in 2017 by Vaswani et al in thier paper Attention Is All You Need. (link to this once annotated). Originally developed for natural language processing (NLP) and entering the public eye through ChatGPT from OpenAI, transformers are now being used in a variety of applications, including computer vision and speech recognition.

I will discuss Transformers in their application to language models then expand on this to tabular data.

#### Basics

At a high level, transformers are a type of neural network architecture that uses attention mechanisms to process input data. Unlike traditional neural networks that process input sequentially, transformers can process all inputs in parallel, making them faster and more efficient.

Transformers are typically structured into two parts: an encoder and a decoder. The encoder takes the input data and transforms it into a series of vectors, while the decoder takes those vectors and uses them to generate the output. The basic structure is shown below


-structure of transformer here - https://miro.medium.com/v2/resize:fit:640/format:webp/1*XDtQ3C7XrtVuqTtrxr2UWQ.png
-add in the reference to this guy  - https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452

The Encoder and Decoder have consistent architectures however have thier own set of weights. 
-https://miro.medium.com/v2/resize:fit:640/format:webp/1*F7JlVjpmv-XAEeE9IPyzHA.png

The encoder can have many different architectures (two of which (PreNorm and PostNorm) will be discussed later in applications to tabular data) but fundamentally it features residual Skip connections around the layers and LayerNorm layers.

-https://miro.medium.com/v2/resize:fit:490/format:webp/1*THykpgtL058A9EpkstnUJQ.png

#### Attention

The Self-attention layers are what defines the utlity of the Transformer. Self-attention allows the model to focus on different parts of the input data at different times, which is particularly useful for tasks that involve long input sequences. Theattention mechanism allows the model to focus on the most important parts of the input data, calculating a weight for each input vector based on its relevance to the current output. This weight is then used to calculate a weighted sum of the input vectors, which is passed through a series of linear layers to produce the final output.




One of the key advantages of transformers is their ability to handle variable-length input sequences. This is accomplished by using positional encoding, which encodes the position of each input vector in the sequence as a separate feature. This allows the model to distinguish between inputs based on their position in the sequence, which is critical for many classification tasks.



Overall, transformers have proven to be a powerful tool for data classification, particularly in tasks that involve large amounts of data or complex input sequences. As research continues to advance, it is likely that we will see even more innovative uses of transformers in the future.

















https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452



https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34




https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853



https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3







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
