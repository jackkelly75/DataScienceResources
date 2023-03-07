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

To understand how attention works in more detail, let's first consider a traditional neural network architecture. In a traditional neural network, each input is processed sequentially, with the output of each layer being passed on to the next. This means that each output is generated based only on the input at the current time step, without any consideration of the other inputs.

Attention mechanisms, on the other hand, allow the model to consider all inputs simultaneously and selectively focus on the most relevant parts of the input when generating an output. This is accomplished by assigning a weight to each input based on its relevance to the current output.

The weight for each input is calculated using a function that takes into account the current output and the input itself. This function is typically a dot product between a query vector derived from the current output and a key vector derived from the input. The resulting weight is then used to calculate a weighted sum of the input vectors, which is passed through a series of linear layers to produce the final output.

The attention mechanism can be thought of as a way for the model to learn which inputs are most relevant for a given output. By selectively focusing on the most important parts of the input, the model can generate more accurate outputs with less computational overhead.

One of the key benefits of attention mechanisms is their ability to handle variable-length input sequences. By assigning weights to each input based on its relevance to the current output, the model can effectively process sequences of different lengths without the need for padding or truncation.

In summary, attention mechanisms are a critical component of transformer architectures and allow the model to selectively focus on the most relevant parts of the input data when generating an output. This results in more accurate and efficient classification of data, particularly in tasks that involve large amounts of data or complex input sequences.


https://miro.medium.com/v2/resize:fit:720/format:webp/1*cfNpm7aDO4lD3e-Wkwgc1g.png
https://lilianweng.github.io/posts/2018-06-24-attention/transformer.png (https://lilianweng.github.io/posts/2018-06-24-attention/)

Attention gets calculated using 3 learned matrices — Q, K and V which stand for Query, Key and Value. At first, we multiply Q and K to get the attention matrix. This matrix gets scaled and passed through the softmax layer. Afterwards, we multiply it by the V matrix to get out final values. For more intuitive understanding consider the image below which shows how we get from Input Embeddings to Contextual Embeddings using matrices Q, K and V.

https://miro.medium.com/v2/resize:fit:720/format:webp/1*gjJBI_ERoncASFDQjq-Pmg.png  (from https://towardsdatascience.com/transformers-for-tabular-data-tabtransformer-deep-dive-5fb2438da820)

By repeating this procedure h times (with different Q, K, V matrices) we get multiple contextual embeddings which form our final Multi-Headed Attention.




#### Training and applying Transformers for NLP

The process of training and then applying a Tranformer to language is explained well in https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452 by ... 
I will preserve for posterity the best parts of the article below.


Training data consists of two parts:
* The source or input sequence (eg. 'You are welcome' in English, for a translation problem)
* The destination or target sequence (eg. 'De nada' in Spanish)
The Transformer’s goal is to learn how to output the target sequence, by using both the input and target sequence.

- include image - https://miro.medium.com/v2/resize:fit:720/format:webp/1*0g4qdq7Rt6QvDalFFAkL5g.png


The Transformer processes the data like this:

1. The input sequence is converted into Embeddings (with Position Encoding) and fed to the Encoder.
2. The stack of Encoders processes this and produces an encoded representation of the input sequence.
3. The target sequence is prepended with a start-of-sentence token, converted into Embeddings (with Position Encoding), and fed to the Decoder.
4. The stack of Decoders processes this along with the Encoder stack’s encoded representation to produce an encoded representation of the target sequence.
5. The Output layer converts it into word probabilities and the final output sequence.
6. The Transformer’s Loss function compares this output sequence with the target sequence from the training data. This loss is used to generate gradients to train the Transformer during back-propagation.


During Inference, we have only the input sequence and don’t have the target sequence to pass as input to the Decoder. The goal of the Transformer is to produce the target sequence from the input sequence alone.

So, like in a Seq2Seq model, we generate the output in a loop and feed the output sequence from the previous timestep to the Decoder in the next timestep until we come across an end-of-sentence token.

The difference from the Seq2Seq model is that, at each timestep, we re-feed the entire output sequence generated thus far, rather than just the last word.

-include image - https://miro.medium.com/v2/resize:fit:720/format:webp/1*-uvybwr8xULd3ug9ZwcSaQ.png

The flow of data during Inference is:
1. The input sequence is converted into Embeddings (with Position Encoding) and fed to the Encoder.
2. The stack of Encoders processes this and produces an encoded representation of the input sequence.
3. Instead of the target sequence, we use an empty sequence with only a start-of-sentence token. This is converted into Embeddings (with Position Encoding) and fed to the Decoder.
4. The stack of Decoders processes this along with the Encoder stack’s encoded representation to produce an encoded representation of the target sequence.
5. The Output layer converts it into word probabilities and produces an output sequence.
6. We take the last word of the output sequence as the predicted word. That word is now filled into the second position of our Decoder input sequence, which now contains a start-of-sentence token and the first word.
7. Go back to step #3. As before, feed the new Decoder sequence into the model. Then take the second word of the output and append it to the Decoder sequence. Repeat this until it predicts an end-of-sentence token. Note that since the Encoder sequence does not change for each iteration, we do not have to repeat steps #1 and #2 each time.

Again,  the above is from https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452 by ....



### Applications to Tabular data

Within the context of the above for NLP tasks, I will now shift to applications to Tabular data. First proposed in TabTransformer: Tabular Data Modeling Using Contextual Embeddings (link here) by Huang _et al_ 2020. They proposed that Transformers can be used to transform categorical embeddings into contextual ones before MLP is used to generate the output from the contextual categorical features and the standard numerical inputs.  

https://miro.medium.com/v2/resize:fit:720/format:webp/1*M5xJfs2_fDjsW-hsVWqE-g.png
-reference this https://towardsdatascience.com/transformers-for-tabular-data-tabtransformer-deep-dive-5fb2438da820

The TabTransformer architecture proposed was as follows 

-https://miro.medium.com/v2/resize:fit:720/format:webp/1*ZfFJ4gfa4p5PpHmGNClj5A.png


#### FT-Transformers
As you can see, this gives little thought to the numerical features, which are treated as usual within the MLP. Gorishniy _et al_ proposed the use of a Feature Tokeniser in Revisiting Deep Learning Models for Tabular Data to expand on TabTransformer to a FT-Transformer, which uses numerical embeddings and [CLS] token for output.

The main advantage of numerical embeddings is the ability to pass numerical and categorical features through the same Transformers and share context (in fact, it is being shown that embedding numerical features can lead to performance increase in DL applications in general - this is an interesting area that will see huge advancements in the next few years (discussed in papers such as On Embeddings for Numerical Features in Tabular Deep Learning and [this blog post and associated paper from Yandex](https://research.yandex.com/blog/embeddings-for-numerical-features-in-tabular-deep-learning)). The FT-Transformer uses linear embeddings for numeric features mening each feature is transformed into a adense vector after being passed through a fully connected layer. Other more advanced approaches to numeric embeddings are discussed below. 

The other change in FT-Transformers is the use of [CLS] tokens. Once all features are embedded, they are appended to a [CLS] token. Once they pass through a Transformer block, these contextualised [CLS] tokens are used as the input to the MLP classifier to get the output.

-https://miro.medium.com/v2/resize:fit:720/format:webp/1*cRWwJ9NgmMnLJU3ncihuMA.png
-from https://towardsdatascience.com/improving-tabtransformer-part-1-linear-numerical-embeddings-dbc3be3b5bb5





#### Advances in Embeddings

https://research.yandex.com/blog/embeddings-for-numerical-features-in-tabular-deep-learning





#### Feature importances

 feature importances can be 
 
 
 One of the biggest advantages of FT-Transformer is the in-built explainability. Since all the features are passed through a Transformer, we can get their attention maps and infer feature importances. These importances are calculated using the following formula
https://miro.medium.com/v2/resize:fit:720/format:webp/1*52d2BCS55BWd_h5c4sOFBw.png

Feature importances formula. Source: Gorishniy et al. (2021)
where p_ihl is the h-th head’s attention map for the [CLS] token from the forward pass of the l-th layer on the i-th sample. The formula basically sums up all the attention scores for [CLS] token across different attention-heads (heads parameter) and Transformer layers (depth parameter) and then divides them by heads x depth. Local importances (p_i) can be averaged across all rows to get the global importances (p).




### Limitations
'FT-Transformer requires more resources (both hardware and time) for training than
simple models such as ResNet and may not be easily scaled to datasets when the number of features
is “too large” (it is determined by the available hardware and time budget). Consequently, widespread
usage of FT-Transformer for solving tabular data problems can lead to greater CO2 emissions
produced by ML pipelines, since tabular data problems are ubiquitous. The main cause of the
described problem lies in the quadratic complexity of the vanilla MHSA with respect to the number
of features. However, the issue can be alleviated by using efficient approximations of MHSA (Tay
et al., 2020). Additionally, it is still possible to distill FT-Transformer into simpler architectures for
better inference performance. We report training times and the used hardware in supplementary.'








https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452



https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34




https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853



https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3



good blog post on attention - https://lilianweng.github.io/posts/2018-06-24-attention/



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
