# DL-Text : pre-processing modules for deep learning (Keras, tensorflow).
This repository consists of modules for pre-processing the textual data. Examples are also given for training deep models (DNN, CNN, RNN, LSTM). There are many additional functionilities which are as follows:
- Preparing data for problems like sentiment analysis, sentence contextual similarity, question answering, machine translation, etc.
- Compute lexical and semantic hand-crafted features like words overlap, n-gram overlap, td-idf, count features, etc.  Most of these features are used in the following papers:
  - [External features for community question answering](http://maroo.cs.umass.edu/getpdf.php?id=1281). 
  - [Voltron: A Hybrid System For Answer Validation Based On Lexical And Distance Features](http://alt.qcri.org/semeval2015/cdrom/pdf/SemEval043.pdf). 
- Implementation of deep models as described in the following papers (for reproducible code refer to [DeepLearn-repo](https://github.com/GauravBh1010tt/DeepLearn)):
  - [WIKIQA: A Challenge Dataset for Open-Domain Question Answering](https://aclweb.org/anthology/D15-1237).
  - [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.6492&rep=rep1&type=pdf).
  - [Neural-based Approaches for Ranking in Community Question Answering](http://aclweb.org/anthology/S/S16/S16-1128.pdf).
- Implementation of evaluation metrics such as MAP, MRR, AP@k, BM25 etc.

## Dependencies
#### The required dependencies are mentioned in requirement.txt. You can install them manually or by using the following command:
```python
$ pip install -r requirements.txt
```

## Prepare the data for NLP problems like sentiment analysis.
#### 1. The data and labels looks like this:
```python
raw_data = ['this,,, is$$ a positive ..sentence','this is a ((*negative ,,@sentence',
        'yet another..'' positive$$ sentence','the last one is ...,negative']
labels = [1,0,1,0]
```
This type of data is commonly used in sentiment analysis type problems. The first step is to clean the data:
```python
from dl_text import dl
data = []
for sent in raw_data:
    data.append(dl.clean(sent))
    
print data
['this is a positive sentence', 'this is a negative sentence', 
'yet another positive sentence', 'the last one is negative']
```
Once the raw data is cleaned, the next step is the prepare that can be passed to the deep models. Use the following function:
```python
data_inp = dl.process_data(sent_l = data, dimx = 10)
```

The `process_data` function preprocesses the data that can be used with deep models. The `process_data` has following parameters:
```python
process_data(sent_l,sent_r,wordVec_model,dimx,dimy,vocab_size,embedding_dim)
```
where,

- `sent_l` : data to be sent to training model (if you are using only one channel, as in the case of sentiment analysis, then use this parameter)
- `sent_r` : data for the second channel (discussed later)
- `wordVec_model` : pre-trained word vector embeddings (either GloVe or Word2vec)
- `dimx` and `dimy` : number of words to be included (if a sentence has lesser words than this value, it will be padded by 0, otherwise extra words will be truncated)
- `vocab_size` : number of unique words to be included in the vocabulary
- `embedding_dim` : size of the embeddings for wordVec_models

#### 2. Using pre-trained word vector embeddings
```python
from dl_text import dl
import gensim

# for 50-dim glove embeddings use:
wordVec_model = dl.loadGloveModel('path_of_the_embeddings/glove.6B.50d.txt')

# for 300 dim word2vec embeddings use: 
wordVec_model = gensim.models.KeyedVectors.load_word2vec_format("path/GoogleNews-vectors-negative300.bin.gz",
                                                                 binary=True)

data_inp, embedding_matrix = dl.process_data(sent_l = data, wordVec_model = wordVec_model, dimx = 10)
```

#### 3. Defining deep models

```python
from dl_text import dl
from keras.layers import Input, Dense, Dropout, Merge, Conv1D, Lambda, Flatten, MaxPooling1D

def model_dnn(dimx, embedding_matrix):
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    embed = dl.word2vec_embedding_layer(embedding_matrix)(inpx)
    flat_embed = Flatten()(embed)
    nnet_h = Dense(units=10,activation='sigmoid')(flat_embed)
    nnet_out = Dense(units=2,activation='sigmoid')(nnet_h)
    model = Model([inpx],nnet_out)
    model.compile(loss='mse',optimizer='adam')
    return model

def model_cnn(dimx, embedding_matrix):
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    embed = dl.word2vec_embedding_layer(embedding_matrix)(inpx)
    sent = Conv1D(nb_filter=3,filter_length=2,activation='relu')(embed)
    pool = MaxPooling1D()(sent)
    flat_embed = Flatten()(pool)
    nnet_h = Dense(units=10,activation='sigmoid')(flat_embed)
    nnet_out = Dense(units=2,activation='sigmoid')(nnet_h)
    model = Model([inpx],nnet_out)
    model.compile(loss='mse',optimizer='adam')
    return model
```
#### 4. Training the models

```python
data = ['this is a positive sentence', 'this is a negative sentence', 'yet another positive sentence', 'the last one is negative']
labels = [1,0,1,0]

data_inp, embedding_matrix = dl.process_data(sent_l = data, wordVec_model = wordVec_model, dimx = 10)

model = model_dnn(dimx = 10, embedding_matrix = embedding_matrix)
model.fit(data_inp, labels)

model = model_cnn(dimx = 10, embedding_matrix = embedding_matrix)
model.fit(data_inp, labels)
```
## Prepare the data for NLP problems like computing sentence similarity, question answering, etc.
#### 1. Creating two channel models
These type of models use two data streams. This can be used to NLP tasks such as question answering, sentence similarity computation, etc. The data looks like this

```python
data_l = ['this is a positive sentence','this is a negative sentence', 
          'yet another positive sentence', 'the last one is negative']
          
data_r = ['positive words are good, better, best, etc.', 'negative words are bad, sad, etc.', 
          'feeling good', 'sooo depressed.']
         
labels = [1,0,1,0]
```
Here, `data_l` and `data_r` can be two sentences for computing sentence similarity, question-answer pairs for question answering problem, etc.
Let's define a model for the these type of tasks

``` python

def model_cnn2(dimx, dimy, embedding_matrix):
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    embedx = dl.word2vec_embedding_layer(embedding_matrix)(inpx)
    inpy = Input(shape=(dimx,),dtype='int32',name='inpy')   
    embedy = dl.word2vec_embedding_layer(embedding_matrix)(inpy)
    
    sent_l = Conv1D(nb_filter=3,filter_length=2,activation='relu')(embedx)
    sent_r = Conv1D(nb_filter=3,filter_length=2,activation='relu')(embedy)
    pool_l = MaxPooling1D()(sent_l)
    pool_r = MaxPooling1D()(sent_r)
    
    combine  = merge(mode='concat')([pool_l, pool_r])
    flat_embed = Flatten()(combine)
    nnet_h = Dense(units=10,activation='sigmoid')(flat_embed)
    nnet_out = Dense(units=2,activation='sigmoid')(nnet_h)
    model = Model([inpx],nnet_out)
    model.compile(loss='mse',optimizer='adam')
    
    return model
```
#### 2. Tarining a two channel deep model

```python

data_inp_l, data_inp_r, embedding_matrix = dl.process_data(sent_l = data_l, sent_r = data_r, 
                                                           wordVec_model = wordVec_model, dimx = 10, dimy = 10)

model = model_cnn2(dimx = 10, dimy = 10, embedding_matrix = embedding_matrix)
model.fit([data_inp_l, data_inp_r], labels)
```
## Hand crafted features - These could be used with problems like sentence similarity, question answering, etc. 
#### 1. Computing lexical and semantic features.
```python
>>> from dl_text import lex_sem_ft

>>> sent1 = 'i like natural language processing'
>>> sent2 = 'i like deep learning'

>>> lex_sem_ft.tokenize(sent1) # tokenizing a sentence
['i', 'like', 'natural', 'language', 'processing']

>>> lex_sem_ft.overlap(sent1,sent2) # number of words common
2
```
Functions currently present in the `lex_sem_ft` are:
- tokenize(sent): tokenize a given string
- length(sent) : Number Of Words In A String (Returns Integer)
- substringCheck(sent1, sent2) : Whether A String Is Subset Of Other (Returns 1 and 0)
- overlap(sent1, sent2): Number Of Same Words In Two Sentences (Returns Float)
- overlapSyn(sent1, sent2): Number Of Synonyms In Two Sentences (Returns Float)
- train_BOW(lst) : Forming Bag Of Words (BOW) (Returns BOW Dictionary)
- Sum_BOW(sent, dic) : Sum Of BOW Values For A Sent (Returns Float)
- train_bigram(lst) : Training Bigram Model (Returns Dictionary of Dictionaries)
- sum_bigram(sent, model) : Total Sum Of Bigram Probablity Of A Sentence (Returns Float)
- train_trigram(lst): Training Trigram Model (Returns Dictionary of Dictionaries)
- sum_trigram(sent, model) : Total Sum Of Trigram Probablity Of A Sentence (Returns Float)
- W2V_train(lst1, lst2) : Word2Vec Training (Returns Vector)
- W2V_Vec(sent1, sent2, vec) : Returns The Difference Between Word2Vec Sum Of All The Words In Two Sentences (Returns Vec)
- LDA_train(doc) : Trains LDA Model (Returns Model)
- LDA(doc1, doc2, lda) : Returns Average Of Probablity Of Word Present In LDA Model For Input Document (Returns Float)

#### 2. Computing text readability features.
```python
>>> from dl_text import rd_ft

>>> sent1 = 'i like natural language processing'
>>> rd_ft.CPW(sent1) # average characters per word
6.0
>>> rd_ft.ED('good','great') # edit distance between two words
4.0
```
Functions currently present in the `rd_ft` are:
- CPW(text) : Average Character Per Word In A Sentence (Returns Float)
- WPS(text) : Number Of Words Per Sentence (Returns Integer)
- SPW(text) : Average Number Of Syllables In Sentence (Returns Float)
- LWPS(text) : Long Words In A Sentence (Returns Integer)
- LWR(text) : Fraction Of Long Words In A Sentence (Returns Float)
- CWPS(text) : Number Of Complex Word Per Sentence (Returns Float)
- DaleChall(text) : Dale-Chall Readability Index (Returns Float)
- ED(s1, s2) : Edit Distance Value For Two String (Returns Integer)
- nouns(text) : Get A List Of Nouns From String (Returns List Of Sting)
- EditDist_Dist(t1,t2) : Average Edit Distance Value For Two String And The Average Edit Distance Between The Nouns Present In Them (Returns Float)
- LCS_Len(a, b) : Longest Common Subsequence (Returns Integer)
- LCW(t1, t2) : Length Of Longest Common Subsequence (Returns Integer)

## Training deep models using textutal sentences and hand features.
#### 1. Preparing the data
```python
from dl_text import dl
from dl_text import lex_sem_ft
from dl_text import rd_ft

data_l = ['this is a positive sentence','this is a negative sentence', 
          'yet another positive sentence', 'the last one is negative']
data_r = ['positive words are good, better, best, etc.', 'negative words are bad, sad, etc.', 
          'feeling good', 'sooo depressed.']
labels = [1,0,1,0]

wordVec_model = dl.loadGloveModel('path_of_the_embeddings/glove.6B.50d.txt')

all_feat = []
for i,j in zip(data_l, data_r):
    feat1 = lex_sem_ft.overlap(i, j)
    feat2 = lex_sem_ft.W2V_Vec(i, j, wordVec_model)
    feat3 = rd_ft.ED(i, j)
    feat4 = rd_ft.LCW(i, j)
    all_feat.append(feat1)
    all_feat.append(feat2)
    all_feat.append(feat3)
    all_feat.append(feat4)
    
data_inp_l, data_inp_r, embedding_matrix = dl.process_data(sent_l = data_l, sent_r = data_r, 
                                                           wordVec_model = wordVec_model, dimx = 10, dimy = 10)
```
#### 2. Let's define a model for incorporating external features with deep models.

``` python

def model_cnn_ft(dimx, dimy, dimft, embedding_matrix):
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    embedx = dl.word2vec_embedding_layer(embedding_matrix)(inpx)
    inpy = Input(shape=(dimx,),dtype='int32',name='inpy')   
    embedy = dl.word2vec_embedding_layer(embedding_matrix)(inpy)
    inpz = Input(shape=(dimft,),dtype='int32',name='inpz')
    
    sent_l = Conv1D(nb_filter=3,filter_length=2,activation='relu')(embedx)
    sent_r = Conv1D(nb_filter=3,filter_length=2,activation='relu')(embedy)
    pool_l = MaxPooling1D()(sent_l)
    pool_r = MaxPooling1D()(sent_r)
    
    combine  = merge(mode='concat')([pool_l, pool_r,inpz])
    flat_embed = Flatten()(combine)
    nnet_h = Dense(units=10,activation='sigmoid')(flat_embed)
    nnet_out = Dense(units=2,activation='sigmoid')(nnet_h)
    model = Model([inpx],nnet_out)
    model.compile(loss='mse',optimizer='adam')
    
    return model
```
#### 3. Training the deep model.
```python
model = model_cnn_ft(dimx = 10, dimy = 10, dimz = len(all_feat), embedding_matrix = embedding_matrix)
model.fit([data_inp_l, data_inp_r, all_feat], labels)
```

## Evaluation metrics - MAP, MRR, AP@k, etc.
The mean average precision (MAP) and mean reciprocal recall (MRR) is computed as:

<img src="https://github.com/GauravBh1010tt/DL-text/blob/master/img.JPG" width="550">

In our implementation we assume that the ground truth is arranged starting with the true labels and is/are followed by false labels.
```python
>>> from dl_text import metrics
>>> pred = [[0,0,1],[0,0,1]] # we have two queries with 3 answers for each; 1 - relevant, 0 - irrelevant

'''Converting the prediction list to dictionary'''

>>> dict1 = {}
>>> for i,j in enumerate(pred):
        dict1[i] = j
        
>>> metrics.Map(dict1)
0.33
>>> metrics.Mrr(dict1)
33.33

>>> pred = [[0,1,1],[0,1,0]]
>>> for i,j in enumerate(pred):
        dict1[i] = j
>>> metrics.Map(dict1)
0.5416666666666666
>>> metrics.Mrr(dict1)
50.0
```
