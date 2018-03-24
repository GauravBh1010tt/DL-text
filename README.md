# DL - Text pre-preprocessing modules for deep learning.
This repository provide you with modules for pre-processing the textual data. There are many additional functionilities including the implemntation of evaluation metrics such as MAP, MRR, AP@k, etc.

### Usage - Prepare the data for training a deep model (DNN, CNN, RNN, LSTM).
#### 1. The data and labels looks like this:
```python
data = ['this is a positve sentence',
        'this is a negative sentence',
        'yet another positve sentence',
        'the last one is negative']
labels = [1,0,1,0]
```
This type of data is commonly used in sentiment analysis type problems.
```python
import dl
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
import dl
import gensim

# for 50-dim glove embeddings use:
wordVec_model = dl.loadGloveModel('path_of_the_embeddings/glove.6B.50d.txt')

# for 300 dim word2vec embeddings use: 
wordVec_model = gensim.models.KeyedVectors.load_word2vec_format("path/GoogleNews-vectors-negative300.bin.gz",binary=True)

data_inp, embedding_matrix = dl.process_data(sent_l = data, wordVec_model = wordVec_model, dimx = 10)
```

#### 3. Defining deep models

```python
import dl
from keras.layers import Input, Dense, Dropout, Merge, Conv1D, Lambda, Flatten, MaxPooling1D

def model_dnn(dimx, embedding_matrix):
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    embed = df.word2vec_embedding_layer(embedding_matrix)(inpx)
    flat_embed = Flatten()(embed)
    nnet_h = Dense(units=10,activation='sigmoid')(flat_embed)
    nnet_out = Dense(units=2,activation='sigmoid')(nnet_h)
    model = Model([inpx],nnet_out)
    model.compile(loss='mse',optimizer='adam')
    return model

def model_cnn(dimx, embedding_matrix):
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    embed = df.word2vec_embedding_layer(embedding_matrix)(inpx)
    sent = Conv1D(nb_filter=3,filter_length=2,activation='relu')(embed)
    pool = MaxPooling1D(sent)
    flat_embed = Flatten()(pool)
    nnet_h = Dense(units=10,activation='sigmoid')(flat_embed)
    nnet_out = Dense(units=2,activation='sigmoid')(nnet_h)
    model = Model([inpx],nnet_out)
    model.compile(loss='mse',optimizer='adam')
    return model
```
#### 4. Training the models

```python
data = ['this is a positve sentence', 'this is a negative sentence', 'yet another positve sentence', 'the last one is negative']
labels = [1,0,1,0]

data_inp, embedding_matrix = dl.process_data(sent_l = data, wordVec_model = wordVec_model, dimx = 10)

model = model_dnn(dimx = 10, embedding_matrix = embedding_matrix)
model.fit(data_inp, labels)

model = model_cnn(dimx = 10, embedding_matrix = embedding_matrix)
model.fit(data_inp, labels)
```

##### 5. Creating two channel models
These type of models use two data streams. This can be used to NLP tasks such as question answering, sentence similarity computation, etc. The data looks like this

```python
data_l = ['this is a positve sentence', 
          'this is a negative sentence', 
          'yet another positve sentence', 
          'the last one is negative']
          
data_r = ['positive words are good, better, best, etc.', 
          'negative words are bad, sad, etc.', 
          'feeling good', 
          'sooo depressed.']
          
labels = [1,0,1,0]
```

Let's define a model for the these type of tasks

``` python

def model_cnn2(dimx, dimy, embedding_matrix):
    inpx = Input(shape=(dimx,),dtype='int32',name='inpx')   
    embedx = df.word2vec_embedding_layer(embedding_matrix)(inpx)
    inpy = Input(shape=(dimx,),dtype='int32',name='inpy')   
    embedy = df.word2vec_embedding_layer(embedding_matrix)(inpy)
    
    sent_l = Conv1D(nb_filter=3,filter_length=2,activation='relu')(embedx)
    sent_r = Conv1D(nb_filter=3,filter_length=2,activation='relu')(embedy)
    pool_l = MaxPooling1D(sent_l)
    pool_r = MaxPooling1D(sent_r)
    
    combine  = merge(mode='concat')([pool_l, pool_r])
    flat_embed = Flatten()(combine)
    nnet_h = Dense(units=10,activation='sigmoid')(flat_embed)
    nnet_out = Dense(units=2,activation='sigmoid')(nnet_h)
    model = Model([inpx],nnet_out)
    model.compile(loss='mse',optimizer='adam')
    
    return model
```
#### 6. Tarining a two channel deep model

```python

data_inp_l, data_inp_r, embedding_matrix = dl.process_data(sent_l = data_l, sent_r = data_r, wordVec_model = wordVec_model, dimx = 10, dimy = 10)

model = model_cnn2(dimx = 10, dimy = 10, embedding_matrix = embedding_matrix)
model.fit([data_inp_l, data_inp_r], labels)
```

