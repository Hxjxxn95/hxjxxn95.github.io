---
layout : post
title: Integer Encoding
author: Hojoon_Kim
date: 2024-01-21 14:30:10 +0900
categories: [Develope, NLP]
tags: [Pytorch, DL, NLTK, IntegerEncoding]
pin: true
math: true
mermaid: true
---
일반적으로 텍스트보단 숫자를 더 잘 처리한다. 그래서 자연어 처리를 하기 위해서는 텍스트를 숫자로 바꿔줘야 한다. 이 과정을 텍스트를 숫자로 바꾸는 인코딩(Encoding)이라고 한다. 보통은 토큰화된 텍스트를 빈도수에 따라 정렬한 후에 부여한다.

## Dictionary 사용하기
```python
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

setences = sent_tokenize(raw_text)
stop_words = set(stopwords.words('english'))
vocab = {}

for sentence in setences:
    words = word_tokenize(sentence)
    words = [w for w in words if not w in stop_words]
    for word in words:
        if word not in vocab:
            vocab[word] = 0
        vocab[word] += 1

vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)

word_to_index = {}
i = 0
for (word, frequency) in vocab_sorted:
    if frequency > 1:
        i += 1
        word_to_index[word] = i

```
이렇게 한 뒤 단어는 빈도수가 낮은 (빈도수가 1인) 단어들은 제외시키고, 빈도수가 높은 상위 n개의 단어만 사용한다. 여기서는 빈도수 상위 5개의 단어만 사용한다고 가정한다. 그러면, 다음과 같이 출력된다.
```python
word_freq = [w for w,c in word_to_index.items() if c >= 6]

for w in word_freq:
    del word_to_index[w]
```
```
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
```

만약 이렇게 인코딩하게 원문장에 있던 단어들이 사라져 OOV 문제가 발생하게 된다. 이를 해결하기 위해 빈도수가 적은 단어들을 하나의 단어로 통합시키는 방법이 있다.

```python
word_to_index['OOV'] = len(word_to_index) + 1
```
```
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'OOV': 6}
```

```python
encoded_sentences = []
for sentence in preprocessed_sentences:
    encoded_sentence = []
    for word in sentence:
        try:
            # 단어 집합에 있는 단어라면 해당 단어의 정수를 리턴.
            encoded_sentence.append(word_to_index[word])
        except KeyError:
            # 만약 단어 집합에 없는 단어라면 'OOV'의 정수를 리턴.
            encoded_sentence.append(word_to_index['OOV'])
    encoded_sentences.append(encoded_sentence)
print(encoded_sentences)
```
```
[[1, 5], [1, 6, 5], [1, 3, 5], [6, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [6, 6, 3, 2, 6, 1, 6], [1, 6, 3, 6]]
```

## Counter 사용하기
Counter 를 사용하면 단어의 빈도수를 좀 더 쉽게 계산할 수 있다.
```python
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

words = word_tokenize(raw_text)
vocab = Counter(words)
vocab
```
```
Counter({'barber': 8, 'secret': 6, 'huge': 5, 'kept': 4, 'person': 3, 'word': 2, 'keeping': 2, 'good': 1, 'knew': 1, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1})
```

## NLTK의 FreqDist 사용하기
```python
from nltk import FreqDist

words = word_tokenize(raw_text)
vocab = FreqDist(words)
vocab
```
```
FreqDist({'barber': 8, 'secret': 6, 'huge': 5, 'kept': 4, 'person': 3, 'word': 2, 'keeping': 2, 'good': 1, 'knew': 1, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1})
```

## Keras의 텍스트 전처리

```python
from tensorflow.keras.preprocessing.text import Tokenizer

preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

tokenizer = Tokenizer()

vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 2, oov_token = 'OOV') # 상위 5개 단어만 사용
# Keras 는 OOV 를 기본적으로는 제거하나, 사용하기 위해서는 oov_token = 'OOV' 를 추가해줘야 하고, 기본적으로 OOV 의 인덱스는 1이다.
tokenizer.fit_on_texts(preprocessed_sentences)
```


## Reference
[위키독스 - 딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155)

