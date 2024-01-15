---
layout : post
title: Tokenization(토큰화)
author: Hojoon_Kim
date: 2024-01-15 20:15:10 +0900
categories: [Develope, NLP]
tags: [nlp,tokenization,토큰]
render_with_liquid: false
---
# Tokenization(토큰화)
자연어 처리에서 얻은 코퍼스(corpus) 데이터가 학습시키기에 적당하게 전처리 되지 않은 상태라면, 보통 의미가 있는 단위로 토큰으로 변환시킨다.
#
## 토큰화의 종류
### 1. 단어 토큰화
단어 토큰화는 단어를 기준으로 토큰화를 진행한다. 보통 의미있는 단위로 토큰을 정의한다.
```python
from nltk.tokenize import word_tokenize
print(word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
```
```
['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
```
위의 결과를 보면, Don't가 Do와 n't로 분리되어있는 것을 볼 수 있다. 이는 영어의 경우에는 단어 토큰화를 사용해도 무방하다. 하지만 한국어의 경우에는 단어 토큰화를 사용하면 의미가 없는 토큰이 생성되기도 한다.
```python
from nltk.tokenize import word_tokenize
print(word_tokenize("고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든."))
```
```
['고기를', '아무렇게나', '구우려고', '하면', '안', '돼', '.', '고기라고', '다', '같은', '게', '아니거든', '.']
```
위의 결과를 보면, '고기를'과 '고기라고'가 다른 토큰으로 분류되어있는 것을 볼 수 있다. 이는 한국어의 경우에는 단어 토큰화를 사용하면 의미가 없는 토큰이 생성되기도 한다는 것을 의미한다. 이를 해결하기 위해 한국어는 보통 형태소 토큰화를 사용한다.
#
### 2. 문장 토큰화
문장 토큰화는 문장을 기준으로 토큰화를 진행한다. 보통 문장의 마침표를 기준으로 토큰화를 진행한다.
```python
from nltk.tokenize import sent_tokenize
print(sent_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
```
```
["Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."]
```
위의 결과를 보면, 마침표를 기준으로 토큰화를 진행한 것을 볼 수 있다. 하지만, 문장 토큰화는 마침표가 단순한 구분자가 아닌 경우에는 제대로 작동하지 않는다.
```python
from nltk.tokenize import sent_tokenize
print(sent_tokenize("I am actively looking for Ph.D. students. and you are a Ph.D student."))
```
```
['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
```
위의 결과를 보면, Ph.D.를 문장의 마침표로 인식하여 문장을 구분한 것을 볼 수 있다. 이러한 문제를 해결하기 위해 정규 표현식을 사용하여 문장 토큰화를 진행한다.
```python
from nltk.tokenize import sent_tokenize
import re
text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(text))
print(re.split('(?<=[.]) +', text))
```
```
['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
```
위의 결과를 보면, 정규 표현식을 사용하여 문장 토큰화를 진행한 것을 볼 수 있다. 하지만, 정규 표현식을 사용하여 문장 토큰화를 진행하면, Ex) Ph.D.와 같은 단어는 Ph.D와 같이 토큰화가 되어버린다. 이러한 문제를 해결하기 위해 NLTK는 영어 문장의 토큰화를 위한 도구를 제공한다.
```python
from nltk.tokenize import sent_tokenize
text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(text))
```
```
['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
```
위의 결과를 보면, Ph.D.를 Ph.D와 같이 토큰화한 것을 볼 수 있다. 이러한 도구는 한국어에는 존재하지 않는다.

## 한국어 토큰화
한국어는 어절 토큰화와 형태소 토큰화를 사용한다.
### 1. KoNLPy
KoNLPy는 한국어 자연어 처리를 위한 파이썬 패키지이다. KoNLPy는 다음과 같은 형태소 분석기를 제공한다.
- Hannanum
- Kkma
- Komoran
- Mecab
- Okt(Twitter)
#
#### 1.1. Hannanum

```python
from konlpy.tag import Hannanum
hannanum = Hannanum()
print(hannanum.analyze(u'아버지가방에들어가신다.'))
print(hannanum.morphs(u'아버지가방에들어가신다.'))
print(hannanum.nouns(u'아버지가방에들어가신다.'))
print(hannanum.pos(u'아버지가방에들어가신다.'))
```
```
[[[('아버지', 'ncn'), ('가방', 'ncn'), ('에', 'jca')], [('아버지', 'ncn'), ('가방', 'ncn'), ('에', 'jcs'), ('들', 'px'), ('어', 'ecx'), ('가', 'px'), ('시', 'ep'), ('ㄴ다', 'ef')]], [[('아버지', 'ncn'), ('가방', 'ncn'), ('에', 'jca'), ('들', 'px'), ('어', 'ecx'), ('가', 'px'), ('시', 'ep'), ('ㄴ다', 'ef')]]]
['아버지', '가방', '에', '들', '어', '가', '시', 'ㄴ다']
['아버지', '가방']
[('아버지', 'N'), ('가방', 'N'), ('에', 'J'), ('들', 'P'), ('어', 'E'), ('가', 'P'), ('시', 'E'), ('ㄴ다', 'E')]
```

#
#### 1.2. Kkma

```python
from konlpy.tag import Kkma
kkma = Kkma()
print(kkma.sentences(u'아버지가방에들어가신다.'))
print(kkma.nouns(u'아버지가방에들어가신다.'))
print(kkma.pos(u'아버지가방에들어가신다.'))
```
```
['아버지가 방에 들어가신다.']
['아버지', '방']
[('아버지', 'NNG'), ('가', 'JKS'), ('방', 'NNG'), ('에', 'JKM'), ('들어가', 'VV'), ('시', 'EPH'), ('ㄴ다', 'EFN'), ('.', 'SF')]
```

#
#### 1.3. Komoran

```python
from konlpy.tag import Komoran
komoran = Komoran()
print(komoran.morphs(u'아버지가방에들어가신다.'))
print(komoran.nouns(u'아버지가방에들어가신다.'))
print(komoran.pos(u'아버지가방에들어가신다.'))
```
```
['아버지', '가방', '에', '들어가', '시', 'ㄴ다', '.']
['아버지', '가방']
[('아버지', 'NNP'), ('가방', 'NNP'), ('에', 'JKB'), ('들어가', 'VV'), ('시', 'EP'), ('ㄴ다', 'EF'), ('.', 'SF')]
```

#
#### 1.4. Mecab

```python
from konlpy.tag import Mecab
mecab = Mecab()
print(mecab.morphs(u'아버지가방에들어가신다.'))
print(mecab.nouns(u'아버지가방에들어가신다.'))
print(mecab.pos(u'아버지가방에들어가신다.'))
```
```
['아버지', '가방', '에', '들어가', '신다', '.']
['아버지', '가방']
[('아버지', 'NNG'), ('가방', 'NNG'), ('에', 'JKB'), ('들어가', 'VV'), ('신다', 'EP+EC'), ('.', 'SF')]
```

#
#### 1.5. Okt(Twitter)

```python
from konlpy.tag import Okt
okt = Okt()
print(okt.morphs(u'아버지가방에들어가신다.'))
print(okt.nouns(u'아버지가방에들어가신다.'))
print(okt.pos(u'아버지가방에들어가신다.'))
```
```
['아버지', '가방', '에', '들어가신다', '.']
['아버지', '가방']
[('아버지', 'Noun'), ('가방', 'Noun'), ('에', 'Josa'), ('들어가신다', 'Verb'), ('.', 'Punctuation')]
```

#
##### TAG 의미 확인 링크 -> [링크](https://docs.google.com/spreadsheets/d/1OGAjUvalBuX-oZvZ_-9tEfYD2gQe7hTGsgUpiiBSXI8/edit#gid=0)

### 성능 비교 코드
```python
#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

from time import time

from konlpy import tag
from konlpy.corpus import kolaw
from konlpy.utils import csvwrite, pprint


def tagging(tagger, text):
    r = []
    try:
        r = getattr(tag, tagger)().pos(text)
    except Exception as e:
        print "Uhoh,", e
    return r


def measure_time(taggers, mult=6):
    doc = kolaw.open('constitution.txt').read()*6
    data = [['n'] + taggers]
    for i in range(mult):
        doclen = 10**i
        times = [time()]
        diffs = [doclen]
        for tagger in taggers:
            r = tagging(tagger, doc[:doclen])
            times.append(time())
            diffs.append(times[-1] - times[-2])
            print '%s\t%s\t%s' % (tagger[:5], doclen, diffs[-1])
            pprint(r[:5])
        data.append(diffs)
        print
    return data


def measure_accuracy(taggers, text):
    print '\n%s' % text
    result = []
    for tagger in taggers:
        print tagger,
        r = tagging(tagger, text)
        pprint(r)
        result.append([tagger] + map(lambda s: ' / '.join(s), r))
    return result


def plot(result):

    from matplotlib import pylab as pl
    import scipy as sp

    if not result:
        result = sp.loadtxt('morph.csv', delimiter=',', skiprows=1).T

    x, y = result[0], result[1:]

    for i in y:
        pl.plot(x, i)

    pl.xlabel('Number of characters')
    pl.ylabel('Time (sec)')
    pl.xscale('log')
    pl.grid(True)
    pl.savefig("images/time.png")
    pl.show()


if __name__=='__main__':

    PLOT = False
    MULT = 6

    examples = [u'아버지가방에들어가신다',  # 띄어쓰기
            u'나는 밥을 먹는다', u'하늘을 나는 자동차', # 중의성 해소
            u'아이폰 기다리다 지쳐 애플공홈에서 언락폰질러버렸다 6+ 128기가실버ㅋ'] # 속어

    taggers = [t for t in dir(tag) if t[0].isupper()]

    # Time
    data = measure_time(taggers, mult=MULT)
    with open('morph.csv', 'w') as f:
        csvwrite(data, f)

    # Accuracy
    for i, example in enumerate(examples):
        result = measure_accuracy(taggers, example)
        result = map(lambda *row: [i or '' for i in row], *result)
        with open('morph-%s.csv' % i, 'w') as f:
            csvwrite(result, f)

    # Plot
    if PLOT:
        plot(result)

```
[REFERENCE](https://konlpy.org/ko/v0.6.0/morph/)