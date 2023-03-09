# Chinese-Grammatical-Error-Correction-System
My undergraduate thesis in 2022 at Shanghai University

## Introduction
Grammatical Error Correction (GEC) is an important Natural Language Processing task, which aims to detect and correct errors in text, such as spelling, grammar and punctuation. 

English GEC techniques started early in the 1960s and developed rapidly. It mainly focuses on words. There are two main types of error: Non-word error and real word error.

The development of Chinese GEC techniques is relatively late beginning in the early 1990s. The characteristics of Chinese make the GEC task more difficult than that of English: The grammatical semantics of Chinese are more flexible and difficult to define by rules; There are no separators between Chinese words and an automatic word separation process must be defined; Errors may affect the effect of word separation, further increasing the difficulty. It is therefore mainly focused on contextual errors, such as redundant words, missing words and wrong words.

This project studies Chinese GEC algorithms and develops a Text Error Correction System. The main tasks are as follows:
- Construction of correcting knowledge set: build a crawler to collect 165,000 items of text from gov.cn with Scrapy.
- Comparison and evaluation of different grammatical error correction models based on rule, N-gram, and BERT
- Hosted models on the website with Flask and developed a web application with Bootstrap4

## Methodology
There are four main methods of Chinese GEC in common use.
- Rules and correcting knowledge set
- Statistical language model
- Deep learning model
- Hybrid model

## Results
![results](https://user-images.githubusercontent.com/64955334/223989361-34db28b6-3396-46d0-9c6c-10a89bd8bf64.png)


## Web Application

Program design

![design](https://user-images.githubusercontent.com/64955334/223971090-5b9d7060-4797-4bea-9d86-557f2a5223f9.jpg)

Text error correction web app 

![中文文本纠错系统 - http___127 0 0 1_5000_correcting](https://user-images.githubusercontent.com/64955334/223094069-47dc6bd8-7eb3-4091-88be-e1d72f541ea3.png)

Correcting knowledge set and confusion dictionary

![纠错知识库](https://user-images.githubusercontent.com/64955334/223094108-7ffdd162-90e5-424d-8e6c-af96758548cc.png)

