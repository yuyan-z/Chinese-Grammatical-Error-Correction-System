# Chinese-Grammatical-Error-Correction-System
My undergraduate thesis in 2022 at Shanghai University

## Introduction
Grammatical Error Correction (GEC) is an important Natural Language Processing task, which aims to detect and correct errors in text, such as spelling, grammar and punctuation. 

English GEC techniques started early in the 1960s and developed rapidly. It mainly focuses on words. There are two main types of error:
- Non-word error. The word does not exist, for example, "bag" is misspelled as "bga". 
- Real word error. The word exists but does not fit the context, for example, 'bag' is misspelled as 'bad'. Real word error is a contextual error related to grammar and semantics, and is a major challenge in text correction. 

The development of Chinese GEC techniques is relatively late beginning in the early 1990s. The characteristics of Chinese make the GEC task more difficult than that of English: The grammatical semantics of Chinese are more flexible and difficult to define by rules; There are no separators between Chinese words and an automatic word separation process must be defined; Wrong words may affect the effect of word separation, further increasing the difficulty. 


This project studies Chinese GEC algorithms and develops a Text Error Correction System.

## Program Design
![design](https://user-images.githubusercontent.com/64955334/223971090-5b9d7060-4797-4bea-9d86-557f2a5223f9.jpg)

## Web Application
![中文文本纠错系统 - http___127 0 0 1_5000_correcting](https://user-images.githubusercontent.com/64955334/223094069-47dc6bd8-7eb3-4091-88be-e1d72f541ea3.png)

Error correction knowledge base and confusion dictionary
![纠错知识库](https://user-images.githubusercontent.com/64955334/223094108-7ffdd162-90e5-424d-8e6c-af96758548cc.png)

