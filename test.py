import random

from bert_corrector import BertCorrector
from macbert_corrector import MacBertCorrector
from utils import generate_error

d = BertCorrector()
# d = MacBertCorrector()

def correct_input():
    """输入文本后改正"""
    sentence_lst = []
    print('请输入句子，q结束：')
    while True:
        text = input()
        # print(sentence)
        if text == 'q':
            print('瞎话鉴定机启动...')
            break
        else:
            for sentence in text.split('。'):
                sentence_lst.append(sentence)

    for sentence in sentence_lst:
        corrected_sent, err = d.correct(sentence)
        print('\n原句: ' + sentence)
        if err == []:
            print('正确')
        else:
            print('改正：' + str(corrected_sent) + '--错误：' + str(err))


def corrct_file(file_path):
    """读取文件后改正"""
    sentence_lst = []
    for sentence in open(file_path, "r",encoding='utf-8'):
        sentence_lst.append(sentence.strip())

    if len(sentence_lst) > 100:
        sentence_lst = random.sample(sentence_lst, 5)

    # 生成错误
    sentence_lst = generate_error(sentence_lst)

    for sentence in sentence_lst:
        corrected_sent, err = d.correct(sentence)
        print('\n原句: ' + sentence)
        if err == []:
            print('正确')
        else:
            print('改正：' + str(corrected_sent) + '\n错误：' + str(err))


def correct_evaluate(file_path):
    """读取文件后改正"""
    sentence_lst = []
    for sentence in open(file_path, "r",encoding='utf-8'):
        sentence_lst.append(sentence.strip())

    for sentence in sentence_lst:
        corrected_sent, err = d.correct(sentence)
        print('\n原句: ' + sentence)
        if err == []:
            print('正确')
        else:
            print('改正：' + str(corrected_sent) + '\n错误：' + str(err))

def generate_error_file(file_path):
    """生成错误文件"""
    sentence_lst = []
    for sentence in open(file_path, "r",encoding='utf-8'):
        sentence_lst.append(sentence.strip())

    if len(sentence_lst) > 100:
        sentence_lst = random.sample(sentence_lst, 5)

    # 生成错误
    sentence_lst = generate_error(sentence_lst)

    f = open("test2.txt", "w")
    f.write('\n'.join(sentence_lst))
    f.close()

if __name__ == '__main__':

    #correct_input()

    file_path = "data/zhengce.txt"
    corrct_file(file_path)

    #generate_error_file(file_path)