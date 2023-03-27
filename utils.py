import random
import re
import pypinyin
from jieba import posseg
from pypinyin import pinyin
import jieba
import os


def to_unicode(text):
    """将文本转化为utf-8编码"""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


# 判断字符类型：中文、英文、数字、其它
def is_chinese_char(c):
    """判断字符c 是否为中文"""
    if '\u4e00' <= c <= '\u9fa5':
        return True
    else:
        return False
def is_english_char(c):
    """判断字符c 是否为英文"""
    if (u'u0041' <= c <= u'u005a') or (u'u0061' <= c <= u'u007a'):
        return True
    else:
        return False
def is_number(c):
    """判断字符c 是否为数字"""
    if u'u0030' <= c <= u'u0039':
        return True
    else:
        return False
def is_other(c):
    """判断字符c 是否非汉字英文数字"""
    if not (is_chinese_char(c) or is_number(c) or is_english_char(c)):
        return True
    else:
        return False


# 判断字符串类型：中文、英文
def is_chinese_string(s):
    """判断字符串s 是否为中文"""
    for c in s:
        if not is_chinese_char(c):
            return False
    return True
def is_english_string(s):
    """判断字符串s 是否为英文"""
    for c in s:
        if c < 'a' or c > 'z':
            return False
    return True


def clean_text(text):
    r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?。，?、…【】《》“”‘’！[\\]^_`{}~：|；．（）〔〕〈〉]+'
    text = re.sub(r1, '', text)
    return text


# 取同音字
def get_homophones_by_char(input_char):
    """根据汉字取同音字"""
    result = []
    # CJK统一汉字区的范围是0x4E00-0x9FA5,即20902个汉字
    for i in range(0x4e00, 0x9fa6):
        if pinyin([chr(i)], style=pypinyin.NORMAL)[0][0] == pinyin(input_char, style=pypinyin.NORMAL)[0][0]:
            result.append(chr(i))
    return result
def get_homophones_by_pinyin(input_pinyin):
    """根据拼音取同音字"""
    result = []
    for i in range(0x4e00, 0x9fa6):
        if pinyin([chr(i)], style=pypinyin.TONE2)[0][0] == input_pinyin:
            # TONE2: 中zho1ng
            result.append(chr(i))
    return result


def split_by_sym(text, include_symbol=True):
    """
    文本切分为句子，以标点符号切分
    :param text: str
    :param include_symbol: bool
    :return: (sentence, idx)
    """
    result = []
    re_han = re.compile("([\u4E00-\u9Fa5a-zA-Z0-9+#&]+)", re.U)
    sentences = re_han.split(text)
    start_idx = 0
    for sentence in sentences:
        if not sentence:
            continue
        if include_symbol:
            result.append((sentence, start_idx))
        else:
            if re_han.match(sentence):
                result.append((sentence, start_idx))
        start_idx += len(sentence)
    return result


def split_by_maxlen(text, maxlen=512):
    """
    文本切分为句子，以句子maxlen切分
    :param text: str
    :param maxlen: int, 最大长度
    :return: list, (sentence, idx)
    """
    result = []
    for i in range(0, len(text), maxlen):
        result.append((text[i:i + maxlen], i))
    return result


def edit_distance_word(word, char_set):
    """
    all edits that are one edit away from 'word'
    :param word:
    :param char_set:
    :return:
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in char_set]
    return set(transposes + replaces)


def segment(sentence, cut_type='word', pos=False):
    """
    切词
    :param sentence:
    :param cut_type: 'word' use jieba.lcut; 'char' use list(sentence)
    :param pos: enable POS
    :return: list
    """
    if pos:
        if cut_type == 'word':
            word_pos_seq = posseg.lcut(sentence)
            word_seq, pos_seq = [], []
            for w, p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq, pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)

class Tokenizer(object):
    def __init__(self, dict_path='', custom_word_freq_dict=None, custom_confusion_dict=None):
        self.model = jieba
        # 初始化大词典
        if os.path.exists(dict_path):
            self.model.set_dictionary(dict_path)
        # 加载用户自定义词典
        if custom_word_freq_dict:
            for w, f in custom_word_freq_dict.items():
                self.model.add_word(w, freq=f)

        # 加载混淆集词典
        if custom_confusion_dict:
            for k, word in custom_confusion_dict.items():
                # 添加到分词器的自定义词典中
                self.model.add_word(k)
                self.model.add_word(word)

    def tokenize(self, unicode_sentence, mode="search"):
        """
        切词并返回切词位置, search mode用于错误扩召回
        :param unicode_sentence: query
        :param mode: search, default, ngram
        :param HMM: enable HMM
        :return: (w, start, start + width) model='default'
        """
        if mode == 'ngram':
            n = 2
            result_set = set()
            tokens = self.model.lcut(unicode_sentence)
            tokens_len = len(tokens)
            start = 0
            for i in range(0, tokens_len):
                w = tokens[i]
                width = len(w)
                result_set.add((w, start, start + width))
                for j in range(i, i + n):
                    gram = "".join(tokens[i:j + 1])
                    gram_width = len(gram)
                    if i + j > tokens_len:
                        break
                    result_set.add((gram, start, start + gram_width))
                start += width
            results = list(result_set)
            result = sorted(results, key=lambda x: x[-1])
        else:
            result = list(self.model.tokenize(unicode_sentence, mode=mode))
        return result


def find_difference(s1, s2):
    """找到 字符串s1 和 字符串s2 不同的字串"""
    # matches = []
    differences= []
    n = 1
    # create a list of the a find_difference.
    i = 0
    j = n
    aList = []
    # create a list of all possible substring of n length in a
    while(j <= len(s1) + 1):
        sub = s1[i:j]
        aList.append(sub)
        i += 1
        j += 1
    # Check each s2 substring to see if it is also an s1 substring.
    i = 0
    j = n
    while(j <= len(s2) + 1):
        sub = s2[i:j]
        # if sub not in matches and sub in aList and len(sub) == n:
        #     matches.append(sub)
        if sub not in differences and sub not in aList and len(sub) == n:
            if sub != '\r':
                differences.append(sub)

        i += 1
        j += 1

    return differences

def substrings(a, b):
    """Return substrings of length n in both a and b"""
    matches = []
    differences= []
    n = 1
    # create a list of the a substrings.
    i = 0
    j = n
    aList = []
    # create a list of all possible substring of n length in a
    while(j <= len(a) + 1):
        sub = a[i:j]
        aList.append(sub)
        i += 1
        j += 1
    # Check each b substring to see if it is also an a substring.
    i = 0
    j = n
    while(j <= len(b) + 1):
        sub = b[i:j]
        if sub not in matches and sub in aList and len(sub) == n:
            matches.append(sub)
        if sub not in differences and sub not in aList and len(sub) == n:
            differences.append(sub)

        i += 1
        j += 1

    return matches


def generate_error(sentence_lst):
    generated_lst = []
    for sentence in sentence_lst:
        tp = random.randint(1, 2)
        # 1. 单字近音错误
        if tp == 1:
            i = random.randint(0, len(sentence)-1)
            generated = sentence.replace(sentence[i], get_homophones_by_char(sentence[i])[0])
        else:
            generated = sentence
        # 2. 单字近形错误
        generated_lst.append(generated)
    return generated_lst




if __name__ == "__main__":
    print(is_english_string('nihao'))
    print(is_other(','))
    print(is_chinese_char('你'))
    print(is_chinese_string('你,'))
    print(is_chinese_string('你hao'))

    sentence = '近日，中共中央、国务院、中央军委印发了《军队功勋荣誉表彰条例》'
    generate_error(sentence)
