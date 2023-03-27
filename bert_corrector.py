import codecs
import operator
import os
import time

import jieba
import torch
from pypinyin import lazy_pinyin
from transformers import pipeline

from utils import is_chinese_string, split_by_sym
import config

device_id = 0 if torch.cuda.is_available() else -1


# print(device_id)


class BertCorrector():
    def __init__(self, device=device_id):
        self.name = 'bert_corrector'
        self.model = pipeline(
            'fill-mask',
            model=config.bert_model_dir,
            tokenizer=config.bert_model_dir,
            device=device,  # gpu device id
        )
        if self.model:
            self.mask = self.model.tokenizer.mask_token

        self.initialized_corrector = False

        self.common_char = None
        self.custom_confusion = None
        self.word_freq = None
        self.same_pinyin = None
        self.same_stroke = None

    @staticmethod
    def load_common_char(path):
        """
        加载常用字集合
        """
        common_char = set()
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                if line:
                    common_char.add(line)
        # print(common_char)
        return common_char

    @staticmethod
    def load_custom_confusion(path):
        """
        加载自定义混淆字典
        """
        custom_confusion = dict()
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split()  # 易错 纠正
                if len(parts) < 2:
                    continue
                confusion = parts[0]
                correct = parts[1]
                # 取词频，默认1
                custom_confusion[confusion] = correct
        # print(custom_confusion)
        return custom_confusion

    @staticmethod
    def load_word_freq_dict(path):
        """
        加载词频字典
        """
        word_freq = dict()
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split()  # 词 频率
                if len(parts) < 2:
                    continue
                word = parts[0]
                freq = int(parts[1])
                word_freq[word] = freq
        # print(word_freq)
        return word_freq

    @staticmethod
    def load_same_pinyin(path):
        """
        加载同音字字典
        """
        same_pinyin = dict()
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split('\t')
                if parts and len(parts) > 2:
                    key_char = parts[0]
                    same_pron_same_tone = set(list(parts[1]))  # 同音同调
                    same_pron_diff_tone = set(list(parts[2]))  # 同音异调
                    value = same_pron_same_tone.union(same_pron_diff_tone)
                    if key_char and value:
                        same_pinyin[key_char] = value
        # print(same_pinyin)
        return same_pinyin

    @staticmethod
    def load_same_stroke(path):
        """
        加载形似字字典
        """
        same_stroke = dict()
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split('\t')
                if parts and len(parts) > 1:
                    for i, c in enumerate(parts):
                        same_stroke[c] = set(list(parts[:i] + parts[i + 1:]))
        # print(same_stroke)
        return same_stroke

    @staticmethod
    def _initialize_corrector(self):
        self.common_char = self.load_common_char(config.common_char_path)
        self.custom_confusion = self.load_custom_confusion(config.custom_confusion_path)
        self.word_freq = self.load_word_freq_dict(config.word_freq_path)
        self.same_pinyin = self.load_same_pinyin(config.same_pinyin_path)
        self.same_stroke = self.load_same_stroke(config.same_stroke_path)

        self.initialized_corrector = True

    def check_corrector_initialized(self):
        if not self.initialized_corrector:
            self._initialize_corrector(self)

    def get_same_pinyin(self, char):
        """
        取该字的同音字集合
        """
        self.check_corrector_initialized()
        return self.same_pinyin.get(char, set())

    def get_same_stroke(self, char):
        """
        取该字的形似字集合
        """
        self.check_corrector_initialized()
        return self.same_stroke.get(char, set())

    def get_confusion_char(self, c):
        """
        取该字的同音字和形近字集合作为混淆字集合
        """
        return self.get_same_pinyin(c).union(self.get_same_stroke(c))

    def get_common_word(self, words):
        """
        取words中 在word_freq中的词 的集合，作为常用词集合
        """
        common_word = set()
        for word in words:
            if word in self.word_freq:
                common_word.add(word)
        return common_word

    @staticmethod
    def get_edit_word(word, char_set):
        """
        返回在该词word编辑距离内的词的集合
        """
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        # deletes = [L + R[1:] for L, R in splits if R]
        # transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in char_set]
        # inserts = [L + c + R for L, R in splits for c in char_set]
        # print(set(deletes + transposes + replaces))
        return set(replaces)

    def word_frequency(self, word):
        """
        取该词的词频
        """
        return self.word_freq.get(word, 0)

    def get_confusion_word(self, word):
        confusion = set()
        # 自定义混淆字典
        if word in self.custom_confusion:
            confusion.add(self.custom_confusion[word])
        # print(word, confusion)

        candidate_words = list(self.get_common_word(self.get_edit_word(word, self.common_char)))
        candidate_words.sort(key=lambda k: self.word_frequency(k), reverse=True)

        for candidate_word in candidate_words:
            if lazy_pinyin(candidate_word) == lazy_pinyin(word):
                # same pinyin
                confusion.add(candidate_word)
            elif len(candidate_word) > 1:
                confusion.add(candidate_word)

        # print(confusion)
        return confusion

    def generate_candidate(self, word, fragment=1):
        """
        生成该词的word纠错候选集
        """
        self.check_corrector_initialized()
        # 1字
        candidates_1 = []

        candidates_1.extend(self.get_confusion_word(word))
        # 同音字和形近字
        if len(word) == 1:
            confusion = [i for i in self.get_confusion_char(word[0]) if i]
            candidates_1.extend(confusion)

        # add all confusion word list
        confusion_word_set = set(candidates_1)
        confusion_word_list = [item for item in confusion_word_set if is_chinese_string(item)]
        confusion_sorted = sorted(confusion_word_list, key=lambda k: self.word_frequency(k), reverse=True)
        return confusion_sorted[:len(confusion_word_list) // fragment + 1]

    def correct(self, text):
        """
        句子纠错
        :param text: 文本
        :return: 改正后的文本corrected_text, 错误[error, correct, begin_idx, end_idx]
        """
        print('\n')
        print('原句:', text)
        text_correct = ''
        err = []
        self.check_corrector_initialized()

        blocks = split_by_sym(text)
        for block, start_idx in blocks:
            block_correct = ''
            idx = start_idx
            idx_unc = start_idx
            for c in block:


                # print(block)
                # print(block_correct, block[idx_unc:])


                block_lst_1 = list(block_correct + block[idx_unc:])
                block_lst_2 = list(block_correct + block[idx_unc:])

                block_lst_1[idx] = self.mask  # [MASK]替换c，实现换词纠错
                block_lst_2.insert(idx, '[MASK]')  # c前插入[MASK]，实现缺字纠错

                sentence_mask_1 = ''.join(block_lst_1)
                sentence_mask_2 = ''.join(block_lst_2)
                # print(idx)
                # print(sentence_mask_1)
                predicts_2 = self.model(sentence_mask_2)
                # 更新sentence_mask_2的token_str
                for predict in predicts_2:
                    predict['token_str'] = predict.get('token_str', '') + c
                    # print(predict)

                predicts = self.model(sentence_mask_1) + predicts_2
                # print(self.model(sentence_mask_1))

                bert_candidates = dict()
                for predict in predicts:
                    # token_id = predict.get('token', 0)
                    # token_str = self.model.tokenizer.convert_ids_to_tokens(token_id)
                    token_str = predict.get('token_str', '')
                    score = predict.get('score', 0)
                    if is_chinese_string(token_str):
                        bert_candidates[token_str] = score
                # print('BERT候选集: ', bert_candidates)    # BERT对每个字预测的token, score

                bert_candidates_next = dict()
                c_next = ''
                try:
                    c_next = block[idx_unc + 1]
                    block_lst_3 = list(block_correct + block[idx_unc:])
                    del block_lst_3[idx]
                    block_lst_3[idx] = self.mask
                    sentence_mask_3 = ''.join(block_lst_3)
                    for predict in self.model(sentence_mask_3):
                        token_str = predict.get('token_str', '')
                        score = predict.get('score', 0)
                        if is_chinese_string(token_str):
                            bert_candidates_next[token_str] = score
                    # print(sentence_mask_3)
                    # print('下一个字的BERT候选集: ', bert_candidates_next)
                except:
                    pass

                # 若当前字s不在BERT候选集，或score<0.5，则开始检错
                if bert_candidates.get(c, 0) < 0.5:
                    candidates = self.generate_candidate(c)  # 规则法生成该字的候选集
                    # print('开始纠错...')
                    # print('规则候选集: ', candidates)

                    for token in bert_candidates.keys():
                        if token != c and token in candidates and bert_candidates[token] > 0.5:
                            err.append((c, token, start_idx + idx, start_idx + idx + 1))
                            print('当前词:', c, '，修改为:', token)
                            c = token
                            break
                        elif len(token) > len(c) and bert_candidates[token] > 0.9:
                            err.append((c, token, start_idx + idx, start_idx + idx + 1))
                            print('当前词:', c, '，添字为:', token)
                            c = token
                            idx += 1
                            break
                        elif bert_candidates_next and c_next in bert_candidates_next and bert_candidates_next[
                            c_next] > 0.9:
                            # c的下一个字 在 删除c后的c的下一个字候选集中，即c不影响下一个字符
                            err.append((c, '', start_idx + idx, start_idx + idx + 1))
                            print('当前词:', c, '，删除')
                            c = ''
                            idx -= 1
                            break
                block_correct += c
                idx += 1
                idx_unc += 1
            text_correct += block_correct
        err = sorted(err, key=operator.itemgetter(2))
        return text_correct, err


if __name__ == "__main__":
    bertcorrector = BertCorrector()
    sentence_lst = [
        '国物院办公厅关与推动公立医院高高质量发展的见',
    ]
    for sentence in sentence_lst:
        corrected_sent, err = bertcorrector.correct(sentence)
        print('\n原句: ' + sentence)
        if err == []:
            print('正确')
        else:
            print('改正：' + str(corrected_sent) + '\n错误：' + str(err))
