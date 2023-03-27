
import operator
import time
import os
from transformers import BertTokenizer, BertForMaskedLM
import torch

import config
from utils import split_by_maxlen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
unk_tokens = [' ', '“', '”', '‘', '’', '\n', '…', '—', '\t', '֍', '']


def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, origin_char in enumerate(origin_text):
        if i >= len(corrected_text):
            continue
        if origin_char in unk_tokens:
            corrected_text = corrected_text[:i] + origin_char + corrected_text[i:]
            continue
        if origin_char != corrected_text[i]:
            if origin_char.lower() == corrected_text[i]:
                # 不处理英语大写字母
                corrected_text = corrected_text[:i] + origin_char + corrected_text[i + 1:]
                continue
            sub_details.append((origin_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details


class MacBertCorrector(object):
    def __init__(self, macbert_model_dir=config.macbert_model_dir):
        super(MacBertCorrector, self).__init__()
        self.name = 'macbert_corrector'
        self.tokenizer = BertTokenizer.from_pretrained(macbert_model_dir)
        self.model = BertForMaskedLM.from_pretrained(macbert_model_dir)
        self.model.to(device)

    def correct(self, text):
        """
        句子纠错
        :param text: 句子文本
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        text_new = ''
        details = []
        # 长句切分为短句
        blocks = split_by_maxlen(text, maxlen=128)
        blocks = [block[0] for block in blocks]
        inputs = self.tokenizer(blocks, padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        for ids, text in zip(outputs.logits, blocks):
            decode_tokens = self.tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).replace(' ', '')
            corrected_text = decode_tokens[:len(text)]
            corrected_text, sub_details = get_errors(corrected_text, text)
            text_new += corrected_text
            details.extend(sub_details)
        return text_new, details


if __name__ == "__main__":
    d = MacBertCorrector()
    sentence_lst = [
        '国物院办公厅关与推动公立医院高高质量发展的yi见',
    ]
    for sentence in sentence_lst:
        corrected_sent, err = d.correct(sentence)
        print('\n原句: ' + sentence)
        if err == []:
            print('正确')
        else:
            print('改正：' + str(corrected_sent) + '\n错误：' + str(err))