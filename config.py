
import os

# 当前路径
pwd_path = os.path.abspath(os.path.dirname(__file__))
# print("pwd_path",pwd_path)

# 模型路径
# 统计语言模型
language_model_path = os.path.join(pwd_path, 'models/lm/people_chars_lm.klm')

# bert-base-chinese
# bert_model_dir = os.path.join(pwd_path, 'models/bert_base_chinese/')
# bert_model_path = os.path.join(pwd_path, 'models/bert_base_chinese/pytorch_model.bin')
# bert_config_path =os.path.join(pwd_path,  'models/bert_base_chinese/config.json')

# bert-finetuned
bert_model_dir = os.path.join(pwd_path, 'models/bert_finetuned/')
bert_model_path = os.path.join(pwd_path, 'models/bert_finetuned/pytorch_model.bin')
bert_config_path =os.path.join(pwd_path,  'models/bert_finetuned/bert_config.json')

# macbert
macbert_model_dir = os.path.join(pwd_path, 'models/macbert/')


# 数据集路径
word_freq_path = os.path.join(pwd_path, 'data/word_freq.txt')
common_char_path = os.path.join(pwd_path, 'data/common_char.txt')    # 常用字
same_pinyin_path = os.path.join(pwd_path, 'data/same_pinyin.txt')    # 同音字
same_stroke_path = os.path.join(pwd_path, 'data/same_stroke.txt')    # 形似字
custom_confusion_path = os.path.join(pwd_path, 'data/custom_confusion.txt')    # 混淆集
proper_name_path = os.path.join(pwd_path, 'data/proper_nouns.txt')    # 专有名词
