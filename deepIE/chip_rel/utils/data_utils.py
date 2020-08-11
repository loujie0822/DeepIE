import re

from transformers import BertTokenizer

from utils import extract_chinese_and_punct

chineseandpunctuationextractor = extract_chinese_and_punct.ChineseAndPunctuationExtractor()
moren_tokenizer = BertTokenizer.from_pretrained('transformer_cpt/bert/', do_lower_case=True)

def covert_to_tokens(text, tokenizer=None, return_orig_index=False, max_seq_length=500):
    if not tokenizer:
        tokenizer =moren_tokenizer
    sub_text = []
    buff = ""
    flag_en = False
    flag_digit = False
    for char in text:
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
            flag_en = False
            flag_digit = False
        else:
            if re.compile('\d').match(char):
                if buff != "" and flag_en:
                    sub_text.append(buff)
                    buff = ""
                    flag_en = False
                flag_digit = True
                buff += char
            else:
                if buff != "" and flag_digit:
                    sub_text.append(buff)
                    buff = ""
                    flag_digit = False
                flag_en = True
                buff += char
    if buff != "":
        sub_text.append(buff)

    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    tokens = []
    text_tmp = ''
    for (i, token) in enumerate(sub_text):
        sub_tokens = tokenizer.tokenize(token) if token != ' ' else []
        text_tmp += token
        for sub_token in sub_tokens:
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
            if len(tokens) >= max_seq_length - 2:
                break
        else:
            continue
        break
    if return_orig_index:
        return tokens, tok_to_orig_start_index, tok_to_orig_end_index
    else:
        return tokens


def search_spo_index(tokens, subject_sub_tokens, object_sub_tokens):
    subject_start_index, object_start_index = -1, -1
    forbidden_index = None
    if len(subject_sub_tokens) > len(object_sub_tokens):
        for index in range(
                len(tokens) - len(subject_sub_tokens) + 1):
            if tokens[index:index + len(
                    subject_sub_tokens)] == subject_sub_tokens:
                subject_start_index = index
                forbidden_index = index
                break

        for index in range(
                len(tokens) - len(object_sub_tokens) + 1):
            if tokens[index:index + len(
                    object_sub_tokens)] == object_sub_tokens:
                if forbidden_index is None:
                    object_start_index = index
                    break
                # check if labeled already
                elif index < forbidden_index or index >= forbidden_index + len(
                        subject_sub_tokens):
                    object_start_index = index

                    break

    else:
        for index in range(
                len(tokens) - len(object_sub_tokens) + 1):
            if tokens[index:index + len(
                    object_sub_tokens)] == object_sub_tokens:
                object_start_index = index
                forbidden_index = index
                break

        for index in range(
                len(tokens) - len(subject_sub_tokens) + 1):
            if tokens[index:index + len(
                    subject_sub_tokens)] == subject_sub_tokens:
                if forbidden_index is None:
                    subject_start_index = index
                    break
                elif index < forbidden_index or index >= forbidden_index + len(
                        object_sub_tokens):
                    subject_start_index = index
                    break

    return subject_start_index, object_start_index
