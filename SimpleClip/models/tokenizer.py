import ftfy
import html
import re

import string

from tokenizers import Tokenizer as HFTokenizer

import torch


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = text.strip()

    return text


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def canonicalize_text(text,
                      *,
                      keep_punctuation_exact_string=None,
                      trans_punctuation=str.maketrans("", "",
                                                      string.punctuation)):
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation)
            for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    text = text.strip()

    return text


class Qwen35Tokenizer:

    def __init__(
            self,
            tokenizer_file='/root/code/SimpleClip_pytorch_training_examples/SimpleClip/models/Qwen3.5_tokenizer.json',
            context_length=96):

        self.tokenizer = HFTokenizer.from_file(tokenizer_file)
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

        # pad必须与eos不同: text_global_pool用
        # (text != pad_id).sum(-1) - 1 定位eos, 两者相同会把整句算成长度0。
        # <|vision_pad|>是词表里真正未被文本使用的占位符;
        # 原实现的pad_token_id=0实际是内容token '!'。
        self.pad_token_id = self.tokenizer.token_to_id('<|vision_pad|>')
        self.eos_token_id = self.tokenizer.token_to_id('<|endoftext|>')

        # 含added_tokens的完整词表上界, 供"模型vocab_size >= tokenizer上界"断言使用
        self.vocab_size = self.tokenizer.get_vocab_size(True)
        self.context_length = context_length

        # caption正文里若出现added_token的字面串(如'<|endoftext|>'、'<think>'),
        # tokenizer会把它解析成对应的特殊id, 这会同时破坏两处逻辑:
        # 1. build_key_padding_mask把正文位置误当PAD屏蔽;
        # 2. text_global_pool用(text != pad_id).sum(-1) - 1定位EOS时算错位置。
        # 注意encode(add_special_tokens=False)并不能抑制added_token解析
        # (它只控制post_processor), 因此必须在文本清洗阶段剥离字面串。
        # 按长度降序排列, 保证正则优先匹配更长的token。
        added_token_list = sorted([
            per_added_token.content for per_added_token in
            self.tokenizer.get_added_tokens_decoder().values()
        ],
                                  key=len,
                                  reverse=True)
        self.added_token_pattern = re.compile('|'.join(
            re.escape(per_added_token)
            for per_added_token in added_token_list))

        print(f'context_length: {self.context_length}, '
              f'vocab_size: {self.vocab_size}, '
              f'pad_token_id: {self.pad_token_id}, '
              f'eos_token_id: {self.eos_token_id}')

    def strip_added_tokens(self, text):
        text = self.added_token_pattern.sub(' ', text)

        return text

    def __call__(self, texts, context_length=None):
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length'

        texts = [
            whitespace_clean(self.strip_added_tokens(basic_clean(text)))
            for text in texts
        ]

        all_ids = []
        for text in texts:
            encoding = self.tokenizer.encode(text)
            ids = encoding.ids

            # truncate if needed (reserve 1 slot for eos token)
            if len(ids) > context_length - 1:
                ids = ids[:context_length - 1]

            assert self.pad_token_id not in ids, f'pad_token_id leaked into caption body: {text[:80]}'
            # append eos token
            ids = ids + [self.eos_token_id]
            # pad to context_length
            ids = ids + [self.pad_token_id] * (context_length - len(ids))
            all_ids.append(ids)

        tokens = torch.tensor(all_ids, dtype=torch.long)

        return tokens
