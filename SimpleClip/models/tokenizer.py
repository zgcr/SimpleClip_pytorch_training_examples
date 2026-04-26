import os
import ftfy
import gzip
import html

import numpy as np
import nltk
import random
import string
import regex

from functools import partial
import sentencepiece as spm
from tokenizers import Tokenizer as HFTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_nltk_init = False


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"),
                    ord("~") + 1)) + list(range(
                        ord("¡"),
                        ord("¬") + 1)) + list(range(ord("®"),
                                                    ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]

    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = text.strip()

    return text


def whitespace_clean(text):
    text = " ".join(text.split())
    text = text.strip()

    return text


def canonicalize_text(text,
                      *,
                      keep_punctuation_exact_string=None,
                      trans_punctuation=str.maketrans("", "",
                                                      string.punctuation)):
    """
    Returns canonicalized `text` (lowercase and punctuation removed).

    From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94

    Args:
      text: string to be canonicalized.
      keep_punctuation_exact_string: If provided, then this exact string kept.
        For example providing '{}' will keep any occurrences of '{}' (but will
        still remove '{' and '}' that appear separately).
    """
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


def _clean_canonicalize(x):
    # basic, remove whitespace, remove punctuation, lower case
    return canonicalize_text(basic_clean(x))


def _clean_lower(x):
    # basic, remove whitespace, lower case
    return whitespace_clean(basic_clean(x)).lower()


def _clean_whitespace(x):
    # basic, remove whitespace
    return whitespace_clean(basic_clean(x))


def get_clean_fn(type: str):
    if type == 'canonicalize':
        return _clean_canonicalize
    elif type == 'lower':
        return _clean_lower
    elif type == 'whitespace':
        return _clean_whitespace
    else:
        assert False, f"Invalid clean function ({type})."


def random_mask_tokenize(texts,
                         context_length,
                         sot_token_id,
                         eot_token_id,
                         encode_fn,
                         shuffle=False):
    all_tokens = [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        tokens = torch.tensor(tokens)
        num_tokens = len(tokens)
        # 2 for sot and eot token
        if num_tokens > context_length - 2:
            num_keep = context_length - 2
            indices = torch.randperm(len(tokens))
            indices = indices[:num_keep]
            if not shuffle:
                indices = torch.sort(indices).values
            tokens = tokens[indices]
            num_tokens = num_keep
        result[i, 0] = sot_token_id
        result[i, 1:num_tokens + 1] = tokens
        result[i, num_tokens + 1] = eot_token_id

    return result


def simple_mask_tokenize(texts, context_length, sot_token_id, eot_token_id,
                         encode_fn):
    all_tokens = [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        num_tokens = len(tokens)
        # 2 for sot and eot token
        if num_tokens > context_length - 2:
            num_keep = context_length - 2
            # high is incl
            start_index = random.randint(0, num_tokens - num_keep)
            tokens = list(tokens[start_index:start_index + num_keep])
        tokens = [sot_token_id] + tokens + [eot_token_id]
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def syntax_mask_tokenize(texts, context_length, sot_token_id, eot_token_id,
                         encode_fn):
    """ Returns the tokenized representation of given input string(s).
    Apply syntax masking before tokenize.
    """
    global _nltk_init
    if not _nltk_init:
        # run them for the first time
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')
        _nltk_init = True

    def get_order(x):
        if x.startswith('NN'):
            return 1
        elif x.startswith('JJ'):
            return 2
        elif x.startswith('VB'):
            return 3
        else:
            return 4

    # syntax masking
    new_texts = []
    for text in texts:
        list_tokens = nltk.tokenize.word_tokenize(text)
        pos_tags = nltk.pos_tag(list_tokens)
        #  sample the words by get_order method
        order_list = [get_order(tag) for _, tag in pos_tags]
        sorted_ids = np.argsort(np.array(order_list))
        # need 2 slots for sot and eot tokens
        sampled_ids = sorted(sorted_ids[:context_length - 2])
        # sample the tokens
        sampled_tokens = np.take(np.array(list_tokens), sampled_ids, axis=0)

        new_text = ''
        for token in sampled_tokens:
            new_text = new_text + str(token) + ' '
        new_text = new_text.strip()
        new_texts.append(new_text)
    texts = new_texts

    all_tokens = [[sot_token_id] + encode_fn(text) + [eot_token_id]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        # still need first truncate because some words produces two tokens
        if len(tokens) > context_length:
            # Truncate
            tokens = tokens[:context_length]
            tokens[-1] = eot_token_id
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def get_reduction_mask_fn(type):
    """ 
    Choose strategy for dropping (masking) tokens to achieve target context length
    """
    assert type in ('simple', 'random', 'shuffle', 'syntax')
    if type == 'simple':
        # randomly select block [start:end]
        return simple_mask_tokenize
    elif type == 'random':
        # randomly drop tokens (keep order)
        return random_mask_tokenize
    elif type == 'shuffle':
        # randomly drop tokens (shuffle order)
        return partial(random_mask_tokenize, shuffle=True)
    elif type == 'syntax':
        # randomly drop prioritized by syntax
        return syntax_mask_tokenize


class SimpleTokenizer:

    def __init__(
            self,
            bpe_path='/root/code/SimpleClip_pytorch_training_examples/SimpleClip/models/bpe_simple_vocab_16e6.txt.gz',
            context_length=77,  # default context length for OpenAI CLIP
            additional_special_tokens=None,
            clean='lower',
            reduction_mask=''):
        '''
        bpe_simple_vocab_16e6.txt.gz文件是一个使用字节对编码(Byte Pair Encoding,BPE)算法生成的词汇表文件.
        文件内的单元是基于统计学上的频次自动生成的子词单元,这些单元不是单个字符,而是基于统计学上的频次自动生成的子词.
        bpe和16e6表示该词汇表是使用BPE算法在包含约16百万单词的语料库训练得到.
        bpe算法是当前最常见tokenizer的编码方法,用于GPT(OpenAI)和Bert(Google).
        '''

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]

        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))

        special_tokens = ['<start_of_text>', '<end_of_text>']
        if additional_special_tokens:
            special_tokens += additional_special_tokens
        vocab.extend(special_tokens)

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}

        special = "|".join(special_tokens)
        self.pat = regex.compile(
            special +
            r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex.IGNORECASE)

        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token_id = self.all_special_ids[0]
        self.eot_token_id = self.all_special_ids[1]
        self.context_length = context_length

        self.clean_fn = get_clean_fn(clean)
        self.reduction_fn = get_reduction_mask_fn(
            reduction_mask) if reduction_mask else None

        print(f'vocab_size: {self.vocab_size}')
        print(f'context_length: {self.context_length}')

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>', )
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[
                        i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word

        return word

    def encode(self, text):
        bpe_tokens = []
        text = self.clean_fn(text)
        for token in regex.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b]
                            for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token]
                              for bpe_token in self.bpe(token).split(' '))

        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text
                          ]).decode('utf-8',
                                    errors="replace").replace('</w>', ' ')

        return text

    def __call__(self, texts, context_length=None):
        """ 
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts: An input string or a list of input strings to tokenize
        context_length: The context length to use; all CLIP models use 77 as the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length'

        if self.reduction_fn is not None:
            # use reduction strategy for tokenize if set, otherwise default to truncation below
            return self.reduction_fn(texts,
                                     context_length=context_length,
                                     sot_token_id=self.sot_token_id,
                                     eot_token_id=self.eot_token_id,
                                     encode_fn=self.encode)

        all_tokens = [[self.sot_token_id] + self.encode(text) +
                      [self.eot_token_id] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                # Truncate
                tokens = tokens[:context_length]
                tokens[-1] = self.eot_token_id
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result


class SigLipTokenizer:
    """
    SentencePiece tokenizer wrapper for SigLIP T5 compatible sentencepiece vocabs.
    Uses sentencepiece library directly to avoid compatibility issues with
    transformers versions that changed GemmaTokenizerFast/T5TokenizerFast APIs.
    """

    def __init__(self, tokenizer_name, context_length=64):

        assert tokenizer_name in ["c4-en", "mc4", "gemma"]

        self.VOCAB_FILES = {
            # english, vocab_size=32000
            "c4-en":
            "/root/code/SimpleClip_pytorch_training_examples/SimpleClip/models/c4-en-sentencepiece.model",
            # used in multilingual models (mT5, PaLI), vocab_size=250000
            "mc4":
            "/root/code/SimpleClip_pytorch_training_examples/SimpleClip/models/mc4-sentencepiece.model",
            # used in SigLIP2 models, vocab_size=256000
            "gemma":
            "/root/code/SimpleClip_pytorch_training_examples/SimpleClip/models/gemma_tokenizer.model",
        }

        vocab_file = self.VOCAB_FILES[tokenizer_name]
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

        self.pad_token_id = 0
        self.eos_token_id = 1
        self.context_length = context_length

        print(f'context_length: {self.context_length}')

    def __call__(self, texts, context_length=None):
        # canonicalize text: basic clean, remove whitespace, remove punctuation, lower case
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length'

        texts = [canonicalize_text(basic_clean(text)) for text in texts]

        all_ids = []
        for text in texts:
            ids = self.sp_model.Encode(text)
            # truncate if needed (reserve 1 slot for eos token)
            if len(ids) > context_length - 1:
                ids = ids[:context_length - 1]
            # append eos token
            ids = ids + [self.eos_token_id]
            # pad to max_length
            ids = ids + [self.pad_token_id] * (context_length - len(ids))
            all_ids.append(ids)

        tokens = torch.tensor(all_ids, dtype=torch.long)

        return tokens


class Qwen35Tokenizer:
    """
    BPE tokenizer wrapper for Qwen3.5 models.
    Uses HuggingFace tokenizers library directly to load tokenizer.json file,
    avoiding compatibility issues with transformers versions.
    Qwen3.5 uses byte-level BPE tokenizer with vocab_size=248058.
    """

    def __init__(
            self,
            tokenizer_file='/root/code/SimpleClip_pytorch_training_examples/SimpleClip/models/Qwen3.5_tokenizer.json',
            context_length=64):

        self.tokenizer = HFTokenizer.from_file(tokenizer_file)
        # disable any built-in truncation and padding from tokenizer.json
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

        # Qwen3.5 special token ids:
        # <|endoftext|>: 248044, <|im_start|>: 248045, <|im_end|>: 248046
        self.pad_token_id = 0
        self.eos_token_id = 248044  # <|endoftext|>
        self.context_length = context_length

        print(f'context_length: {self.context_length}')

    def __call__(self, texts, context_length=None):
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length'

        texts = [canonicalize_text(basic_clean(text)) for text in texts]

        all_ids = []
        for text in texts:
            encoding = self.tokenizer.encode(text)
            ids = encoding.ids
            # truncate if needed (reserve 1 slot for eos token)
            if len(ids) > context_length - 1:
                ids = ids[:context_length - 1]
            # append eos token
            ids = ids + [self.eos_token_id]
            # pad to context_length
            ids = ids + [self.pad_token_id] * (context_length - len(ids))
            all_ids.append(ids)

        tokens = torch.tensor(all_ids, dtype=torch.long)

        return tokens


class DeepSeekV4Tokenizer:
    """
    BPE tokenizer wrapper for DeepSeek-V4 models.
    Uses HuggingFace tokenizers library directly to load tokenizer.json file,
    avoiding compatibility issues with transformers versions.
    DeepSeek-V4 uses byte-level BPE tokenizer with 128000 base vocab tokens
    plus special tokens.

    NOTE: eos_token_id=1 is very small compared to regular BPE token ids
    (range ~3-128000+), so this tokenizer is NOT compatible with
    pool_type='argmax' (used by CLIP models). It should only be used with
    pool_type='last' (used by SigLIP models), where the pooling position
    is determined by (text != pad_id).sum() - 1.
    When using this tokenizer, make sure to set text_pad_id=2 in the model.
    """

    def __init__(
            self,
            tokenizer_file='/root/code/SimpleClip_pytorch_training_examples/SimpleClip/models/DeepSeek-V4_tokenizer.json',
            context_length=64):

        self.tokenizer = HFTokenizer.from_file(tokenizer_file)
        # disable any built-in truncation and padding from tokenizer.json
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

        # DeepSeek-V4 special token ids:
        # <｜begin▁of▁sentence｜>: 0, <｜end▁of▁sentence｜>: 1, <｜▁pad▁｜>: 2
        self.pad_token_id = 2  # <｜▁pad▁｜>
        self.eos_token_id = 1  # <｜end▁of▁sentence｜>
        self.context_length = context_length

        print(f'context_length: {self.context_length}')

    def __call__(self, texts, context_length=None):
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length'

        texts = [canonicalize_text(basic_clean(text)) for text in texts]

        all_ids = []
        for text in texts:
            encoding = self.tokenizer.encode(text)
            ids = encoding.ids
            # truncate if needed (reserve 1 slot for eos token)
            if len(ids) > context_length - 1:
                ids = ids[:context_length - 1]
            # append eos token
            ids = ids + [self.eos_token_id]
            # pad to context_length
            ids = ids + [self.pad_token_id] * (context_length - len(ids))
            all_ids.append(ids)

        tokens = torch.tensor(all_ids, dtype=torch.long)

        return tokens
