#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/12/27 17:42
# @Author  : hbl
# @File    : triton_ft_client.py
# @Desc    :

from triton_client_mt import client_init, MT5Req, pad_multi_lines, partition
import gevent.ssl
import numpy as np
import tritonclient.http as httpclient
import sentencepiece as sp
from utils import elapsed_timer, read_paral_data
from tqdm import tqdm
import sys
#sys.path.append('/opt/app')
#from mttool.metric import metric

lang2short = {'English': 'en','Korean': 'ko', 'Japanese': 'ja', 'Chinese': 'zh', 'Bengali': 'bn', 'Filipino': 'fil',
              'Hindi': 'hi', 'Indonesian': 'id', 'Lao': 'lo', 'Malay': 'ms', 'Thai': 'th', 'Urdu': 'ur', 'Vietnamese': 'vi',
             'French': 'fr', 'Spanish': 'es', 'Italian': 'it', 'German': 'de'}

#bleu_dict = {}
#for val in lang2short.values():
#    bleu_dict[val] = metric.Bleu(val)

def bleu_(preds, labels, lang):
    total = 0.0
    d = {}
    for p, g,  in zip(preds, labels):
        try:
            if lang not in d:
                d[lang] = ([p], [g])
            else:
                d[lang][0].append(p)
                d[lang][1].append(g)
        except:
            print("===============================")

    for lang in d:
        total += bleu_dict[lang].multi_evaluate_with_preprocess(d[lang][0], d[lang][1])
    bleu = total / len(d)
    return bleu


class Mt5Ft(MT5Req):

    def __init__(self, host):
        super().__init__(host)

    def preprocess(self, inputs, model, fr, to):
        lang1, lang2 = self.lang_dict[fr], self.lang_dict[to]
        prefix = "translate {} to {}: ".format(lang1, lang2)
        arr = [prefix + text for text in inputs]

        ids = [self.spms[model].encode(text) + [self.eos] for text in arr]
        return ids

    def infer(self, model, ids,
            inputs='input_ids', sequence_length='sequence_length', max_output_len='max_output_len',
            output0='output_ids', output1='sequence_length',
            request_compression_algorithm=None,
            response_compression_algorithm=None):
        input_arr = []
        outputs = []
        bz = len(ids)
        seq_lens = [[len(arr)] for arr in ids]
        ids = pad_multi_lines(ids, self.pad)
        sl = len(ids[0])
        max_len = min(sl * 1.6 + 8, 256)
        outputs.append(httpclient.InferRequestedOutput(output0, binary_data=True))
        outputs.append(httpclient.InferRequestedOutput(output1, binary_data=True))
        input_arr.append(httpclient.InferInput("input_ids", [bz, sl], "UINT32"))
        input_arr.append(httpclient.InferInput("sequence_length", [bz, 1], "UINT32"))
        input_arr.append(httpclient.InferInput("max_output_len", [bz, 1], "UINT32"))

        input_arr[0].set_data_from_numpy(np.array(ids).astype(np.uint32), binary_data=False)
        # input_arr[0].set_data_from_numpy(np.array([[1, 1, 1, 1]]).astype(np.uint32), binary_data=False)
        input_arr[1].set_data_from_numpy(np.array(seq_lens).astype(np.uint32), binary_data=False)
        input_arr[2].set_data_from_numpy(np.array([[max_len]]*len(seq_lens)).astype(np.uint32), binary_data=False)

        results = client.infer(
            model_name=model,
            inputs=input_arr,
            outputs=outputs,
            # query_params=query_params,
            request_compression_algorithm=False,
            response_compression_algorithm=False, timeout=1000)

        output_ids = results.as_numpy(output0)
        decoder_lens = results.as_numpy(output1)
        return output_ids



if __name__ == '__main__':

    client = client_init("10.5.210.91:5780")


    cli = Mt5Ft(client)

    print(cli.inference(['สวัสดีครับ', 'ที่ตั้งโรงแรมดี'], 'th', 'zh'))
    dataset = read_paral_data('/data/bak/evaluation/bn_en_fil_hi_id_lo_ms_th_ur_vi_zh/eval/data/TestEval/TripAdvisor/comment/en_vi/origin',
                              'ta_comment', ['en', 'vi'])
    fr, to = 'vi', 'en'
    char_count = 0
    for line in dataset['vi']:
        char_count += len(line)

    for batch in [1,2,4,8]:
        if batch == 1:
            data = [[text]  for text in dataset[fr]]
        else:
            data = partition(dataset[fr], batch)

        rets = []
        with elapsed_timer() as elapsed:
            for par in tqdm(data):
                rets+= cli.inference(par, 'vi', 'en')
                elap = elapsed()
 #           score = bleu_(rets, dataset[to], to)
            score = None
            print('batch: {}, cps:{}, elapsed: {}s bleu: {}'.format(batch, char_count / elap,
                                                                                 '%.6f' % elap,score))


# print(cli.inference(['你好'], 'zh', 'th'))


# bz = 1
# sl = 4
#
# output0 = 'output_ids'
# output1 = 'sequence_length'
#
# base_path = '/data/mt/hbl/models/mt_server/translation_model'
#
# spm = sp.SentencePieceProcessor(
#             model_file=base_path + '/translate_bnenfilhiidlomsthurvizh_ctrip/mT5_base_ctrip/Vocab/spiece.model'),
#
#
#
#
#


