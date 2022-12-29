# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
This example is used to verify the correctess on summarization task. So, we don't
put benchmark testing in this example.
'''

from __future__ import print_function
import argparse
import json
import numpy as np
import os
import sys
import torch
import torch.distributed as dist
from datasets import load_dataset, load_metric
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path + "/../../../3rdparty/transformers/src/")

from contextlib import contextmanager
from timeit import default_timer
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from tqdm import tqdm
import configparser
import math
import datetime

os.environ['http_proxy'] = 'http://ntproxy.qa.nt.ctripcorp.com:8080'
os.environ['https_proxy'] = 'http://ntproxy.qa.nt.ctripcorp.com:8080'
# os.environ['LD_LIBRARY_PATH'] = '/opt/conda/lib/python3.8/site-packages/torch/lib:/opt/conda/lib/python3.8/site-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.t5.utils.ft_decoding import FTT5DecodingWeight, FTT5Decoding, FTT5
from examples.pytorch.t5.utils.ft_encoder import FTT5EncoderWeight, FTT5Encoder
sys.path.append('/data/mt/hbl')
from mttool.metric import metric

lang_dict = {'en': 'English', 'ko': 'Korean', 'ja': 'Japanese', 'zh': 'Chinese',
         'bn': 'Bengali', 'fil': 'Filipino', 'hi': 'Hindi', 'id': 'Indonesian', 'lo': 'Lao', 'ms': 'Malay', 'th': 'Thai', 'ur': 'Urdu', 'vi': 'Vietnamese', 'fr': 'French', 'es': 'Spanish', 'it': 'Italian', 'de': 'German'}

lang2short = {'English': 'en','Korean': 'ko', 'Japanese': 'ja', 'Chinese': 'zh', 'Bengali': 'bn', 'Filipino': 'fil',
              'Hindi': 'hi', 'Indonesian': 'id', 'Lao': 'lo', 'Malay': 'ms', 'Thai': 'th', 'Urdu': 'ur', 'Vietnamese': 'vi',
             'French': 'fr', 'Spanish': 'es', 'Italian': 'it', 'German': 'de'}

bleu_dict = {}
for val in lang2short.values():
    bleu_dict[val] = metric.Bleu(val)



def partition(ls, size):
  return [ls[i:i + size] for i in range(0, len(ls), size)]

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def read_paral_data(path, filename, langs):
    data = {}
    with open(os.path.join(path, f'{filename}.{langs[0]}'), encoding='utf-8') as f0:
        data[langs[0]] = [line.strip() for line in f0]

    with open(os.path.join(path, f'{filename}.{langs[1]}'), encoding='utf-8') as f1:
        data[langs[1]] = [line.strip() for line in f1]
    return data


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


def bleu_calc(tokenizer, preds, labels, inputs):
    total = 0.0
    d = {}
    for p, g, i in zip(preds, labels, inputs):
        try:
            lang = tokenizer.decode(i).split(':', 1)[0].split('to ')[1].strip()
            if lang not in d:
                d[lang] = ([p], [g])
            else:
                d[lang][0].append(p)
                d[lang][1].append(g)
        except:
            print("===============================")
            print(tokenizer.decode(i))

    for lang in d:
        total += bleu_dict[lang2short[lang]].multi_evaluate_with_preprocess(d[lang][0], d[lang][1])
    bleu = total / len(d)
    return bleu



def main():
    base_dir = '/data/mt/hbl/FasterTransformer_master'
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_model_location', type=str,
                        default='/models/T5/HF/t5-base/c-models/')
    parser.add_argument('--hf_model_location', type=str,
                        default='/models/T5/HF/t5-base/')
    parser.add_argument('--disable_summarize', action='store_true')
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--test_ft', action='store_true')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument("--cache_path", type=str, default="/workdir/datasets/ccdv/")
    parser.add_argument("--max_ite", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--ft_use_hf_config", action="store_true",
                        help="use the hyper-parameters from the hf model")
    parser.add_argument('--lib_path', type=str, default=os.path.join(base_dir, '/lib/libth_t5.so'),
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--rougeLsum_threshold', type=float,
                        help='Threshold of FT rougeLsum score')

    args = parser.parse_args()

    # if dist.is_mpi_available():
    #     try:
    #         dist.init_process_group(backend='mpi')
    #         rank = dist.get_rank()
    #     except:
    #         rank = dist.get_rank()
    # else:
    #     rank = 0
    rank = 0
    disable_summarize = args.disable_summarize
    test_hf = args.test_hf
    test_ft = args.test_ft

    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    ft_model_location = args.ft_model_location + f"/{tensor_para_size}-gpu/"
    hf_model_location = args.hf_model_location

    tokenizer = AutoTokenizer.from_pretrained(hf_model_location)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = read_paral_data('/data/bak/evaluation/bn_en_fil_hi_id_lo_ms_th_ur_vi_zh/eval/data/TestEval/TripAdvisor/comment/en_vi/origin',
                              'ta_comment', ['en', 'vi'])

    if rank == 0 and test_hf:
        start_time = datetime.datetime.now()
        if args.data_type == "fp32":
            model = T5ForConditionalGeneration.from_pretrained(hf_model_location, torch_dtype=torch.float32).cuda()
        elif args.data_type == "fp16":
            model = T5ForConditionalGeneration.from_pretrained(hf_model_location, torch_dtype=torch.float16).cuda()
        elif args.data_type == "bf16":
            model = T5ForConditionalGeneration.from_pretrained(hf_model_location, torch_dtype=torch.bfloat16).cuda()
        stop_time = datetime.datetime.now()
        print(f"[INFO] load HF model spend {(stop_time - start_time).total_seconds()} sec")

    if test_ft:
        ckpt_config = configparser.ConfigParser()

        ckpt_config_path = os.path.join(ft_model_location, 'config.ini')
        if os.path.isfile(ckpt_config_path):
            ckpt_config.read(ckpt_config_path)
        else:
            assert False, "[ERROR] This example only support loading model with FT format directly."

        weight_data_type = np.float32
        weight_data_type = {"fp16": np.float16, "fp32": np.float32}[ckpt_config.get("encoder", "weight_data_type")]
        relative_attention_max_distance = 128
        encoder_config = T5Config(vocab_size=ckpt_config.getint("encoder", "vocab_size"),
                                  d_model=ckpt_config.getint("encoder", "d_model"),
                                  d_kv=ckpt_config.getint("encoder", "d_kv"),
                                  d_ff=ckpt_config.getint("encoder", "d_ff"),
                                  num_layers=ckpt_config.getint("encoder", "num_layers"),
                                  num_decoder_layers=ckpt_config.getint("encoder", "num_decoder_layers"),
                                  num_heads=ckpt_config.getint("encoder", "num_heads"),
                                  relative_attention_num_buckets=ckpt_config.getint(
                                      "encoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                  feed_forward_proj=ckpt_config.get("encoder", "feed_forward_proj"),
                                  pad_token_id=ckpt_config.getint("encoder", "pad_token_id"),
                                  eos_token_id=ckpt_config.getint("encoder", "eos_token_id"),
                                  is_gated_act=ckpt_config.getboolean("encoder", "is_gated_act", fallback=0),
                                  )
        decoder_config = T5Config(vocab_size=ckpt_config.getint("decoder", "vocab_size"),
                                  d_model=ckpt_config.getint("decoder", "d_model"),
                                  d_kv=ckpt_config.getint("decoder", "d_kv"),
                                  d_ff=ckpt_config.getint("decoder", "d_ff"),
                                  num_layers=ckpt_config.getint("decoder", "num_layers"),
                                  num_decoder_layers=ckpt_config.getint("decoder", "num_decoder_layers"),
                                  num_heads=ckpt_config.getint("decoder", "num_heads"),
                                  relative_attention_num_buckets=ckpt_config.getint(
                                      "decoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                  feed_forward_proj=ckpt_config.get("decoder", "feed_forward_proj"),
                                  pad_token_id=ckpt_config.getint("decoder", "pad_token_id"),
                                  eos_token_id=ckpt_config.getint("decoder", "eos_token_id"),
                                  decoder_start_token_id=ckpt_config.getint("decoder", "decoder_start_token_id"),
                                  is_gated_act=ckpt_config.getboolean("decoder", "is_gated_act", fallback=0),
                                  )
        assert decoder_config.feed_forward_proj == encoder_config.feed_forward_proj
        assert decoder_config.feed_forward_proj == encoder_config.feed_forward_proj

        t5_with_bias = ckpt_config.getboolean("structure", "t5_with_bias")
        use_gated_activation = encoder_config.is_gated_act
        position_embedding_type = 0 if ckpt_config.get('structure', 'position_embedding_type') == 'relative' else 1
        activation_type = encoder_config.feed_forward_proj

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1660
        # if tie_word_embeddings == True, scale the decoder output by sequence_output = sequence_output * (self.model_dim**-0.5)
        tie_word_embeddings = ckpt_config.getboolean("decoder", "tie_word_embeddings")
        # tie_word_embeddings = True
        ft_encoder_weight = FTT5EncoderWeight(
            encoder_config,
            tensor_para_size,
            pipeline_para_size,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type
        )
        ft_decoding_weight = FTT5DecodingWeight(
            decoder_config,
            tensor_para_size,
            pipeline_para_size,
            t5_with_bias=t5_with_bias,
            use_gated_activation=use_gated_activation,
            position_embedding_type=position_embedding_type,
            weight_data_type=weight_data_type,
        )

        start_time = datetime.datetime.now()
        ft_encoder_weight.load_from_bin(ft_model_location)
        stop_time = datetime.datetime.now()
        print(f"[INFO] load FT encoder model spend {(stop_time - start_time).total_seconds()} sec")
        start_time = datetime.datetime.now()
        ft_decoding_weight.load_from_bin(ft_model_location)
        stop_time = datetime.datetime.now()
        print(f"[INFO] load FT decoding model spend {(stop_time - start_time).total_seconds()} sec")
        if args.data_type == "fp32":
            ft_encoder_weight.to_float()
            ft_decoding_weight.to_float()
        elif args.data_type == "fp16":
            ft_encoder_weight.to_half()
            ft_decoding_weight.to_half()
        elif args.data_type == "bf16":
            ft_encoder_weight.to_bfloat16()
            ft_decoding_weight.to_bfloat16()

        ft_encoder_weight.to_cuda()
        ft_decoding_weight.to_cuda()

        q_scaling = 1.0 / (math.sqrt(encoder_config.d_kv))
        remove_padding = True
        ft_encoder = FTT5Encoder(ft_encoder_weight.w, args.lib_path, encoder_config.num_heads,
                                 encoder_config.d_kv, encoder_config.d_ff,
                                 encoder_config.d_model, remove_padding, encoder_config.num_layers,
                                 encoder_config.relative_attention_num_buckets,
                                 relative_attention_max_distance, False, q_scaling, tensor_para_size,
                                 pipeline_para_size, t5_with_bias,
                                 position_embedding_type, activation_type=activation_type)

        ft_decoding = FTT5Decoding(ft_decoding_weight.w, args.lib_path,
                                   decoder_config.num_heads, decoder_config.d_kv,
                                   decoder_config.d_ff, encoder_config.d_model,
                                   decoder_config.d_model, decoder_config.num_layers,
                                   decoder_config.decoder_start_token_id, decoder_config.eos_token_id,
                                   decoder_config.vocab_size, q_scaling,
                                   decoder_config.relative_attention_num_buckets, max_distance=relative_attention_max_distance,
                                   tensor_para_size=tensor_para_size, pipeline_para_size=pipeline_para_size,
                                   t5_with_bias=t5_with_bias, position_embedding_type=position_embedding_type,
                                   activation_type=activation_type, tie_word_embeddings=tie_word_embeddings)

        ft_t5 = FTT5(ft_encoder, ft_decoding)

    top_k = 1
    output_len = args.max_seq_len


    def translate(dataset, fr, to, char_count, engine='ft', batch=1):
        mt_res = []
        if batch == 1:
            data = dataset[fr]
        else:
            data = partition(dataset[fr], batch)
        with elapsed_timer() as elapsed:
            for d in tqdm(data, 'ft'):
                if batch == 1:
                    line = 'translate {} to {}: '.format(lang_dict[fr], lang_dict[to]) + d
                else:
                    line = ['translate {} to {}: '.format(lang_dict[fr], lang_dict[to]) + text1 for  text1 in d]
                if engine == 'ft':
                    line_tokens = tokenizer(line, return_tensors='pt', padding=True)
                    with torch.no_grad():
                        output, ft_output_len = ft_t5(line_tokens,
                                                      None,
                                                      1,
                                                      output_len+1,
                                                      top_k,
                                                      0.0,
                                                       beam_search_diversity_rate=0.0,
                                                      is_return_output_log_probs=False,
                                                      is_return_cum_log_probs=False)
                    # tokens = output[0][0]
                    for o, l in zip(output, ft_output_len):
                        mt_res.append(tokenizer.decode(o[0][:l[0]]))
                else:
                    line_encoded = tokenizer.encode(line, return_tensors='pt')
                    line_encoded = line_encoded.cuda()

                    with torch.no_grad():
                        output = model.generate(line_encoded,
                                                max_length=output_len + 1,
                                                top_k=top_k,
                                                eos_token_id=tokenizer.eos_token_id,
                                                pad_token_id=tokenizer.pad_token_id)

                    # tokens = output[0].cpu().numpy()
                    mt_res.append(tokenizer.decode(output[0]))
            elap = elapsed()

            assert (len(mt_res)==len(dataset[to]))
            mt_res = [line.replace('<pad>', '').replace('</s>', '').strip() for line in mt_res]
            score = bleu_(mt_res, dataset[to], to)
            # score=0

            print('engine: {}, batch: {}, cps:{}, elapsed: {}s, bleu: {}'.format(engine, batch, char_count/elap,'%.6f' % elap, score))



        # output_lines = ".".join(output_lines.split('.')[:4]) + "."
        # return output_lines, tokens


    # def mt_hf(dataset, fr, to):
    #     mt_res = []
    #     with elapsed_timer() as elapsed:
    #         for text1 in tqdm(dataset[fr], 'hf'):
    #             line = 'translate {} to {}: '.format(lang_dict[fr], lang_dict[to]) + text1
    #
    #             line_encoded = tokenizer.encode(line, return_tensors='pt')
    #             line_encoded = line_encoded.cuda()
    #
    #             with torch.no_grad():
    #                 output = model.generate(line_encoded,
    #                                         max_length=output_len + 1,
    #                                         top_k=top_k,
    #                                         eos_token_id=tokenizer.eos_token_id,
    #                                         pad_token_id=tokenizer.pad_token_id)
    #
    #             # tokens = output[0].cpu().numpy()
    #             mt_res.append(tokenizer.decode(output[0]))
    #         print('耗时：{}s'.format('%.6f' % elapsed()))
    #         assert(len(mt_res), len(dataset[to]))
    #         mt_res = [line.replace('<pad>', '').replace('</s>', '').strip() for line in mt_res]
    #         print("==========bleu=========",bleu_(mt_res, dataset[to], to))
    #     # output_lines = ".".join(output_lines.split('.')[:4]) + "."
    #     # return output_lines, tokens
    char_count = 0
    for line in dataset['vi']:
        char_count += len(line)
    print(f"=============char_count: {char_count}, count:{len(dataset['vi'])}")
    for batch in [1, 2,4,8]:
        translate(dataset, 'vi', 'en', char_count, batch=batch)

    # translate(dataset, 'vi', 'en', 'hugeface')
    # mt_hf(dataset, 'vi', 'en')



if __name__ == '__main__':
    main()
