import gevent.ssl
import numpy as np
import tritonclient.http as httpclient
import sentencepiece as sp
from tqdm import tqdm


def pad_multi_lines(lines_list, pad_symbol):
  max_line_length = max([len(line) for line in lines_list])
  if len(lines_list) * max_line_length % 2 == 1: max_line_length += 1
  lines_list_padded = [line + [pad_symbol] * (max_line_length - len(line)) for line in lines_list]
  return lines_list_padded



def client_init(url="localhost:8000",
                ssl=False, key_file=None, cert_file=None, ca_certs=None, insecure=False,
                verbose=False):
    """

    :param url:
    :param ssl: Enable encrypted link to the server using HTTPS
    :param key_file: File holding client private key
    :param cert_file: File holding client certificate
    :param ca_certs: File holding ca certificate
    :param insecure: Use no peer verification in SSL communications. Use with caution
    :param verbose: Enable verbose output
    :return:
    """
    if ssl:
        ssl_options = {}
        if key_file is not None:
            ssl_options['keyfile'] = key_file
        if cert_file is not None:
            ssl_options['certfile'] = cert_file
        if ca_certs is not None:
            ssl_options['ca_certs'] = ca_certs
        ssl_context_factory = None
        if insecure:
            ssl_context_factory = gevent.ssl._create_unverified_context
        triton_client = httpclient.InferenceServerClient(
            url=url,
            verbose=verbose,
            ssl=True,
            ssl_options=ssl_options,
            insecure=insecure,
            ssl_context_factory=ssl_context_factory)
    else:
        triton_client = httpclient.InferenceServerClient(
            url=url, verbose=verbose)

    return triton_client


class TritonReq(object):

    enzh = ('en', 'zh')
    arel = ('ar', 'el', 'en', 'fa', 'hu', 'is', 'pl', 'ru', 'uk', 'zh')  # arelenfahuisplruukzh
    bnen = ('bn', 'en', 'fil', 'hi', 'id', 'lo', 'ms', 'th', 'ur', 'vi', 'zh')  # bnenfilhiidlomsthurvizh
    csda = ('cs', 'da', 'de', 'et', 'en', 'es', 'fi', 'fr', 'ga', 'it', 'nl', 'pt', 'ro', 'sv', 'tr', 'zh')  # csdadeetenesfifrgaitnlptrosvtrzh
    enja = ('en', 'ja', 'ko', 'zh', 'zhyue')  # enjakozhzhyue
    base_path = '/data/mt/hbl/models/mt_server/translation_model'
    spms = {
        'enzh': sp.SentencePieceProcessor(
            model_file=base_path + '/translate_enzh_ctrip/transformer_relative_big_ctrip/Vocab/spm_model_en_zh_32768.model'),
        'arelenfahuisplruukzh': sp.SentencePieceProcessor(
            model_file=base_path + '/translate_arelenfahuisplruukzh_ctrip/transformer_relative_big_ctrip/Vocab/spm_model_ar_el_en_fa_hu_is_pl_ru_uk_zh_32768.model'),
        'bnenfilhiidlomsthurvizh': sp.SentencePieceProcessor(
            model_file=base_path + '/translate_bnenfilhiidlomsthurvizh_ctrip/mT5_base_ctrip/Vocab/spiece.model'),
        'csdadeetenesfifrgaitnlptrosvtrzh': sp.SentencePieceProcessor(
            model_file=base_path + '/translate_csdadeetenesfifrgaitnlptrosvtrzh_ctrip/transformer_relative_big_ctrip/Vocab/spm_model_cs_da_de_en_es_et_fi_fr_ga_it_nl_pt_ro_sv_tr_zh_32768.model'),
        'enjakozhzhyue': sp.SentencePieceProcessor(
            model_file=base_path + '/translate_enjakozhzhyue_ctrip/transformer_relative_big_ctrip/Vocab/spm_model_en_ja_ko_zh_zhyue_32768.model')
    }

    def __init__(self, client):
        self.eos = 2
        self.pad = 0
        self.client = client

    def preprocess(self, inputs, model, fr, to):
        ids = [self.spms[model].encode(text) + [self.eos] for text in inputs]
        return pad_multi_lines(ids, self.pad)

    def postprocess(self, targets, model):
        sents = []
        for i in range(len(targets)):
            sent = targets[i][0]
            sent = [int(j) for j in sent]
            sent = sent[0: sent.index(self.eos)] if self.eos in sent else sent
            sents.append(sent)
        return self.spms[model].decode(sents)

    def infer(self, model, ids, tgt_lang, mT5=False, adapter_id=-1,
          inputs='inputs', target_lang='target_lang',
          task_space='task_space', max_decode_length='max_decode_length',
          output0='greedy_targets', output1='greedy_scores', output2='encode_status',
          request_compression_algorithm=None,
          response_compression_algorithm=None):

        input_arr = []
        outputs = []
        # batch_size=8
        # \batch_sizeM置G件Dmax_batch_sizeinferYY
        # INPUT0AINPUT1为M置G件中DBM称
        bz = len(ids)
        sl = len(ids[0])
        max_len = min(sl * 1.6 + 8, 256)
        input_arr.append(httpclient.InferInput(inputs, [bz,sl], "INT32"))
        if not mT5:
            input_arr.append(httpclient.InferInput(target_lang, [bz, 1], "INT32"))
            input_arr.append(httpclient.InferInput(task_space, [bz, 1], "INT32"))
        input_arr.append(httpclient.InferInput(max_decode_length, [bz, 1], "INT32"))
        # input_arr.append(httpclient.InferInput(decode_length_scale, [bz, 1], "FP16"))

        # Initialize the data

        input_arr[0].set_data_from_numpy(np.array(ids).astype(np.int32), binary_data=False)
        if mT5:
            input_arr[1].set_data_from_numpy(np.array([[256]] * bz).astype(np.int32), binary_data=False)
            # input_arr[2].set_data_from_numpy(np.array([[1.6]] * bz).astype(np.float16), binary_data=True)
        else:
            input_arr[1].set_data_from_numpy(np.array([[tgt_lang]] * bz).astype(np.int32), binary_data=False)
            input_arr[2].set_data_from_numpy(np.array([[adapter_id]] * bz).astype(np.int32), binary_data=False)
            input_arr[3].set_data_from_numpy(np.array([[max_len]]*bz).astype(np.int32), binary_data=False)
            # input_arr[4].set_data_from_numpy(np.array([[1.6]]*bz).astype(np.float16), binary_data=True)

        # OUTPUT0AOUTPUT1为M置G件中DBM称
        outputs.append(httpclient.InferRequestedOutput(output0, binary_data=True))
        outputs.append(httpclient.InferRequestedOutput(output1, binary_data=True))
        # outputs.append(httpclient.InferRequestedOutput(output2, binary_data=True))

        # query_params = {'task_space111': [[-1]]*bz}

        results = self.client.infer(
            model_name=model,
            inputs=input_arr,
            outputs=outputs,
            # query_params=query_params,
            request_compression_algorithm=request_compression_algorithm,
            response_compression_algorithm=response_compression_algorithm, timeout=1000)
        # print(results.as_numpy(output0))
        # print(results.as_numpy(output1))
        # print(results.as_numpy(output2))

        return results.as_numpy(output0)

    def get_model(self, fr, to):
        if fr in self.enzh and to in self.enzh:
            return 'enzh', self.enzh.index(to)
        elif fr in self.arel and to in self.arel:
            return 'arelenfahuisplruukzh', self.arel.index(to)
        elif fr in self.bnen and to in self.bnen:
            return 'bnenfilhiidlomsthurvizh', self.bnen.index(to)
        elif fr in self.csda and to in self.csda:
            return 'csdadeetenesfifrgaitnlptrosvtrzh', self.csda.index(to)
        elif fr in self.enja and to in self.enja:
            return 'enjakozhzhyue', self.enja.index(to)

    def inference(self, arr, fr, to, adapter_id=-1):
        model, tgt_lang_idx = self.get_model(fr, to)
        ids = self.preprocess(arr, model, fr, to)
        mT5 = True if model == 'bnenfilhiidlomsthurvizh' else False
        targets = self.infer(model, ids, tgt_lang_idx, mT5, adapter_id)
        return self.postprocess(targets, model)


class MT5Req(TritonReq):

    lang_dict = {'en': 'English', 'ko': 'Korean', 'ja': 'Japanese', 'zh': 'Chinese',
         'bn': 'Bengali', 'fil': 'Filipino', 'hi': 'Hindi', 'id': 'Indonesian', 'lo': 'Lao', 'ms': 'Malay', 'th': 'Thai', 'ur': 'Urdu', 'vi': 'Vietnamese'}

    def __init__(self, host):
        super().__init__(host)
        self.eos = 1

    def preprocess(self, inputs, model, fr, to):
        lang1, lang2 = self.lang_dict[fr], self.lang_dict[to]
        prefix = "translate {} to {}: ".format(lang1, lang2)
        arr = [prefix + text for text in inputs]
        return super().preprocess(arr, model, fr, to)


def warmup(client):
    tri = TritonReq(client)
    # print(tri.inference(['你好', '今天天气不错', '酒店很近'], 'zh', 'en'))
    # print(tri.inference(['你好', '今天天气不错', '酒店很近'], 'zh', 'ja'))
    # print(tri.inference(['你好', '今天天气不错',  '酒店很近'], 'zh', 'ar'))
    # print(tri.inference(['你好', '今天天气不错', '酒店很近'], 'zh', 'en'))
    # tri.inference(['你好', '今天天气不错', '酒店很近'], 'zh', 'en')
    # tri.inference(['你好', '今天天气不错', '酒店很近'], 'zh', 'en')
    # print(tri.inference(['你好', '今天天气不错', '酒店很近'], 'zh', 'cs'))
    #
    tri2 = MT5Req(client)
    # print(tri2.inference(['今天天气不错', '你好', '酒店很近'], 'zh', 'th'))

    print("=====================warmup finish")
    return tri, tri2

def partition(ls, size):
    return [ls[i:i + size] for i in range(0, len(ls), size)]


def mt_txt(service, file, fr, to):
    res = []
    with open(file, encoding='utf-8') as f:
        data = [line.strip() for line in f]
        parts = partition(data[:2000], 64)
        for part in tqdm(parts, fr+to):
            res += service.inference(part, fr, to)

        for s, t in zip(data, res):
            if '<unk>' in t:
                print(f'{s}\t\t{t}')

    with open(file + '.' + to, 'w', encoding='utf-8') as f:
        for line in res:
            f.write(line.replace('<unk>', '') + '\n')

if __name__ == '__main__':
    # eos = 2
    # pad = 0
    # model_name = 'enzh'
    # langs = ['en', 'zh']
    # client = client_init('10.5.210.91:8000')
    # texts = ['今天天气不错', '你好', '酒店很近']
    # # texts = ['酒店位于林芝大峡谷景区内,需购买景区门票进入']*2
    # ids = [ spms[model_name].encode(text) + [eos] for text in texts]
    # pad_ids = pad_multi_lines(ids, pad)
    #
    # target_ids = infer(client, model_name, pad_ids, tgt_lang=langs.index('en'))
    #
    # sents = []

    # client = client_init('10.21.35.77:8080')
    client = client_init('triton.uat.qa.nt.ctripcorp.com')
    # client = client_init('10.97.6.219:8080')
    general_ser, mT5_ser = warmup(client)
    base_dir = '../data/triton_test/'

    text = 'The hotel is located in the scenic area. Guests need to purchase scenic tickets to enter. The hotel can help purchase discount tickets for guests. Please contact the hotel for details at 18157389131.'
    text = '开始处理, ProcessStatus=P'
    print(general_ser.inference([text], 'zh', 'en', 0))



    # mt_txt(general_ser, base_dir + 'cchat.ja', 'ja', 'zh')
    # mt_txt(general_ser, base_dir + 'comment.en', 'en', 'zh')
    # mt_txt(general_ser, base_dir + 'corpus.ru', 'ru', 'zh')
    # mt_txt(general_ser, base_dir + 'mono.ko', 'ko', 'zh')
    # print(mT5_ser.inference([
    #     'ที่พักถือว่าโอเค​ ดูสะอาด​ ภายในห้องพักไม่ได้ใหม่แต่ไม่แย่​ เปนฟีลแบบห้องพักสมัยก่อนหน่อยๆ​โดยรวมโอเคไม่วุ่นวายดี แต่วันที่ไปพายุเข้าฝนตกหนักตลอดคืน เลยไม่ได้​ใช้บริการสระว่ายน้ำ​ พนักงาน​ก็บริการดี',
    #     'เดินทางสะดวก ใจกลางเมือง หาของกินง่าย เงียบสงบ ราคาไม่แพง',
    #     'เดินทางสะดวกมากจากสนามบิน ห้องพักมีหลายแบบให้เลือก เงียบสงบ สระว่ายน้ำด้านบนใหญ่ดี วิวสวยพนักงานน่ารักมากๆบริการดี เราเลยพักยาวเลย']
    #                             , 'th', 'zh'))
    # mt_txt(mT5_ser, base_dir + 'comment.th', 'th', 'zh')




    # print(tri.inference(['今天天气不错', '你好', '酒店很近'], 'zh', 'en'))
    # print(tri.inference(['今天天气不错', '你好', '酒店很近'], 'zh', 'en'))
    # print(tri.inference(['今天天气不错', '你好', '酒店很近'], 'zh', 'en'))





    # tri = TritonReq(client)
    # print(tri.inference(['今天天气不错', '你好', '酒店很近'], 'zh', 'ja'))

    # tri2 = MT5Req(client)
    # print(tri2.inference(['今天天气不错', '你好', '酒店很近', '今天天气不错', '你好', '酒店很近'], 'zh', 'th'))



