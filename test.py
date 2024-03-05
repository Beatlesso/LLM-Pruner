from datasets import *
from torch.utils.data.dataset import Dataset

import os
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

# # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
# # testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
# # traindata.save_to_disk('/home/luoyicong/LLM-Pruner/dataset/wikitext/traindata')
# # testdata.save_to_disk('/home/luoyicong/LLM-Pruner/dataset/wikitext/testdata')
# traindata = load_from_disk('/home/luoyicong/LLM-Pruner/dataset/wikitext/traindata')
# testdata = load_from_disk('/home/luoyicong/LLM-Pruner/dataset/wikitext/testdata')

# # traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
# # valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
# # traindata.save_to_disk('/home/luoyicong/LLM-Pruner/dataset/ptb_text/traindata')
# # valdata.save_to_disk('/home/luoyicong/LLM-Pruner/dataset/ptb_text/valdata')
# traindata = load_from_disk('/home/luoyicong/LLM-Pruner/dataset/ptb_text/traindata')
# testdata = load_from_disk('/home/luoyicong/LLM-Pruner/dataset/ptb_text/valdata')

env_name="{}".format("ok")
print(env_name)