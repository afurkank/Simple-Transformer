"""
Ahmet Furkan Karacık
afurkank @ github

You need to install torch, tokenizers, torchtext and torchdata.
Some versions of torchtext and torchdata may not be compatible
with certain versions of torch, so please check if your
torchtext and torchdata are supported for your torch version 
from these links:
https://github.com/pytorch/text/blob/main/README.rst
https://github.com/pytorch/data/blob/main/README.md

In 0.6.0 version of torchdata, there is an issue with the 
DataLoader. It may still be a problem in the future,
so I suggest you use the following configuration to run this script:

torchdata v0.5.1
torchtext v0.14.1
torch     v1.13.1

you can download them like this:

pip install torch==1.13.1 torchtext==0.14.1 torchdata==0.5.1
pip install tokenizers

If you are going to use newer versions, you might need to install
portalocker like this:

pip install 'portalocker>=2.0.0'
"""

from torchtext.datasets import Multi30k
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

MAX_LEN = 64
VOCAB_SIZE = 32768
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

f = open("parallelcorpus.txt", "a")
for i in train_iter:
    for x in [x.rstrip("\n") for x in i]:
        f.write(x)
        f.write(' ')
f.close()

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(["parallelcorpus.txt"], trainer)

tokenizer.enable_padding(pad_id=PAD_IDX, length=MAX_LEN)
tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    special_tokens=[
        ("[BOS]", tokenizer.token_to_id("[BOS]")),
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
    ],
)

tokenizer.save("tokenizer")
"""
To load the trained tokenizer, use:

tokenizer = Tokenizer.from_file(tokenizer_path)

inside your model.

Be careful about the path, it needs to be a raw string.

i.e. tokenizer_path = r"C:\Users\furkan\Transformer\trained_tokenizer"

"""
