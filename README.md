# Transformer
This is a very simplistic and from scratch implementation of transformer model based on the paper "Attention Is All You Need".

It is implemented for machine translation tasks. Consisting of 3 encoder and 3 decoder layers(this can be changed by playing with hyperparameters), the model takes input as only the source and target for training. Therefore, creating masks or implementing any sort of embedding/positional encoding is not necessary as they are all done inside the model.

Training uses about 5.4 GB of VRAM with batch size and sequence length of 64, which is reasonable as it was implemented for understanding the model rather than creating a highly optimized but complex model.

I hope you find it helpful for understanding how transformers work, at least for machine translation.
