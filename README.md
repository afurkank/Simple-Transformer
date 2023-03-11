# Transformer
This is a very simplistic and from scratch implementation of a transformer model based on the paper "Attention Is All You Need".

It is implemented for machine translation tasks. Consisting of 6 encoder and decoder layers(this can be changed by playing with hyperparameters), the model takes input as only the source and target for training. Therefore, creating masks or implementing any sort of embedding/positional encoding is not necessary as they are all done inside the model.

Training uses about 6 hours and around 4.3 GB of VRAM with batch size of 32, sequence length of 64 and number of layers 6, which is reasonable as it was implemented for understanding the model rather than creating a highly optimized but complex model.

I hope you find it helpful for understanding how transformers work, at least for machine translation.
