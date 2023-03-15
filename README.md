# Transformer
This is a very simple and from scratch implementation of a transformer model based on the paper "Attention Is All You Need".

It is implemented for machine translation tasks. The model takes only the tokenized source and target as inputs for training. Therefore, creating masks or implementing any sort of embedding/positional encoding is not necessary as they are all done inside the model. However, you can still inspect how masks are created and embeddings/positional encodings are done.

I hope you find it helpful for understanding how transformers work, at least for a machine translation task.

# Training

Training uses around 4 GB of VRAM with batch size of 32, sequence length of 64 and number of layers 3, which is reasonable as it was implemented for understanding the model rather than creating a highly optimized but complex model.

Some observations about training:

> Using Adam as optimizer rather than SGD yields much faster convergence and better generalization.

> I haven't tried any learning rate below 0.0001, but when lr is higher than that the model cannot learn as good.

> Increasing the number of layers from 3 to 6 did not result in better generalization. Considering the slower training and more VRAM usage of increased layer number, I do not think it is necessary to go up to 6 layers for my model, at least with as small of a training dataset as I used. However, I did not try 4 or 5 as number of layers; maybe they can give better results.

# Inference

For inference, I used Greedy Decoding as it was the simplest one to implement.

There is a simple example which translates the given German sentence correctly. However, the model produces less accurate outputs as the sentence gets more complicated. I assume training the model with more data can solve this problem. Also, using methods such as Beam Search instead of Greedy Decoding will probably increase the accuracy of the translation as well.
