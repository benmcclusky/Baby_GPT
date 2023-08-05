# Baby_GPT
Miniature GPT (10.7m parameters) optimised for training on CPU's based on Andrej Karpathy's [NanoGPT](https://github.com/karpathy/nanoGPT)

Currently trains on a 1m token dataset of Shakespere text (see sample below)

![image](https://github.com/benmcclusky/Baby_GPT/assets/121236905/92c24a42-beb6-46f2-8d80-09fdd272aed1)

Easily trained on your average CPU however at a laughable MFU (0.05%). Takes <5 minutes to train with current hyperparameters

Able to generate samples of text using the generate.py script (see sample below). Results are pretty nonsensical but the model captures the basic structure of the text and some simple words.

![image](https://github.com/benmcclusky/Baby_GPT/assets/121236905/0ba2ab09-db56-4d10-8232-9bf7932e7f65)
