# Long Convs

This folder shows an example of adapting a Long Conv backbone, as presented in [Simple Hardware-Efficient Long Convolutions for Sequence Modeling](https://arxiv.org/abs/2302.06646), to use FlashFFTConv.
These files are sourced from the standalone example in the main repo.

## Changes to Use FlashFFTConv in Long Convs

We describe the changes necessary to use FlashFFTConv in M2-BERT:

Create an instance of `FlashFFTConv` in `LongConvModel`. In [flashfftconv_long_convs.py](flashfftconv_long_convs.py), lines 113-122:
```Python
self.flashfftconv = FlashFFTConv(2048)

...

for _ in self.layers:
    layer = LongConv(d_model, L=1024, dropout=dropout, **conv_kwargs)
    layer.flashfftconv = self.flashfftconv
    self.conv_layers.append(layer)
```

Note that we set FFT size to twice the sequence length, to create a bidirectional convolution.