# Monarch Mixer BERT

This folder shows an example of adapting M2-BERT to use FlashFFTConv.
The original files are sourced from the [M2-BERT](https://github.com/HazyResearch/m2/tree/main/bert/src) implementation.

## Requirements

Install model-specific requirements:
```
pip install -r requirements.txt
```

## Usage

We have sample configs for M2-BERT models of different sizes that you can benchmark:
```
python benchmark_fwd.py configs/m2-110M.yaml
python benchmark_fwd.py configs/m2-110M-flashfftconv.yaml
```

## Changes to Use FlashFFTConv in M2-BERT

We describe the changes necessary to use FlashFFTConv in M2-BERT:

Create an instance of `FlashFFTConv` in `BERTEncoder`. In [bert_layers.py](bert_layers.py), lines 294-301:
```Python
seqlen = config.max_position_embeddings
if config.use_flashfftconv:
    self.flashfftconv = FlashFFTConv(seqlen * 2, dtype=torch.float16) # 2x for padding, may need bfloat16
self.layer = nn.ModuleList(
    [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
if config.use_flashfftconv:
    for layer in self.layer:
        layer.attention.flashfftconv = self.flashfftconv # add it to the layers
```

Then, we adapt the actual sequence mixer to use `flashfftconv` in [monarch_mixer_sequence_mixer_flashfftconv.py](monarch_mixer_sequence_mixer_flashfftconv.py).

We make a couple more optimizations:
* We use our fast depthwise kernel.
* We introduce an "inference mode" that simply loads the convolution kernel from weights, instead of recomputing it every time (which is especially expensive for short kernels). An alternative is to use a fast kernel to generate the convolution kernel, as in the [M2 repo](https://github.com/HazyResearch/m2/tree/main/csrc/flashmm).