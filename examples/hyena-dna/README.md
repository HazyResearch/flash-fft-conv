# Hyena-DNA

This folder shows an example of adapting Hyena-DNA to use FlashFFTConv.

## Requirements

This code downloads the config from HuggingFace, so you need to have git lfs installed.

To check, run:
```
git lfs install
```
If it fails, you can install git lfs using your favorite package manager.

## Usage

We have sample scripts to benchmark Hyena DNA using PyTorch, vs. using FlashFFTConv:
```
python benchmark_hyena_dna_fwd.py
python benchmark_flash_dna_fwd.py
```

## Changes to Use FlashFFTConv in Hyena

We describe the changes necessary to use FlashFFTConv in HyenaDNA:

Create an instance of `FlashFFTConv` in `LMBackbone`. In [hyenadna_flashfftconv.py](hyenadna_flashfftconv.py), lines 716-721:
```Python
seqlen = layer['l_max']
seqlen = next_power_of_2(seqlen) * 2
self.flashfftconv = FlashFFTConv(seqlen, dtype=torch.float16) # may need bfloat16

for layer in self.layers:
    layer.mixer.flashfftconv = self.flashfftconv
```

Note that HyenaDNA does not use sequence lengths that are powers of two, so we need to find the next closest power of two (lines 688-689).

Then, we adapt the Hyena layers to use the `flashfftconv` variable (lines 269-289).

We make a couple more optimizations:
* We use our fast depthwise kernel.
* We introduce an "inference mode" that simply loads the convolution kernel from weights, instead of recomputing it every time. An alternative is to use a fast kernel to generate the convolution kernel, as in the [M2 repo](https://github.com/HazyResearch/m2/tree/main/csrc/flashmm).
* In this benchmarking code, the weights have different names than in the PyTorch code, so the model will not load pretrained weights out of the box. We are working on a minimal example that can load the pretrained weights.