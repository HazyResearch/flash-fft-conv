# Hyena

This folder shows an example of adapting Hyena to use FlashFFTConv.
The original files are sourced from [safari](https://github.com/HazyResearch/safari).

## Requirements

Install model-specific requirements. See the [safari repo](https://github.com/HazyResearch/safari/tree/main) for instructions.

This code depends on an old version of FlashAttention (0.2.8) for the MLP interface.

## Usage

We have sample configs for Hyena models of different sizes that you can benchmark:
```
python benchmark_fwd.py experiment=pile/hyena.yaml
python benchmark_fwd.py experiment=pile/hyena-flashfft.yaml
```

## Changes to Use FlashFFTConv in Hyena

We describe the changes necessary to use FlashFFTConv in Hyena:

Create an instance of `FlashFFTConv` in `LMBackbone`. In [src/models/sequence/long_conv_lm.py](src/models/sequence/long_conv_lm.py), lines 193-197:
```Python
if use_flashfftconv:
    self.flashfftconv = FlashFFTConv(layer['l_max'] * 2, dtype=torch.float16)

    for layer in self.layers:
        layer.mixer.flashfftconv = self.flashfftconv
```

Then, we adapt Hyena to use the `flashfftconv` variable in [src/models/sequence/hyena-flashfft.py](src/models/sequence/hyena-flashfft.py).

We make a couple more optimizations:
* We use our fast depthwise kernel.
* We introduce an "inference mode" that simply loads the convolution kernel from weights, instead of recomputing it every time. An alternative is to use a fast kernel to generate the convolution kernel, as in the [M2 repo](https://github.com/HazyResearch/m2/tree/main/csrc/flashmm).