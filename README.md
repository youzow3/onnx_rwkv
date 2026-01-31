# ONNX-RWKV: RWKV implementation in ONNX

ONNX-RWKV is a project to make RWKV series ONNX implementation. Since I started this after RWKV-7 release, currently only RWKV-7 is supported. The goal is to make RWKV-7 and later models' ONNX implementations.

## Supported Data Types

- fp32
- fp16/fp32 mixed
- bf16

## Supported Models

- RWKV-7
- RWKV-7a (RWKV-7 with DeepEmbed)

## Supported Sampling Methods

- Presence penalties
    - Alpha Presence
    - Alpha Frequency
    - Alpha Decay
- TopK
- Temperature
- TopP

>
> Default sampling parameter is same settings described at [this page](https://huggingface.co/BlinkDL/rwkv7-g1).
>

>
> Sampling implementation is almost same as rwkv pip package. However:
>
> 1. Alpha presence penalty (occurence in this and the pip package code) is applied to all tokens instead of all tokens except digits and tabs.
> 2. TopK can be applied because TopK is only way to get sorted logits which is needed to compute TopP.
> 3. As the rwkv pip package does, temperature is applied after TopK and TopP, not before TopK and TopP.
>

## Supported Training Methods

- SFT
- REINFORCE (token level reward)

>
> Both type receive "mask" INT64 tensor (batch, seq), which is used to mask pad token.
>
> REINFORCE also receives "reward" FLOAT tensor (batch, seq), which is token level reward.
>

To use SFT, just use --training flag.
To use REINFORCE, specify --training and --rl flags.

>
> Model generation is done with [my custom onnxruntime](https://github.com/youzow3/onnxruntime). Merge main, controlflow, and transpose_fix.
>

Training model generation is tested with rwkv7-g1-0.1b, rwkv7-g1c-1.5b, and rwkv7-g1d-2.9b. However, none of them are actually tested with ONNXRuntime Training API to train.

## Using models on Chatbot

[Chatbot](https://github.com/youzow3/chatbot) is my current test environment. If you want to use RWKV on the program, you need to specify -s to generate sampling included ONNX file, which is required for my RWKV Module for Chatbot. (See rwkv.c on the repo.)

I tested fp32 RWKV-7 G1 0.1b, and fp32 RWKV-7a G1b 0.1b.

## NOTE

Implementation problem may still exist.

~~Example code in C is available at cli-chat/.~~

The example code still available, but I recommend using Chatbot.
