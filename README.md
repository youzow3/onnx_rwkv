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
>
> 2. TopK can be applied because TopK is only way to get sorted logits which is needed to compute TopP.
>
> 3. As the rwkv pip package does, temperature is applied after TopK and TopP, not before TopK and TopP.
>

## Supported Training Methods (via `training.py`)

- SFT

>
> Additional model input is presented: "mask" INT64 tensor (batch, seq), which is used to mask pad token.
>

>
> Model generation is done with [my custom onnxruntime](https://github.com/youzow3/onnxruntime). Merge main, controlflow, and transpose_fix.
>

Training model generation is tested with rwkv7-g1g-2.9b.

### PEFT methods

- LoRA

## Using models on Chatbot

[Chatbot](https://github.com/youzow3/chatbot) is my current test environment. If you want to use RWKV on the program, you need to specify -s to generate sampling included ONNX file, which is required for my RWKV Module for Chatbot. (See rwkv.c on the repo.)

I tested fp32 RWKV-7 G1 0.1b, and fp32 RWKV-7a G1b 0.1b.

## NOTE

`LinearAttention` is presented in Opset27, but it cannot be used because they don't support generalized delta rule. Do I need to open issue?

I don't know, but I think RWKV-8 cannot be implemented in pure-ONNX because of ROSA.

Implementation problem may exist.

~~Example code in C is available at cli-chat/.~~

The example code still available, but I recommend using Chatbot.

