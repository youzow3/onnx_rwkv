# ONNX-RWKV: RWKV implementation in ONNX

ONNX-RWKV is a project to make RWKV series ONNX implementation. Since I started this after RWKV-7 release, currently only RWKV-7 is supported. The goal is to make RWKV-7 and later models' ONNX implementations.

## Supported Data Types

* fp32
* fp16/fp32 mixed
* bf16

## NOTE

I used Python 3.12 for this script.

Lots of implementation problem exists. Because of this, I checked only RWKV-7 G1 with fp32 on CPU. 0.1b, 1.5b, and 2.9b are worked with cli-chat.

Example code in C is available at cli-chat/.
