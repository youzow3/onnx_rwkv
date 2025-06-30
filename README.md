# ONNX-RWKV: RWKV implementation in ONNX

ONNX-RWKV is a project to make RWKV series ONNX implementation. Since I started this after RWKV-7 release, currently only RWKV-7 is supported. The goal is to make RWKV-7 and later models' ONNX implementations.

## Supported Data Types

Fully fp32 only. I want to implement bf16, and fp16/fp32 mixed too.

## NOTE

I used Python 3.12 for this script.

Lots of implementation problem exists. Because of this, I checked only RWKV-7 G1 0.1b (and it worked).

Example code in C is available at cli-chat/.
