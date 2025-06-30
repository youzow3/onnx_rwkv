# CLI-Chat

It's just a toy example written in C with no depedencies other than onnxruntime. In this code, I used original vocaburary file "rwkv_vocab_v20230424.txt" from BlinkDL/RWKV-LM. use program with "-m \[your onnx model\] -t \[path to rwkv_vocab_v20230424.txt\]"

## NOTE

* Sampling method in this program is random sampling with temparture t=0.7. See rwkv_decode().

* The code is too chaostic for me, so I don't have any plan to improve it.
