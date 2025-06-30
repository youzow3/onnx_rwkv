project('cli-chat', 'c')

onnxruntime = dependency('libonnxruntime')

executable('cli-chat', 'main.c', link_args: '-lm', dependencies: [onnxruntime])
