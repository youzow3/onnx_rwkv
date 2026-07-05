import onnx
import onnx.inliner
import onnxruntime.training.onnxblock as onnxblock
import onnxruntime.training.onnxblock.blocks as blocks
import ml_dtypes
import numpy as np

import argparse
from typing import Any


class SFTLoss(onnxblock.TrainingBlock):

    def __init__(self, vocab_size: int):
        assert isinstance(vocab_size, int)

        super().__init__()
        self.vocab_size: int = vocab_size
        self.pad: onnxblock.Block = blocks._BinaryOp("Pad")
        self.cast: onnxblock.Block = blocks.Cast(onnx.TensorProto.FLOAT)
        self.reshape: onnxblock.Block = blocks._BinaryOp("Reshape")
        self.loss_fn: onnxblock.Block = onnxblock.loss.CrossEntropyLoss(
            reduction="none")
        self.inputlike: onnxblock.Block = blocks.InputLike("x")
        self.mul: onnxblock.Block = blocks.Mul()
        self.div: onnxblock.Block = blocks.Div()
        self.sum: onnxblock.Block = blocks.ReduceSum(keepdims=False)

    def build(self, logit: str) -> str:
        self.base.graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array([0, -1, 0, 1], dtype=np.int64), "loss_pad"))
        self.base.graph.initializer.append(
            onnx.numpy_helper.from_array(np.array([-1], dtype=np.int64),
                                         "x_flatten_shape"))
        self.base.graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array([-1, self.vocab_size], dtype=np.int64),
                "logit_flatten_shape"))

        x_padded: str = self.pad("x", "loss_pad")
        x_flattened: str = self.reshape(x_padded, "x_flatten_shape")
        logit_flattened: str = self.reshape(logit, "logit_flatten_shape")

        # CrossEntropyLoss will use ValueInfo of the logit.
        self.base.graph.value_info.append(
            onnx.helper.make_tensor_value_info(logit_flattened,
                                               onnx.TensorProto.FLOAT,
                                               ["B(T-1)", "V"]))

        loss: str = self.loss_fn(logit_flattened, x_flattened)
        mask: str = self.inputlike("mask")
        mask_casted: str = self.cast(mask)
        mask_flattened: str = self.reshape(mask_casted, "x_flatten_shape")
        loss_masked: str = self.mul(loss, mask_flattened)
        loss_masked_sum: str = self.sum(loss_masked)
        mask_sum: str = self.sum(mask_flattened)
        return self.div(loss_masked_sum, mask_sum)


def main(args: argparse.Namespace) -> int:
    onnx_np_dtype_table: dict[int, np.dtype] = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.BFLOAT16: ml_dtypes.bfloat16
    }
    model: onnx.ModelProto = onnx.load_model(args.model,
                                             load_external_data=False)
    model = onnx.inliner.inline_local_functions(model)
    model = onnx.inliner.inline_selected_functions(
        model, [("", "GroupNormalization")], inline_schema_functions=True)
    model_update_needed: bool = False
    rng: np.random.Generator = np.random.default_rng()

    vocab_size: int = -1
    for t in model.graph.initializer:
        if t.name == "emb.weight":
            vocab_size = int(t.dims[0])
    if vocab_size == -1:
        print("Failed to find emb.weight in model.")
        return 1

    loss: onnxblock.TrainingBlock = SFTLoss(vocab_size)
    adapter_tensors: list[onnx.TensorProto] = []
    adapter_nodes: list[onnx.NodeProto] = []
    for initializer in model.graph.initializer:
        name: str = initializer.name
        if initializer.data_type == onnx.TensorProto.INT64:
            continue
        if initializer.name.endswith("_0"):
            continue  # default state vectors/matrixes (If op grad is not supported)

        if args.lora and len(initializer.dims) == 2:
            model_update_needed = True
            a: int
            b: int
            A_name: str = f"_{name}_A"
            B_name: str = f"_{name}_B"
            initializer.name = f"_{name}"
            a, b = initializer.dims

            # Append A and B matrix
            A: onnx.TensorProto = onnx.numpy_helper.from_array(
                rng.normal(0.0, args.lora_sigma, (a, args.lora_dim)).astype(
                    onnx_np_dtype_table[initializer.data_type]), A_name)
            B: onnx.TensorProto = onnx.numpy_helper.from_array(
                np.zeros((args.lora_dim, b),
                         onnx_np_dtype_table[initializer.data_type]), B_name)
            adapter_tensors += [A, B]
            # Create node such W_lora = W + A @ B
            adapter_nodes += [
                onnx.helper.make_node("MatMul", [A_name, B_name],
                                      [f"_{name}_AB"]),
                onnx.helper.make_node("Add", [f"_{name}", f"_{name}_AB"],
                                      [name])
            ]
            loss.requires_grad(A_name)
            loss.requires_grad(B_name)
        elif args.miss and len(initializer.dims) == 2:
            model_update_needed = True
            initializer.name = f"_{name}"
            m: int = initializer.dims[0]
            n: int = initializer.dims[1]
            m_expand: int = m // args.miss_dim
            n_expand: int = n // args.miss_dim
            assert m_expand * n_expand * args.miss_dim * args.miss_dim == m * n, f"MiSS cannot be expanded correctly: miss-dim: {args.miss_dim}, target-dim: {m, n}"
            D: onnx.TensorProto = onnx.numpy_helper.from_array(
                np.zeros((args.miss_dim, args.miss_dim),
                         onnx_np_dtype_table[initializer.data_type]),
                f"_{name}_D")
            adapter_tensors.append(D)
            adapter_nodes += [
                onnx.helper.make_node("Constant", [],
                                      [f"_{name}_D_expand_shape"],
                                      value_ints=[
                                          m_expand * n_expand, args.miss_dim,
                                          args.miss_dim
                                      ]),
                onnx.helper.make_node("Constant", [], [f"_{name}_shape"],
                                      value_ints=[m, n]),
                onnx.helper.make_node(
                    "Expand", [f"_{name}_D", f"_{name}_D_expand_shape"],
                    [f"_{name}_D_expanded_im"]),
                onnx.helper.make_node(
                    "Reshape", [f"_{name}_D_expanded_im", f"_{name}_shape"],
                    [f"_{name}_D_expanded"]),
                onnx.helper.make_node("Add",
                                      [f"_{name}", f"_{name}_D_expanded"],
                                      [name])
            ]
        else:
            loss.requires_grad(
                initializer.name)  # Assume all parameters are trainable

    if len(adapter_tensors) > 0:
        model.graph.initializer.extend(adapter_tensors)
    if len(adapter_nodes) > 0:
        model_update_needed = True
        # Prepend `adapter_nodes`
        model.graph.node.reverse()
        model.graph.node.extend(list(reversed(adapter_nodes)))
        model.graph.node.reverse()
    training_model: onnx.ModelProto
    eval_model: onnx.ModelProto
    model_params: Any
    with onnxblock.base(model):
        _ = loss("head")
        training_model, eval_model = loss.to_model_proto()
        model_params = loss.parameters()

    # make initializer for state gradients
    for k in model.graph.output:
        tensor_type: onnx.TypeProto.Tensor
        if k.type.WhichOneof("value") == "optional_type":
            tensor_type = k.type.optional_type.elem_type.tensor_type
        else:
            tensor_type = k.type.tensor_type

        dtype_table: dict[int, np.dtype] = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.FLOAT16: np.float16,
            onnx.TensorProto.BFLOAT16: ml_dtypes.bfloat16,
        }

        if k.name.startswith("wkv_next_"):
            dim: list[int] = [d.dim_value for d in tensor_type.shape.dim]
            dim[0] = 1  # does need batch size from x?
            tensor: np.ndarray = np.zeros(dim,
                                          dtype_table[tensor_type.elem_type])
            training_model.graph.initializer.append(
                onnx.numpy_helper.from_array(tensor, k.name + "_grad"))

    # make input order (x, mask, states...) instead of (x, states..., mask)
    mask_idx: int = None
    for idx, i in enumerate(training_model.graph.input):
        if i.name == "mask":
            mask_idx = idx
            break
    assert isinstance(mask_idx, int)
    input_len: int = len(training_model.graph.input)
    training_model.graph.input.extend(
        [training_model.graph.input[0], training_model.graph.input[mask_idx]])
    training_model.graph.input.extend(training_model.graph.input[1:mask_idx])
    training_model.graph.input.extend(training_model.graph.input[mask_idx +
                                                                 1:input_len])
    del training_model.graph.input[:input_len]
    assert len(training_model.graph.input) == input_len

    clip_grad: onnxblock.Block = onnxblock.optim.ClipGradNorm(args.clip_grad)
    optimizer: onnxblock.ForwardBlock = onnxblock.optim.AdamW(
        betas=[args.beta1, args.beta2],
        eps=args.adam_eps,
        weight_decay=0.0,
        clip_grad=clip_grad)
    optimizer_model: onnx.ModelProto
    with onnxblock.empty_base():
        _ = optimizer(model_params)
        optimizer_model = optimizer.to_model_proto()

    if model_update_needed:
        # onnx.save_model(model, args.model, save_as_external_data=True, location=f"{args.model}.data")
        pass
    if args.checkpoint:
        onnxblock.save_checkpoint(model_params, args.checkpoint)
    if args.training_model:
        onnx.save_model(training_model,
                        args.training_model,
                        save_as_external_data=True,
                        location=f"{args.training_model}.data")
    if args.eval_model:
        onnx.save_model(eval_model,
                        args.eval_model,
                        save_as_external_data=True,
                        location=f"{args.eval_model}.data")
    if args.optimizer_model:
        onnx.save_model(optimizer_model, args.optimizer_model)

    return 0


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-model", help="Optimizer model", type=str)
    parser.add_argument("--training-model", help="Training model", type=str)
    parser.add_argument("--eval-model", help="Eval model path", type=str)
    parser.add_argument("--checkpoint", help="Checkpoint file path", type=str)
    parser.add_argument("--lora", help="Enable LoRA", action="store_true")
    parser.add_argument("--lora-dim",
                        help="LoRA dimension",
                        default=8,
                        type=int)
    parser.add_argument("--lora-sigma",
                        help="LoRA initialization parameter",
                        default=1.0,
                        type=float)
    parser.add_argument("--miss", help="Enable MiSS", action="store_true")
    parser.add_argument("--miss-dim",
                        help="MiSS dimension",
                        default=16,
                        type=int)
    parser.add_argument("--beta1",
                        help="beta1 for Adam",
                        default=0.9,
                        type=float)
    parser.add_argument("--beta2",
                        help="beta2 for Adam",
                        default=0.99,
                        type=float)
    parser.add_argument("--adam_eps",
                        help="eps for Adam",
                        default=1e-18,
                        type=float)
    parser.add_argument("--clip_grad",
                        help="gradient clipping",
                        default=0.1,
                        type=float)
    parser.add_argument("model", help="Target ONNX model", type=str)
    exit(main(parser.parse_args()))
