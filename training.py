import onnx
import onnxruntime.training.artifacts as artifacts
import onnxruntime.training.onnxblock as onnxblock

import argparse

from typing import Any


class ModelLoss(onnxblock.TrainingBlock):
    def __init__(self):
        super().__init__()
        self.loss_fn: onnxblock.Block = onnxblock.loss.CrossEntropyLoss(
                reduction="none")
        self.mul: onnxblock.Block = onnxblock.blocks.Mul()
        self.sum: onnxblock.Block = onnxblock.blocks.ReduceSum(
                keepdims=False)
        self.div: onnxblock.Block = onnxblock.blocks.Div()

    def build(self, logit: str) -> str:
        mask: str = onnxblock.blocks.InputLike(logit)("mask")
        return self.div(self.sum(
            self.mul(self.loss_fn(logit, "target"), mask)), self.sum(mask))


def main(args: argparse.Namespace) -> int:
    optim_table: dict[str, artifacts.OptimType] = {
            "adamw": artifacts.OptimType.AdamW,
            "sgd": artifacts.OptimType.SGD
    }

    model: onnx.ModelProto = onnx.load_model(args.model)

    # Find what params to train. Need fix if LoRA were implemented.
    target_params: list[str] = []
    frozen_params: list[str] = []
    for initializer in model.graph.initializer:
        name: str = initializer.name
        if not name.startswith("blocks.0.") and (name.endswith("ln0.weight") or name.endswith("ln0.bias")):
            continue
        target_params.append(initializer.name)

    loss: onnxblock.TrainingBlock = ModelLoss()
    for p in target_params:
        loss.requires_grad(p, True)
    for p in frozen_params:
        loss.requires_grad(p, False)

    with onnxblock.base(model):
        _ = loss("head")
        training_model: onnx.ModelProto
        eval_model: onnx.ModelProto
        training_model, eval_model = loss.to_model_proto()
        model_params: Any = loss.parameters()

    optimizer: onnxblock.ForwardBlock = onnxblock.optim.SGD()
    with onnxblock.empty_base():
        _ = optimizer(model_params)
        optimizer_model: onnx.ModelProto = optimizer.to_model_proto()

    onnxblock.save_checkpoint(model_params, "data/checkpoint")
    onnx.save(training_model, "data/training.onnx")
    onnx.save(eval_model, "data/eval.onnx")
    onnx.save(optimizer_model, "data/optimizer.onnx")

    return 0


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="The path to the ONNX model")
    parser.add_argument("optimizer", type=str,
                        choices=["sgd", "adamw"],
                        help="The optimizer to train the model")
    parser.add_argument("--artifact_directory", type=str, default=None,
                        help="The directory to save generated artifact")
    parser.add_argument("--prefix", type=str, default=None,
                        help="The prefix to used for the generated artifact.")
    exit(main(parser.parse_args()))
