import argparse
import glob
import ml_dtypes
import numpy as np
import onnx
import safetensors.numpy as safetensors

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, List, Set

DTYPE_TABLE: dict[str, np.dtype] = {
    "auto": None,
    "fp32": np.dtype(np.float32),
    "fp16": np.dtype(np.float16),
    "fp16-mixed": np.dtype(np.float16),
    "bf16": np.dtype(ml_dtypes.bfloat16),
    "bf16-mixed": np.dtype(ml_dtypes.bfloat16),
    "i8": np.dtype(np.int8),
}
WKV_DTYPE_TABLE: dict[str, np.dtype] = DTYPE_TABLE.copy()
WKV_DTYPE_TABLE["fp16-mixed"] = np.dtype(np.float32)
WKV_DTYPE_TABLE["bf16-mixed"] = np.dtype(np.float32)
QUANT_DTYPE_TABLE: dict[str, np.dtype] = DTYPE_TABLE.copy()

DTYPE_TO_ELEM_TYPE: dict[np.dtype, int] = {
    np.dtype(np.float32): onnx.TensorProto.FLOAT,
    np.dtype(np.float16): onnx.TensorProto.FLOAT16,
    np.dtype(ml_dtypes.bfloat16): onnx.TensorProto.BFLOAT16,
    np.dtype(np.int8): onnx.TensorProto.INT8,
    np.dtype(ml_dtypes.int4): onnx.TensorProto.INT4
}

ELEM_TYPE_TO_DTYPE: dict[int, np.dtype] = {
    onnx.TensorProto.FLOAT: np.dtype(np.float32),
    onnx.TensorProto.FLOAT16: np.dtype(np.float16),
    onnx.TensorProto.BFLOAT16: np.dtype(ml_dtypes.bfloat16),
    onnx.TensorProto.INT8: np.dtype(np.int8),
    onnx.TensorProto.INT4: np.dtype(ml_dtypes.int4)
}


@dataclass
class HyperParameter:
    V: int
    C: int
    H: int
    N: int
    D_de: int | None  # dim for deepembd
    D_w: int  # lora dim for w1/w2
    D_a: int  # lora dim for a1/a2
    D_v: int  # lora dim for v1/v2
    D_g: int  # lora dim for g1/g2
    nlayers: int
    elem_type: int
    wkv_elem_type: int
    quant_elem_type: int | None  # None if no quantization

    @staticmethod
    def from_parameters(parameters: Dict[str, np.ndarray]) -> HyperParameter:
        V, C = parameters["rwkv7.emb.weight"].shape
        H, N = parameters["rwkv7.blocks.0.att.r_k"].shape
        D_de = parameters["rwkv7.blocks.0.ffn.s1"].shape[
            1] if "rwkv7.blocks.0.ffn.s1" in parameters.keys() else None
        D_w = parameters["rwkv7.blocks.0.att.w1"].shape[1]
        D_a = parameters["rwkv7.blocks.0.att.a1"].shape[1]
        D_v = parameters["rwkv7.blocks.0.att.v1"].shape[1]
        D_g = parameters["rwkv7.blocks.0.att.g1"].shape[1]
        nlayers: int = 0
        while f"rwkv7.blocks.{nlayers}.att.r_k" in parameters.keys():
            nlayers += 1
        dtype: int = DTYPE_TO_ELEM_TYPE[parameters["rwkv7.emb.weight"].dtype]
        return HyperParameter(V, C, H, N, D_de, D_w, D_a, D_v, D_g, nlayers,
                              dtype, dtype, None)


class Model:

    def __init__(self, args: argparse.Namespace,
                 hyper_parameters: HyperParameter):

        self.args = deepcopy(args)
        self.hparam = deepcopy(hyper_parameters)
        self.dtype: np.dtype = DTYPE_TABLE[args.dtype]
        self.wkv_dtype: np.dtype = WKV_DTYPE_TABLE[args.dtype]
        self.quant_dtype: np.dtype = QUANT_DTYPE_TABLE[args.quantize]
        self.quant_filter: set[str] = {"rwkv7.emb.weight", "head.weight"}

        if self.dtype is None:
            self.dtype = ELEM_TYPE_TO_DTYPE[self.hparam.elem_type]
        else:
            self.hparam.elem_type = DTYPE_TO_ELEM_TYPE[self.dtype]
        if self.wkv_dtype is None:
            self.wkv_dtype = ELEM_TYPE_TO_DTYPE[self.hparam.wkv_elem_type]
        else:
            self.hparam.wkv_elem_type = DTYPE_TO_ELEM_TYPE[self.wkv_dtype]
        if self.quant_dtype is None:
            self.quant_dtype = None if self.hparam.quant_elem_type is None else ELEM_TYPE_TO_DTYPE[
                self.hparam.quant_elem_type]
        else:
            self.hparam.quant_elem_type = DTYPE_TO_ELEM_TYPE[self.quant_dtype]

        for layer in range(self.hparam.nlayers):
            self.quant_filter.add(f"rwkv7.blocks.{layer}.att.r_k")

        self.wkv_graph: onnx.GraphProto = None
        self.layer_norm_merge_else: onnx.GraphProto = None
        self.layer_norm_merge_then: onnx.GraphProto = None
        self.sampling_else: onnx.GraphProto = None
        self.sampling_then: onnx.GraphProto = None

        self.dynamic_quantize_hparam_computed: set[str] = set()
        self.dynamic_quantized: set[str] = set()

    def _mul(self,
             a: str,
             b: str,
             output_value: str,
             a_type_proto: onnx.TypeProto = None,
             b_type_proto: onnx.TypeProto = None,
             is_a_param: bool = False,
             is_b_param: bool = False) -> List[onnx.NodeProto]:
        vn: Callable[[str],
                     str] = lambda name: f"buf.mul.{hash(output_value)}.{name}"
        mul_nodes: list[onnx.NodeProto] = []
        if self.args.evolution_strategy and ((a_type_proto is not None) or
                                             (b_type_proto is not None)):
            a_p: str = a
            b_p: str = b
            if a_type_proto is not None:
                a_p = vn("a")

                a_merge_outputs: list[onnx.ValueInfoProto] = [
                    onnx.helper.make_value_info(f"{a}.population",
                                                a_type_proto)
                ]
                a_merge_else_nodes: list[onnx.NodeProto] = []
                a_merge_else_nodes.append(
                    onnx.helper.make_node("Unsqueeze", [a, "0"],
                                          [f"{a}.population"]))
                a_merge_else_graph: onnx.GraphProto = onnx.helper.make_graph(
                    a_merge_else_nodes,
                    f"graph.{hash(a + output_value)}.merge.else", [],
                    a_merge_outputs)

                a_merge_then_nodes: list[onnx.NodeProto] = []
                a_merge_then_nodes.append(
                    onnx.helper.make_node("OptionalGetElement",
                                          [f"{a}.population.opt"],
                                          [f"{a}.population_"]))
                a_merge_then_nodes.append(
                    onnx.helper.make_node("Add", [a, f"{a}.population_"],
                                          [f"{a}.population"]))
                a_merge_then_graph: onnx.GraphProto = onnx.helper.make_graph(
                    b_merge_then_nodes,
                    f"graph.{hash(a + output_value)}.merge.then", [],
                    a_merge_outputs)

                mul_nodes.append(
                    onnx.helper.make_node("OptionalHasElement",
                                          [f"{a}.population.opt"],
                                          [f"{a}.population.opt.has_element"]))
                mul_nodes.append(
                    onnx.helper.make_node("If",
                                          [f"{a}.population.opt.has_element"],
                                          [a_p],
                                          else_branch=a_merge_else_graph,
                                          then_branch=a_merge_then_graph))
            if b_type_proto is not None:
                b_p = vn("b")

                b_merge_outputs: list[onnx.ValueInfoProto] = [
                    onnx.helper.make_value_info(f"{b}.population",
                                                b_type_proto)
                ]
                b_merge_else_nodes: list[onnx.NodeProto] = []
                b_merge_else_nodes.append(
                    onnx.helper.make_node("Unsqueeze", [b, "0"],
                                          [f"{b}.population"]))
                b_merge_else_graph: onnx.GraphProto = onnx.helper.make_graph(
                    b_merge_else_nodes,
                    f"graph.{hash(b + output_value)}.merge.else", [],
                    b_merge_outputs)

                b_merge_then_nodes: list[onnx.NodeProto] = []
                b_merge_then_nodes.append(
                    onnx.helper.make_node("OptionalGetElement",
                                          [f"{b}.population.opt"],
                                          [f"{b}.population_"]))
                b_merge_then_nodes.append(
                    onnx.helper.make_node("Add", [b, f"{b}.population_"],
                                          [f"{b}.population"]))
                b_merge_then_graph: onnx.GraphProto = onnx.helper.make_graph(
                    b_merge_then_nodes,
                    f"graph.{hash(b + output_value)}.merge.then", [],
                    b_merge_outputs)

                mul_nodes.append(
                    onnx.helper.make_node("OptionalHasElement",
                                          [f"{b}.population.opt"],
                                          [f"{b}.population.opt.has_element"]))
                mul_nodes.append(
                    onnx.helper.make_node("If",
                                          [f"{b}.population.opt.has_element"],
                                          [b_p],
                                          else_branch=b_merge_else_graph,
                                          then_branch=b_merge_then_graph))
            mul_nodes.append(
                onnx.helper.make_node("Mul", [a_p, b_p], [output_value]))
        else:
            mul_nodes.append(
                onnx.helper.make_node("Mul", [a, b], [output_value]))
        return mul_nodes

    def _layer_norm(self,
                    x: str,
                    a: str,
                    b: str,
                    output_value: str,
                    axis: int = -1,
                    epsilon: float = 1e-5,
                    stash_type: int = 1) -> List[onnx.NodeProto]:
        vn: Callable[[str],
                     str] = lambda name: f"buf.ln.{hash(output_value)}.{name}"
        ln_nodes: list[onnx.NodeProto] = []
        param_type: onnx.TypeProto = onnx.helper.make_tensor_type_proto(
            self.hparam.elem_type, ["P", 1, 1, self.hparam.C])
        merge_outputs: list[onnx.ValueInfoProto] = [
            onnx.helper.make_value_info(vn("a"), param_type),
            onnx.helper.make_value_info(vn("b"), param_type)
        ]
        if self.args.evolution_strategy:
            merge_else_nodes: list[onnx.NodeProto] = []
            merge_else_nodes.append(
                onnx.helper.make_node("Reshape", [a, "111C"], [vn("a")]))
            merge_else_nodes.append(
                onnx.helper.make_node("Reshape", [b, "111C"], [vn("b")]))
            merge_else_graph: onnx.GraphProto = onnx.helper.make_graph(
                merge_else_nodes, f"graph.{hash(output_value)}.ln.else", [],
                merge_outputs)

            merge_then_nodes: list[onnx.NodeProto] = []
            merge_then_nodes.append(
                onnx.helper.make_node("OptionalGetElement",
                                      [f"{a}.population.opt"],
                                      [vn("a.population")]))
            merge_then_nodes.append(
                onnx.helper.make_node("OptionalGetElement",
                                      [f"{b}.population.opt"],
                                      [vn("b.population")]))
            merge_then_nodes.append(
                onnx.helper.make_node("Add", [a, vn("a.population")],
                                      [vn("a")]))
            merge_then_nodes.append(
                onnx.helper.make_node("Add", [b, vn("b.population")],
                                      [vn("b")]))
            merge_then_graph: onnx.GraphProto = onnx.helper.make_graph(
                merge_then_nodes, f"graph.{hash(output_value)}.ln.then", [],
                merge_outputs)

            ln_nodes.append(
                onnx.helper.make_node("OptionalHasElement",
                                      [f"{a}.population.opt"],
                                      [f"{a}.population.opt.has_element"]))
            ln_nodes.append(
                onnx.helper.make_node("OptionalHasElement",
                                      [f"{b}.population.opt"],
                                      [f"{b}.population.opt.has_element"]))
            ln_nodes.append(
                onnx.helper.make_node("And", [
                    f"{a}.population.opt.has_element",
                    f"{b}.population.opt.has_element"
                ], [vn("has_element")]))
            ln_nodes.append(
                onnx.helper.make_node("If", [vn("has_element")],
                                      [vn("a"), vn("b")],
                                      else_branch=merge_else_graph,
                                      then_branch=merge_then_graph))

            ln_nodes.append(
                onnx.helper.make_node("LayerNormalization", [x, "1.0"],
                                      [vn(output_value)],
                                      axis=axis,
                                      epsilon=epsilon,
                                      stash_type=stash_type))
            ln_nodes.append(
                onnx.helper.make_node(
                    "Mul", [vn(output_value), vn("a")], [vn("scaled")]))
            ln_nodes.append(
                onnx.helper.make_node("Add",
                                      [vn("scaled"), vn("b")], [output_value]))
        else:
            ln_nodes.append(
                onnx.helper.make_node("LayerNormalization", [x, a, b],
                                      [output_value]))
        return ln_nodes

    def _group_norm(self,
                    x: str,
                    a: str,
                    b: str,
                    output_value: str,
                    num_groups: int,
                    epsilon: float = 1e-5,
                    stash_type: int = 1) -> List[onnx.NodeProto]:
        vn: Callable[[str],
                     str] = lambda name: f"buf.ln.{hash(output_value)}.{name}"
        ln_nodes: list[onnx.NodeProto] = []
        param_type: onnx.TypeProto = onnx.helper.make_tensor_type_proto(
            self.hparam.elem_type, ["P", 1, 1, self.hparam.C])
        merge_outputs: list[onnx.ValueInfoProto] = [
            onnx.helper.make_value_info(f"{a}.population", param_type),
            onnx.helper.make_value_info(f"{b}.population", param_type)
        ]
        if self.args.evolution_strategy:
            merge_else_nodes: list[onnx.NodeProto] = []
            merge_else_nodes.append(
                onnx.helper.make_node("Reshape", [a, "111C"],
                                      [f"{a}.population"]))
            merge_else_nodes.append(
                onnx.helper.make_node("Reshape", [b, "111C"],
                                      [f"{b}.population"]))
            merge_else_graph: onnx.GraphProto = onnx.helper.make_graph(
                merge_else_nodes, f"graph.{hash(output_value)}.gn.else", [],
                merge_outputs)

            merge_then_nodes: list[onnx.NodeProto] = []
            merge_then_nodes.append(
                onnx.helper.make_node("OptionalGetElement",
                                      [f"{a}.population.opt"],
                                      [f"{a}.population_"]))
            merge_then_nodes.append(
                onnx.helper.make_node("OptionalGetElement",
                                      [f"{b}.population.opt"],
                                      [f"{b}.population_"]))
            merge_then_nodes.append(
                onnx.helper.make_node("Add", [a, f"{a}.population_"],
                                      [f"{a}.population"]))
            merge_then_nodes.append(
                onnx.helper.make_node("Add", [b, f"{b}.population_"],
                                      [f"{b}.population"]))
            merge_then_graph: onnx.GraphProto = onnx.helper.make_graph(
                merge_then_nodes, f"graph.{hash(output_value)}.gn.then", [],
                merge_outputs)

            ln_nodes.append(
                onnx.helper.make_node("OptionalHasElement",
                                      [f"{a}.population.opt"],
                                      [f"{a}.population.opt.has_element"]))
            ln_nodes.append(
                onnx.helper.make_node("OptionalHasElement",
                                      [f"{b}.population.opt"],
                                      [f"{b}.population.opt.has_element"]))
            ln_nodes.append(
                onnx.helper.make_node("And", [
                    f"{a}.population.opt.has_element",
                    f"{b}.population.opt.has_element"
                ], [vn("has_element")]))
            ln_nodes.append(
                onnx.helper.make_node("If", [vn("has_element")],
                                      [vn("a"), vn("b")],
                                      else_branch=merge_else_graph,
                                      then_branch=merge_then_graph))

            ln_nodes.append(
                onnx.helper.make_node("GroupNormalization", [x, "1.0", "0.0"],
                                      [vn(output_value)],
                                      epsilon=epsilon,
                                      num_groups=num_groups,
                                      stash_type=stash_type))
            ln_nodes.append(
                onnx.helper.make_node(
                    "Mul", [vn(output_value), vn("a")], [vn("scaled")]))
            ln_nodes.append(
                onnx.helper.make_node("Add",
                                      [vn("scaled"), vn("b")], [output_value]))
        else:
            ln_nodes.append(
                onnx.helper.make_node("GroupNormalization", [x, a, b],
                                      [output_value],
                                      epsilon=epsilon,
                                      num_groups=num_groups,
                                      stash_type=stash_type))
        return ln_nodes

    def _dynamic_quantize_hparam(self, x: str, x_scale: str, x_zero_point: str,
                                 qmin: str, qmax: str, qrange: str,
                                 to: int) -> List[onnx.NodeProto]:
        if x in self.dynamic_quantize_hparam_computed:
            return []
        hparam_nodes: list[onnx.NodeProto] = []
        hparam_nodes.append(
            onnx.helper.make_node("Sub", [f"{x}.max", f"{x}.min"],
                                  [f"{x}.range"]))
        hparam_nodes.append(
            onnx.helper.make_node("Div", [f"{x}.range", qrange], [x_scale]))
        hparam_nodes.append(
            onnx.helper.make_node("Div", [f"{x}.min", x_scale],
                                  [f"{x}.min.scaled"]))
        hparam_nodes.append(
            onnx.helper.make_node("Sub", [qmin, f"{x}.min.scaled"],
                                  [f"{x_zero_point}.fp"]))
        hparam_nodes.append(
            onnx.helper.make_node("Round", [f"{x_zero_point}.fp"],
                                  [f"{x_zero_point}.fp.rounded"]))
        hparam_nodes.append(
            onnx.helper.make_node("Clip",
                                  [f"{x_zero_point}.fp.rounded", qmin, qmax],
                                  [f"{x_zero_point}.fp.clipped"]))
        hparam_nodes.append(
            onnx.helper.make_node("Cast", [f"{x_zero_point}.fp.clipped"],
                                  [x_zero_point],
                                  to=to))
        self.dynamic_quantize_hparam_computed.add(x)
        return hparam_nodes

    def _dynamic_quantize(self, x: str, x_quant: str, x_scale: str,
                          x_zero_point: str, qmin: str, qmax: str, qrange: str,
                          to: int) -> List[onnx.NodeProto]:
        if x in self.dynamic_quantized:
            return []

        quant_nodes: list[onnx.NodeProto] = []
        quant_nodes.append(
            onnx.helper.make_node("ReduceMin", [x], [f"{x}.min"], keepdims=0))
        quant_nodes.append(
            onnx.helper.make_node("ReduceMax", [x], [f"{x}.max"], keepdims=0))
        quant_nodes += self._dynamic_quantize_hparam(x, x_scale, x_zero_point,
                                                     qmin, qmax, qrange, to)
        quant_nodes.append(
            onnx.helper.make_node("QuantizeLinear", [x, x_scale, x_zero_point],
                                  [x_quant]))
        self.dynamic_quantized.add(x)
        return quant_nodes

    def _matmul(self,
                A: str,
                B: str,
                output_value: str,
                transA: bool = False,
                transB: bool = False,
                auto_quantize: bool = True,
                quantizeA: bool = False,
                quantizeB: bool = False,
                dequantize_output: bool = True,
                isparamA: bool = False,
                isparamB: bool = False) -> List[onnx.NodeProto]:
        vn: Callable[
            [str],
            str] = lambda name: f"buf.matmul.{hash(output_value)}.{name}"
        matmul_nodes: list[onnx.NodeProto] = []

        if (self.quant_dtype == np.int8) and auto_quantize:
            if quantizeA:
                matmul_nodes += self._dynamic_quantize(A, f"{A}.quantized",
                                                       f"{A}.a", f"{A}.b",
                                                       "INT8_MIN", "INT8_MAX",
                                                       "INT8_RANGE",
                                                       onnx.TensorProto.INT8)
            if quantizeB:
                matmul_nodes += self._dynamic_quantize(B, f"{B}.quantized",
                                                       f"{B}.a", f"{B}.b",
                                                       "INT8_MIN", "INT8_MAX",
                                                       "INT8_RANGE",
                                                       onnx.TensorProto.INT8)

        if self.args.evolution_strategy and (isparamA or isparamB):
            assert isparamA ^ isparamB, "This MatMul must be pre-computed"
            merge_outputs: list[onnx.ValueInfoProto] = [
                onnx.helper.make_tensor_value_info("output",
                                                   self.hparam.elem_type,
                                                   ["P", 1, None, None])
            ]
            if isparamA:

                a_merge_else_nodes: list[onnx.NodeProto] = []
                a_merge_else_nodes += self._matmul(
                    A,
                    B,
                    "output",
                    transA=transA,
                    transB=transB,
                    auto_quantize=auto_quantize,
                    quantizeA=quantizeA,
                    quantizeB=quantizeB,
                    dequantize_output=dequantize_output)
                a_merge_else_graph: onnx.GraphProto = onnx.helper.make_graph(
                    a_merge_else_nodes,
                    f"graph.matmul.{hash(output_value)}.else", [],
                    merge_outputs)

                a_merge_then_nodes: list[onnx.NodeProto] = []
                if (self.quant_dtype == np.int8) and auto_quantize:
                    a_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{A}.population.a.opt"],
                                              [vn("a.quantized")]))
                    a_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{A}.population.b.opt"],
                                              [vn("b.quantized")]))
                    a_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{A}.population.a.a.opt"],
                                              [vn("a.a")]))
                    a_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{A}.population.a.b.opt"],
                                              [vn("a.b")]))
                    a_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{A}.population.b.a.opt"],
                                              [vn("b.a")]))
                    a_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{A}.population.b.b.opt"],
                                              [vn("b.b")]))
                else:
                    a_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{A}.population.a.opt"],
                                              [vn("a")]))
                    a_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{A}.population.b.opt"],
                                              [vn("b")]))
                a_merge_then_nodes += self._matmul(
                    A,
                    B,
                    vn("out"),
                    transA=transA,
                    transB=transB,
                    auto_quantize=auto_quantize,
                    quantizeA=quantizeA,
                    quantizeB=quantizeB,
                    dequantize_output=dequantize_output)
                if transA:
                    a_merge_then_nodes += self._matmul(
                        vn("a"),
                        B,
                        vn("aB"),
                        transB=transB,
                        auto_quantize=auto_quantize,
                        quantizeB=quantizeB)
                    a_merge_then_nodes += self._matmul(
                        vn("b"),
                        vn("aB"),
                        vn("p"),
                        auto_quantize=auto_quantize,
                        quantizeB=True)
                else:
                    a_merge_then_nodes += self._matmul(
                        vn("b"),
                        B,
                        vn("bB"),
                        transB=transB,
                        auto_quantize=auto_quantize,
                        quantizeB=quantizeB)
                    a_merge_then_nodes += self._matmul(
                        vn("a"),
                        vn("bB"),
                        vn("p"),
                        auto_quantize=auto_quantize,
                        quantizeB=True)
                a_merge_then_nodes.append(
                    onnx.helper.make_node("Add",
                                          [vn("out"), vn("p")], ["output"]))
                a_merge_then_graph: onnx.GraphProto = onnx.helper.make_graph(
                    a_merge_then_nodes,
                    f"graph.matmul.{hash(output_value)}.then", [],
                    merge_outputs)

                matmul_nodes.append(
                    onnx.helper.make_node("OptionalHasElement",
                                          [f"{A}.population.a.opt"],
                                          [vn("a.has_element")]))
                matmul_nodes.append(
                    onnx.helper.make_node("OptionalHasElement",
                                          [f"{A}.population.b.opt"],
                                          [vn("b.has_element")]))
                matmul_nodes.append(
                    onnx.helper.make_node(
                        "And", [vn("a.has_element"),
                                vn("b.has_element")], [vn("has_element")]))
                matmul_nodes.append("If", [vn("has_element")], [output_value],
                                    else_branch=a_merge_else_graph,
                                    then_branch=a_merge_then_graph)
            elif isparamB:
                b_merge_else_nodes: list[onnx.NodeProto] = []
                b_merge_else_nodes += self._matmul(
                    A,
                    B,
                    "output",
                    transA=transA,
                    transB=transB,
                    auto_quantize=auto_quantize,
                    quantizeA=quantizeA,
                    quantizeB=quantizeB,
                    dequantize_output=dequantize_output)
                b_merge_else_graph: onnx.GraphProto = onnx.helper.make_graph(
                    b_merge_else_nodes,
                    f"graph.matmul.{hash(output_value)}.else", [],
                    merge_outputs)

                b_merge_then_nodes: list[onnx.NodeProto] = []
                if (self.quant_dtype == np.int8) and auto_quantize:
                    b_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{B}.population.a.opt"],
                                              [vn("a.quantized")]))
                    b_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{B}.population.b.opt"],
                                              [vn("b.quantized")]))
                    b_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{B}.population.a.a.opt"],
                                              [vn("a.a")]))
                    b_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{B}.population.a.b.opt"],
                                              [vn("a.b")]))
                    b_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{B}.population.b.a.opt"],
                                              [vn("b.a")]))
                    b_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{B}.population.b.b.opt"],
                                              [vn("b.b")]))
                else:
                    b_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{B}.population.a.opt"],
                                              [vn("a")]))
                    b_merge_then_nodes.append(
                        onnx.helper.make_node("OptionalGetElement",
                                              [f"{B}.population.b.opt"],
                                              [vn("b")]))
                b_merge_then_nodes += self._matmul(
                    A,
                    B,
                    vn("out"),
                    transA=transA,
                    transB=transB,
                    auto_quantize=auto_quantize,
                    quantizeA=quantizeA,
                    quantizeB=quantizeB,
                    dequantize_output=dequantize_output)
                if transB:
                    b_merge_then_nodes += self._matmul(
                        A,
                        vn("b"),
                        vn("Ab"),
                        transA=transA,
                        auto_quantize=auto_quantize,
                        quantizeA=quantizeA)
                    b_merge_then_nodes += self._matmul(
                        vn("Ab"),
                        vn("a"),
                        vn("p"),
                        auto_quantize=auto_quantize,
                        quantizeA=True)
                else:
                    b_merge_then_nodes += self._matmul(
                        A,
                        vn("a"),
                        vn("Aa"),
                        transA=transA,
                        auto_quantize=auto_quantize,
                        quantizeA=quantizeA)
                    b_merge_then_nodes += self._matmul(
                        vn("Aa"),
                        vn("b"),
                        vn("p"),
                        auto_quantize=auto_quantize,
                        quantizeA=True)
                b_merge_then_nodes.append(
                    onnx.helper.make_node("Add",
                                          [vn("out"), vn("p")], ["output"]))
                b_merge_then_graph: onnx.GraphProto = onnx.helper.make_graph(
                    b_merge_then_nodes,
                    f"graph.matmul.{hash(output_value)}.then", [],
                    merge_outputs)

                matmul_nodes.append(
                    onnx.helper.make_node("OptionalHasElement",
                                          [f"{B}.population.a.opt"],
                                          [vn("a.has_element")]))
                matmul_nodes.append(
                    onnx.helper.make_node("OptionalHasElement",
                                          [f"{B}.population.b.opt"],
                                          [vn("b.has_element")]))
                matmul_nodes.append(
                    onnx.helper.make_node(
                        "And", [vn("a.has_element"),
                                vn("b.has_element")], [vn("has_element")]))
                matmul_nodes.append(
                    onnx.helper.make_node("If", [vn("has_element")],
                                          [output_value],
                                          else_branch=b_merge_else_graph,
                                          then_branch=b_merge_then_graph))
            else:
                assert False, "Must not be happened"
        elif (self.quant_dtype == np.int8) and auto_quantize:
            A_T: str = f"{A}.quantized"
            B_T: str = f"{B}.quantized"
            if transA:
                A_T = f"{A}.quantized.T"
                matmul_nodes.append(
                    onnx.helper.make_node("Transpose", [f"{A}.quantized"],
                                          [A_T]))
            if transB:
                B_T = f"{B}.quantized.T"
                matmul_nodes.append(
                    onnx.helper.make_node("Transpose", [f"{B}.quantized"],
                                          [B_T]))

            matmul_nodes.append(
                onnx.helper.make_node("Mul", [f"{A}.a", f"{B}.a"],
                                      [vn("output_value.a")]))
            mm_value: str = vn("output_value.quantized"
                               ) if dequantize_output else output_value
            matmul_nodes.append(
                onnx.helper.make_node("MatMulInteger",
                                      [A_T, B_T, f"{A}.b", f"{B}.b"],
                                      [mm_value]))
            if dequantize_output:
                matmul_nodes.append(
                    onnx.helper.make_node("DequantizeLinear", [
                        mm_value,
                        vn("output_value.a"),
                    ], [output_value]))
        elif self.args.dtype == ml_dtypes.int4:
            raise NotImplementedError("int4 MatMul is not implemented yet")
        else:
            A_T: str = A
            B_T: str = B
            if transA:
                A_T = f"{A}.T"
                matmul_nodes.append(
                    onnx.helper.make_node("Transpose", [A], [A_T]))
            if transB:
                B_T = f"{B}.T"
                matmul_nodes.append(
                    onnx.helper.make_node("Transpose", [B], [B_T]))
            matmul_nodes.append(
                onnx.helper.make_node("MatMul", [A_T, B_T], [output_value]))
        return matmul_nodes

    def _cast_parameters(
            self, parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.dtype is None:
            return parameters  # No need to cast.
        casted_parameters: dict[str, np.ndarray] = {}
        for k, v in parameters.items():
            if k == "rwkv7.emb.weight":
                # Pre compute ln0 layer normalization
                v = (v - np.mean(v, axis=-1, keepdims=True)) / (
                    np.sqrt(np.var(v, axis=-1, keepdims=True)) + 1e-5
                ) * parameters["rwkv7.blocks.0.ln0.weight"] + parameters[
                    "rwkv7.blocks.0.ln0.bias"]
            elif k.endswith(".weight") and not k == "rwkv7.emb.weight" and (
                    len(v.shape) == 2):
                v = v.T

            if self.quant_dtype == np.dtype(np.int8) and len(
                    v.shape) == 2 and k not in self.quant_filter:
                v_min: np.ndarray = np.min(v)
                v_max: np.ndarray = np.max(v)
                scale: np.ndarray = None
                zero_point: np.ndarray = None
                quant: np.ndarray = None
                if quant is None:
                    scale = (v_max - v_min) / 255.0
                    zero_point = np.clip(np.round(-128.0 - v_min / scale),
                                         -128.0, 127.0)
                    quant = v / scale + zero_point
                assert scale is not None
                assert zero_point is not None
                assert quant is not None
                casted_parameters[f"{k}.a"] = np.astype(scale, self.dtype)
                casted_parameters[f"{k}.b"] = np.astype(zero_point, np.int8)
                casted_parameters[f"{k}.quantized"] = np.astype(
                    np.clip(np.round(quant), -128.0, 127.0), np.int8)
                print(
                    f"Quantized (scale: {scale}, zero_point: {zero_point}): {k}"
                )
            elif self.dtype in (np.float32, np.float16, ml_dtypes.bfloat16):
                casted_parameters[k] = v.astype(self.dtype)
                print(f"Casted: {k}")
            else:
                raise NotImplementedError(
                    f"dtype {self.dtype} is not supported.")

        param_names: Set[str] = casted_parameters.keys()
        for layer_id in range(self.hparam.nlayers):
            # Check casted parameters
            if self.args.optional_state:
                if not f"rwkv7.blocks.{layer_id}.att.ts_state" in param_names:
                    casted_parameters[
                        f"rwkv7.blocks.{layer_id}.att.ts_state"] = np.zeros(
                            (1, 1, self.hparam.C), self.dtype)
                if not f"rwkv7.blocks.{layer_id}.att.time_state" in param_names:
                    casted_parameters[
                        f"rwkv7.blocks.{layer_id}.att.time_state"] = np.zeros(
                            (1, self.hparam.H, self.hparam.N, self.hparam.N),
                            self.wkv_dtype)
                if not f"rwkv7.blocks.{layer_id}.ffn.ts_state" in param_names:
                    casted_parameters[
                        f"rwkv7.blocks.{layer_id}.ffn.ts_state"] = np.zeros(
                            (1, 1, self.hparam.C), self.dtype)
            pass
        return casted_parameters

    def _constants_tensors(self) -> List[onnx.TensorProto]:
        tensors: list[onnx.TensorProto] = []
        tensors.append(
            onnx.numpy_helper.from_array(
                np.array((self.hparam.V, ), dtype=np.int64), "V"))
        tensors.append(
            onnx.numpy_helper.from_array(
                np.array((self.hparam.C, ), dtype=np.int64), "C"))
        tensors.append(
            onnx.numpy_helper.from_array(
                np.array((self.hparam.H, ), dtype=np.int64), "H"))
        tensors.append(
            onnx.numpy_helper.from_array(
                np.array((self.hparam.N, ), dtype=np.int64), "N"))
        if self.hparam.D_de is not None:
            tensors.append(
                onnx.numpy_helper.from_array(
                    np.array((self.hparam.D_de, ), dtype=np.int64), "D"))
        tensors.append(
            onnx.helper.make_tensor("eps", self.hparam.elem_type, [], 1e-12))
        tensors.append(
            onnx.helper.make_tensor("0.0", self.hparam.elem_type, [], 0.0))
        tensors.append(
            onnx.helper.make_tensor("0.0f", onnx.TensorProto.FLOAT, [], 0.0))
        tensors.append(
            onnx.helper.make_tensor("1.0", self.hparam.elem_type, [], 1.0))
        tensors.append(
            onnx.helper.make_tensor("-exp(-0.5)", self.hparam.elem_type, [],
                                    -np.exp(-0.5)))
        tensors.append(
            onnx.helper.make_tensor("INT8_MIN", self.hparam.elem_type, [],
                                    -128.0))
        tensors.append(
            onnx.helper.make_tensor("INT8_MAX", self.hparam.elem_type, [],
                                    127.0))
        tensors.append(
            onnx.helper.make_tensor("INT8_RANGE", self.hparam.elem_type, [],
                                    255.0))
        tensors.append(
            onnx.helper.make_tensor("INT64_MAX", onnx.TensorProto.INT64, [1],
                                    [np.iinfo(np.int64).max]))
        tensors.append(
            onnx.numpy_helper.from_array(np.array((0, ), dtype=np.int64), "0"))
        tensors.append(
            onnx.numpy_helper.from_array(np.array((1, ), dtype=np.int64), "1"))
        tensors.append(
            onnx.numpy_helper.from_array(np.array((3, ), dtype=np.int64), "3"))
        tensors.append(
            onnx.numpy_helper.from_array(np.array((-1, ), dtype=np.int64),
                                         "-1"))
        tensors.append(
            onnx.numpy_helper.from_array(np.array((-2, ), dtype=np.int64),
                                         "-2"))
        tensors.append(
            onnx.numpy_helper.from_array(
                np.array((0.0, 1.0), dtype=np.float32), "(0.0, 1.0)"))

        if self.args.sampling:
            tensors.append(
                onnx.numpy_helper.from_array(
                    np.ones((self.hparam.V, ), dtype=np.bool),
                    "penalty_target"))
            tensors.append(
                onnx.helper.make_tensor("alpha_presence",
                                        onnx.TensorProto.FLOAT, [],
                                        self.args.alpha_presence))
            tensors.append(
                onnx.helper.make_tensor("alpha_frequency",
                                        onnx.TensorProto.FLOAT, [],
                                        self.args.alpha_frequency))
            tensors.append(
                onnx.helper.make_tensor("alpha_decay", onnx.TensorProto.FLOAT,
                                        [], self.args.alpha_decay))
            tensors.append(
                onnx.helper.make_tensor("topp", onnx.TensorProto.FLOAT, [],
                                        self.args.topp))
            tensors.append(
                onnx.helper.make_tensor("topk", onnx.TensorProto.INT64, [1],
                                        [self.args.topk]))
            tensors.append(
                onnx.helper.make_tensor("temperature", onnx.TensorProto.FLOAT,
                                        [], self.args.temperature))
        if self.args.evolution_strategy:
            tensors.append(
                onnx.numpy_helper.from_array(
                    np.array((1, 1, 1, self.hparam.C), dtype=np.int64),
                    "111C"))
        return tensors

    def _rt_constants_nodes(self) -> List[onnx.NodeProto]:
        nodes: list[onnx.NodeProto] = []
        nodes.append(
            onnx.helper.make_node("Shape", ["x"], ["B"], start=0, end=1))
        nodes.append(
            onnx.helper.make_node("Shape", ["x"], ["T"], start=1, end=2))
        nodes.append(
            onnx.helper.make_node("Concat", ["B", "T"], ["BT"], axis=0))
        nodes.append(
            onnx.helper.make_node("Concat", ["B", "1", "C"], ["B1C"], axis=0))
        nodes.append(
            onnx.helper.make_node("Concat", ["B", "T", "C"], ["BTC"], axis=0))
        nodes.append(
            onnx.helper.make_node("Concat", ["B", "T", "H", "N"], ["BTHN"],
                                  axis=0))
        nodes.append(
            onnx.helper.make_node("Concat", ["B", "H", "N", "N"], ["BHNN"],
                                  axis=0))
        nodes.append(onnx.helper.make_node("Mul", ["B", "T"], ["G"]))
        nodes.append(
            onnx.helper.make_node("Concat", ["G", "C"], ["GC"], axis=0))
        if self.hparam.D_de is not None:
            nodes.append(
                onnx.helper.make_node("Concat", ["B", "T", "D"], ["BTD"],
                                      axis=0))
            nodes.append(
                onnx.helper.make_node("Concat", ["B", "T", "1", "D"], ["BT1D"],
                                      axis=0))
            nodes.append(
                onnx.helper.make_node("Concat", ["B", "T", "D", "D"], ["BTDD"],
                                      axis=0))
        if self.args.evolution_strategy:
            P_output: list[onnx.ValueInfoProto] = [
                onnx.helper.make_tensor_value_info("output",
                                                   onnx.TensorProto.INT64, [1])
            ]

            P_else_nodes: list[onnx.NodeProto] = []
            P_else_nodes.append(
                onnx.helper.make_node("Identity", ["1"], ["output"]))
            P_else_graph: onnx.GraphProto = onnx.helper.make_graph(
                P_else_nodes, "P.else.graph", [], P_output)

            P_then_nodes: list[onnx.NodeProto] = []
            P_then_nodes.append(
                onnx.helper.make_node("OptionalGetElement", ["P"],
                                      ["_output"]))
            P_then_nodes.append(
                onnx.helper.make_node("Unsqueeze", ["_output", "0"],
                                      ["output"]))
            P_then_graph: onnx.GraphProto = onnx.helper.make_graph(
                P_then_nodes, "P.then.graph", [], P_output)

            nodes.append(
                onnx.helper.make_node("OptionalHasElement", ["P"],
                                      ["P.has_element"]))
            nodes.append(
                onnx.helper.make_node("If", ["P.has_element"], ["P.element"],
                                      else_branch=P_else_graph,
                                      then_branch=P_then_graph))
            nodes.append(
                onnx.helper.make_node("Concat", ["P.element", "B", "T", "C"],
                                      ["PBTC"],
                                      axis=0))
            nodes.append(
                onnx.helper.make_node("Concat",
                                      ["P.element", "B", "T", "H", "N"],
                                      ["PBTHN"],
                                      axis=0))
            nodes.append(
                onnx.helper.make_node("Concat",
                                      ["P.element", "B", "H", "N", "N"],
                                      ["PBHNN"],
                                      axis=0))
            pass
        else:
            nodes.append(
                onnx.helper.make_node("Concat", ["B", "T", "C"], ["PBTC"],
                                      axis=0))
            nodes.append(
                onnx.helper.make_node("Concat", ["B", "T", "H", "N"],
                                      ["PBTHN"],
                                      axis=0))
            nodes.append(
                onnx.helper.make_node("Concat", ["B", "H", "N", "N"],
                                      ["PBHNN"],
                                      axis=0))
        return nodes

    def _fallback_nodes(
        self,
        opt: str,
        fallback: str,
        shape: str,
        type_proto: onnx.TypeProto,
        value: str,
    ) -> List[onnx.NodeProto]:
        output_value_info: onnx.ValueInfoProto = onnx.helper.make_value_info(
            value, type_proto)

        then_nodes: list[onnx.NodeProto] = []
        then_nodes.append(
            onnx.helper.make_node("OptionalGetElement", [opt], [value]))
        fallback_then_graph: onnx.GraphProto = onnx.helper.make_graph(
            then_nodes, f"{value}.fallback_then_graph", [],
            [output_value_info])

        else_nodes: list[onnx.NodeProto] = []
        else_nodes.append(
            onnx.helper.make_node("Expand", [fallback, shape], [value]))
        fallback_else_graph: onnx.GraphProto = onnx.helper.make_graph(
            else_nodes, f"{value}.fallback_else_graph", [],
            [output_value_info])

        fallback_nodes: list[onnx.NodeProto] = []
        fallback_nodes.append(
            onnx.helper.make_node("OptionalHasElement", [opt],
                                  [f"{opt}.has_element"]))
        fallback_nodes.append(
            onnx.helper.make_node("If", [f"{opt}.has_element"], [value],
                                  else_branch=fallback_else_graph,
                                  then_branch=fallback_then_graph))
        return fallback_nodes

    # Maybe need to cast `value` to fp32.
    def _normalize(self,
                   value: str,
                   output_value: str,
                   axis: str = "-1",
                   eps: str = "eps") -> List[onnx.NodeProto]:
        normalize_nodes: list[onnx.NodeProto] = []
        normalize_nodes.append(
            onnx.helper.make_node("ReduceL2", [value, axis], [f"{value}.l2"]))
        normalize_nodes.append(
            onnx.helper.make_node("Max", [f"{value}.l2", eps],
                                  [f"{value}.l2_eps"]))
        normalize_nodes.append(
            onnx.helper.make_node("Div", [value, f"{value}.l2_eps"],
                                  [output_value]))
        return normalize_nodes

    # x @ W^T, W must be a parameter
    def _linear_transposed(self,
                           x: str,
                           weight: str,
                           output_value: str,
                           auto_quantize: bool = True) -> List[onnx.NodeProto]:
        return self._matmul(x,
                            weight,
                            output_value,
                            auto_quantize=auto_quantize,
                            quantizeA=True,
                            isparamB=True)

    def _time_shift(self, x: str, x_last: str, shifted_value: str,
                    next_value: str) -> List[onnx.NodeProto]:
        time_shift_nodes: list[onnx.NodeProto] = []
        time_shift_nodes.append(
            onnx.helper.make_node("Slice", [x, "0", "-1", "1"],
                                  [f"{x}.shifted_"]))
        time_shift_nodes.append(
            onnx.helper.make_node("Concat", [x_last, f"{x}.shifted_"],
                                  [shifted_value],
                                  axis=1))
        time_shift_nodes.append(
            onnx.helper.make_node("Slice", [x, "-1", "INT64_MAX", "1"],
                                  [next_value]))
        return time_shift_nodes

    def _lerp(self,
              a: str,
              b: str,
              x: str,
              output_value: str,
              is_a_param: bool = False,
              is_b_param: bool = False,
              is_x_param: bool = False) -> List[onnx.NodeProto]:
        vn: Callable[
            [str], str] = lambda name: f"buf.lerp.{hash(output_value)}.{name}"
        lerp_nodes: list[onnx.NodeProto] = []
        if self.args.evolution_strategy and (is_a_param or is_b_param
                                             or is_x_param):
            a_p: str = a
            b_p: str = b
            x_p: str = x

            outputs: list[onnx.ValueInfoProto] = [
                onnx.helper.make_tensor_value_info(vn("output"),
                                                   self.hparam.elem_type,
                                                   ["P", 1, 1, None])
            ]

            merge_else_nodes: list[onnx.NodeProto] = []
            merge_else_nodes.append(
                onnx.helper.make_node("Sub", [b, a], [vn("sub")]))
            merge_else_nodes.append(
                onnx.helper.make_node("Mul", [vn("sub"), x_p], [vn("mul")]))
            merge_else_nodes.append(
                onnx.helper.make_node("Add", [a, vn("mul")], [vn("output")]))
            merge_else_graph: onnx.GraphProto = onnx.helper.make_graph(
                merge_else_nodes, f"graph.lerp.{hash(output_value)}.else", [],
                outputs)

            merge_then_nodes: list[onnx.NodeProto] = []
            vn_a: str = a
            vn_b: str = b
            vn_x: str = x
            if is_a_param:
                vn_a = vn("a")
                merge_then_nodes.append(
                    onnx.helper.make_node("OptionalGetElement",
                                          [f"{a}.population.opt"],
                                          [vn("a.opt")]))
                merge_then_nodes.append(
                    onnx.helper.make_node("Add", [a, vn("a.opt")], [vn_a]))
            if is_b_param:
                vn_b = vn("b")
                merge_then_nodes.append(
                    onnx.helper.make_node("OptionalGetElement",
                                          [f"{b}.population.opt"],
                                          [vn("b.opt")]))
                merge_then_nodes.append(
                    onnx.helper.make_node("Add", [b, vn("b.opt")], [vn_b]))
            if is_x_param:
                vn_x = vn("x")
                merge_then_nodes.append(
                    onnx.helper.make_node("OptionalGetElement",
                                          [f"{x}.population.opt"],
                                          [vn("x.opt")]))
                merge_then_nodes.append(
                    onnx.helper.make_node("Add", [x, vn("x.opt")], [vn_x]))
            merge_then_nodes.append(
                onnx.helper.make_node("Sub", [vn_b, vn_a], [vn("sub")]))
            merge_then_nodes.append(
                onnx.helper.make_node("Mul", [vn("sub"), x], [vn("mul")]))
            merge_then_nodes.append(
                onnx.helper.make_node("Add", [vn_a, vn("mul")],
                                      [vn("output")]))
            merge_then_graph: onnx.GraphProto = onnx.helper.make_graph(
                merge_then_nodes, f"graph.lerp.{hash(output_value)}.then", [],
                outputs)

            has_elements: list[str] = []
            if is_a_param:
                lerp_nodes.append(
                    onnx.helper.make_node("OptionalHasElement",
                                          [f"{a}.population.opt"],
                                          [vn("a.has_element")]))
                has_elements.append(vn("a.has_element"))
            if is_b_param:
                lerp_nodes.append(
                    onnx.helper.make_node("OptionalHasElement",
                                          [f"{b}.population.opt"],
                                          [vn("b.has_element")]))
                has_elements.append(vn("b.has_element"))
            if is_x_param:
                lerp_nodes.append(
                    onnx.helper.make_node("OptionalHasElement",
                                          [f"{x}.population.opt"],
                                          [vn("x.has_element")]))
                has_elements.append(vn("x.has_element"))

            assert len(has_elements) > 0
            cond: str = vn("has_element")
            match len(has_elements):
                case 1:
                    cond = has_elements[0]
                case 2:
                    lerp_nodes.append(
                        onnx.helper.make_node(
                            "And", [has_elements[0], has_elements[1]],
                            [vn("has_element")]))
                case 3:
                    lerp_nodes.append(
                        onnx.helper.make_node(
                            "And", [has_elements[0], has_elements[1]],
                            [vn("_has_element")]))
                    lerp_nodes.append(
                        onnx.helper.make_node(
                            "And", [vn("_has_element"), has_elements[2]],
                            [vn("has_element")]))

            lerp_nodes.append(
                onnx.helper.make_node("If", [cond], [output_value],
                                      else_branch=merge_else_graph,
                                      then_branch=merge_then_graph))
        else:
            lerp_nodes.append(onnx.helper.make_node("Sub", [b, a],
                                                    [vn("sub")]))
            lerp_nodes.append(
                onnx.helper.make_node("Mul", [vn("sub"), x], [vn("mul")]))
            lerp_nodes.append(
                onnx.helper.make_node("Add", [a, vn("mul")], [output_value]))
        return lerp_nodes

    # lerp(a, b, x) = a + (b - a) * x
    # ba = b - a
    def _lerp_pre_computed(self,
                           a: str,
                           ba: str,
                           x: str,
                           output_value: str,
                           is_x_param: bool = False) -> List[onnx.NodeProto]:
        vn: Callable[
            [str],
            str] = lambda name: f"buf.lerp_pre_computed.{hash(output_value)}.{name}"
        lerp_nodes: list[onnx.NodeProto] = []
        if self.args.evolution_strategy and is_x_param:
            merge_outputs: list[onnx.ValueInfoProto] = [
                onnx.helper.make_tensor_value_info("output",
                                                   self.hparam.elem_type,
                                                   ["P", None, None, None])
            ]

            merge_else_nodes: list[onnx.NodeProto] = []
            merge_else_nodes.append(
                onnx.helper.make_node("Mul", [ba, x], [vn("mul")]))
            merge_else_nodes.append(
                onnx.helper.make_node("Add", [a, vn("mul")], ["output"]))
            merge_else_graph: onnx.GraphProto = onnx.helper.make_graph(
                merge_else_nodes,
                f"graph.lerp_pre_computed.{hash(output_value)}.else", [],
                merge_outputs)

            merge_then_nodes: list[onnx.NodeProto] = []
            merge_then_nodes.append(
                onnx.helper.make_node("OptionalGetElement",
                                      [f"{x}.population.opt"],
                                      [vn("x.population")]))
            merge_then_nodes.append(
                onnx.helper.make_node("Add", [x, vn("x.population")],
                                      [vn("x")]))
            merge_then_nodes.append(
                onnx.helper.make_node("Mul", [ba, vn("x")], [vn("mul")]))
            merge_then_nodes.append(
                onnx.helper.make_node("Add", [a, vn("mul")], ["output"]))
            merge_then_graph: onnx.GraphProto = onnx.helper.make_graph(
                merge_then_nodes,
                f"graph.lerp_pre_computed.{hash(output_value)}.then", [],
                merge_outputs)

            lerp_nodes.append(
                onnx.helper.make_node("OptionalHasElement",
                                      [f"{x}.population.opt"],
                                      [vn("x.has_element")]))
            lerp_nodes.append(
                onnx.helper.make_node("If", [vn("x.has_element")],
                                      [output_value],
                                      else_branch=merge_else_graph,
                                      then_branch=merge_then_graph))
        else:
            lerp_nodes.append(
                onnx.helper.make_node("Mul", [ba, x], [vn("mul")]))
            lerp_nodes.append(
                onnx.helper.make_node("Add", [a, vn("mul")], [output_value]))
        return lerp_nodes

    # f(x @ A) @ B + bias, A, B, and bias (if provided) must be parameters.
    def _loramlp(self, x: str, A: str, B: str, bias: str | None, f: str,
                 output_value: str) -> List[onnx.NodeProto]:
        vn: Callable[
            [str],
            str] = lambda name: f"buf.loramlp.{hash(output_value)}.{name}"
        loramlp_nodes: list[onnx.NodeProto] = []
        loramlp_nodes += self._matmul(x,
                                      A,
                                      vn("A"),
                                      quantizeA=True,
                                      isparamB=True)
        loramlp_nodes.append(onnx.helper.make_node(f, [vn("A")], [vn("f")]))
        if bias is None:
            loramlp_nodes += self._matmul(vn("f"),
                                          B,
                                          output_value,
                                          quantizeA=True,
                                          isparamB=True)
        else:
            loramlp_nodes += self._matmul(vn("f"),
                                          B,
                                          vn("B"),
                                          quantizeA=True,
                                          isparamB=True)
            loramlp_nodes.append(
                onnx.helper.make_node("Add", [vn("B"), bias], [output_value]))
        return loramlp_nodes

    def _wkv(self, wkv_state: str, r: str, w: str, k: str, v: str, a: str,
             b: str, output_state: str,
             output_value: str) -> List[onnx.NodeProto]:
        if self.wkv_graph is None:
            P: list[str] = ["P"] if self.args.evolution_strategy else []
            wkv_graph_inputs: list[onnx.ValueInfoProto] = []
            wkv_graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    "wkv_state", self.hparam.wkv_elem_type,
                    P + ["B", self.hparam.H, self.hparam.N, self.hparam.N]))
            wkv_graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    "w", self.hparam.wkv_elem_type,
                    P + ["B", self.hparam.H, 1, self.hparam.N]))
            wkv_graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    "a", self.hparam.wkv_elem_type,
                    P + ["B", self.hparam.H, self.hparam.N]))
            wkv_graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    "b", self.hparam.wkv_elem_type,
                    P + ["B", self.hparam.H, self.hparam.N]))
            wkv_graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    "v", self.hparam.wkv_elem_type,
                    P + ["B", self.hparam.H, self.hparam.N]))
            wkv_graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    "k", self.hparam.wkv_elem_type,
                    P + ["B", self.hparam.H, self.hparam.N]))

            wkv_graph_outputs: list[onnx.ValueInfoProto] = []
            wkv_graph_outputs.append(
                onnx.helper.make_tensor_value_info(
                    "wkv_next", self.hparam.wkv_elem_type,
                    P + ["B", self.hparam.H, self.hparam.N, self.hparam.N]))
            wkv_graph_outputs.append(
                onnx.helper.make_tensor_value_info(
                    "wkv_out", self.hparam.wkv_elem_type,
                    P + ["B", self.hparam.H, self.hparam.N, self.hparam.N]))

            wkv_graph_nodes: list[onnx.NodeProto] = []
            wkv_graph_nodes.append(
                onnx.helper.make_node("Einsum", ["wkv_state", "a"],
                                      ["wkv_state.a"],
                                      equation="...ij,...j->...i"))
            wkv_graph_nodes.append(
                onnx.helper.make_node("Einsum", ["wkv_state.a", "b"],
                                      ["wkv_state.a.b"],
                                      equation="...i,...j->...ij"))
            wkv_graph_nodes.append(
                onnx.helper.make_node("Einsum", ["v", "k"], ["v.k"],
                                      equation="...i,...j->...ij"))
            wkv_graph_nodes.append(
                onnx.helper.make_node("Mul", ["wkv_state", "w"],
                                      ["wkv_state.w"]))
            wkv_graph_nodes.append(
                onnx.helper.make_node("Sub", ["wkv_state.w", "wkv_state.a.b"],
                                      ["wkv_state.w.a.b"]))
            wkv_graph_nodes.append(
                onnx.helper.make_node("Add", ["wkv_state.w.a.b", "v.k"],
                                      ["wkv_next"]))
            wkv_graph_nodes.append(
                onnx.helper.make_node("Identity", ["wkv_next"], ["wkv_out"]))

            self.wkv_graph = onnx.helper.make_graph(wkv_graph_nodes,
                                                    "wkv_graph",
                                                    wkv_graph_inputs,
                                                    wkv_graph_outputs)

        wkv_nodes: list[onnx.NodeProto] = []
        wkv_nodes.append(
            onnx.helper.make_node("Cast", [r], [f"{r}.casted"],
                                  to=self.hparam.wkv_elem_type))
        wkv_nodes.append(
            onnx.helper.make_node("Cast", [w], [f"{w}._casted"],
                                  to=self.hparam.wkv_elem_type))
        wkv_nodes.append(
            onnx.helper.make_node("Unsqueeze", [f"{w}._casted", "-2"],
                                  [f"{w}.casted"]))
        wkv_nodes.append(
            onnx.helper.make_node("Cast", [k], [f"{k}.casted"],
                                  to=self.hparam.wkv_elem_type))
        wkv_nodes.append(
            onnx.helper.make_node("Cast", [v], [f"{v}.casted"],
                                  to=self.hparam.wkv_elem_type))
        wkv_nodes.append(
            onnx.helper.make_node("Cast", [a], [f"{a}.casted"],
                                  to=self.hparam.wkv_elem_type))
        wkv_nodes.append(
            onnx.helper.make_node("Cast", [b], [f"{b}.casted"],
                                  to=self.hparam.wkv_elem_type))

        assert isinstance(self.wkv_graph, onnx.GraphProto)
        wkv_nodes.append(
            onnx.helper.make_node(
                "Scan", [
                    wkv_state, f"{w}.casted", f"{a}.casted", f"{b}.casted",
                    f"{v}.casted", f"{k}.casted"
                ], [output_state, f"_{output_value}.casted"],
                body=self.wkv_graph,
                num_scan_inputs=5,
                scan_input_axes=[2] *
                5 if self.args.evolution_strategy else [1] * 5,
                scan_output_axes=[2] if self.args.evolution_strategy else [1]))

        wkv_nodes.append(
            onnx.helper.make_node("Einsum",
                                  [f"_{output_value}.casted", f"{r}.casted"],
                                  [f"{output_value}.casted"],
                                  equation="...ij,...j->...i"))
        wkv_nodes.append(
            onnx.helper.make_node("Cast", [f"{output_value}.casted"],
                                  [output_value],
                                  to=self.hparam.elem_type))
        return wkv_nodes

    def _tmix(self, layer_id: int, x: str, v_first: str, x_last: str,
              wkv_state: str, output: str, x_next: str,
              wkv_state_next: str) -> List[onnx.NodeProto]:
        vn: Callable[
            [str],
            str] = lambda name: f"buf.rwkv7.blocks.{layer_id}.att.{name}"
        pn: Callable[[str],
                     str] = lambda name: f"rwkv7.blocks.{layer_id}.att.{name}"

        tmix_nodes: list[onnx.NodeProto] = []
        tmix_nodes += self._time_shift(x, x_last, vn("x.shifted"), x_next)
        tmix_nodes.append(
            onnx.helper.make_node("Sub", [vn("x.shifted"), x],
                                  [vn("x.lerp_pre_compute")]))
        tmix_nodes += self._lerp_pre_computed(x,
                                              vn("x.lerp_pre_compute"),
                                              pn("x_r"),
                                              vn("xr"),
                                              is_x_param=True)
        tmix_nodes += self._lerp_pre_computed(x,
                                              vn("x.lerp_pre_compute"),
                                              pn("x_w"),
                                              vn("xw"),
                                              is_x_param=True)
        tmix_nodes += self._lerp_pre_computed(x,
                                              vn("x.lerp_pre_compute"),
                                              pn("x_k"),
                                              vn("xk"),
                                              is_x_param=True)
        tmix_nodes += self._lerp_pre_computed(x,
                                              vn("x.lerp_pre_compute"),
                                              pn("x_v"),
                                              vn("xv"),
                                              is_x_param=True)
        tmix_nodes += self._lerp_pre_computed(x,
                                              vn("x.lerp_pre_compute"),
                                              pn("x_a"),
                                              vn("xa"),
                                              is_x_param=True)
        tmix_nodes += self._lerp_pre_computed(x,
                                              vn("x.lerp_pre_compute"),
                                              pn("x_g"),
                                              vn("xg"),
                                              is_x_param=True)

        tmix_nodes += self._linear_transposed(vn("xr"),
                                              pn("receptance.weight"), vn("r"))
        tmix_nodes += self._loramlp(vn("xw"), pn("w1"), pn("w2"), pn("w0"),
                                    "Tanh", vn("_w"))
        tmix_nodes += self._linear_transposed(vn("xk"), pn("key.weight"),
                                              vn("_k"))
        tmix_nodes += self._linear_transposed(vn("xv"), pn("value.weight"),
                                              vn("_v"))
        tmix_nodes += self._loramlp(vn("xa"), pn("a1"), pn("a2"), pn("a0"),
                                    "Identity", vn("_a"))
        tmix_nodes.append(
            onnx.helper.make_node("Sigmoid", [vn("_a")], [vn("a")]))
        tmix_nodes += self._loramlp(vn("xg"), pn("g1"), pn("g2"), None,
                                    "Sigmoid", vn("g"))
        tmix_nodes += self._mul(vn("_k"),
                                pn("k_k"),
                                vn("_kk"),
                                is_b_param=True)
        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [vn("_kk"), "PBTHN"],
                                  [vn("_kk.head")]))
        tmix_nodes += self._normalize(vn("_kk.head"), vn("kk.head"))
        tmix_nodes += self._lerp("1.0",
                                 vn("a"),
                                 pn("k_a"),
                                 vn("a.lerp"),
                                 is_x_param=True)
        tmix_nodes.append(
            onnx.helper.make_node("Mul", [vn("_k"), vn("a.lerp")], [vn("k")]))
        tmix_nodes.append(
            onnx.helper.make_node("Sigmoid", [vn("_w")], [vn("w.sigmoid")]))
        tmix_nodes.append(
            onnx.helper.make_node("Mul",
                                  ["-exp(-0.5)", vn("w.sigmoid")],
                                  [vn("w.sigmoid.clipped")]))
        tmix_nodes.append(
            onnx.helper.make_node("Exp", [vn("w.sigmoid.clipped")], [vn("w")]))

        v: str = vn("_v")
        if layer_id == 0:
            tmix_nodes.append(
                onnx.helper.make_node("Identity", [vn("_v")], [v_first]))
        else:
            tmix_nodes += self._loramlp(vn("xv"), pn("v1"), pn("v2"), pn("v0"),
                                        "Identity", vn("_vg"))
            tmix_nodes.append(
                onnx.helper.make_node("Sigmoid", [vn("_vg")], [vn("vg")]))
            tmix_nodes.append(
                onnx.helper.make_node("Sub", [v_first, vn("_v")],
                                      [vn("_v.lerp_pre_compute")]))
            tmix_nodes += self._lerp_pre_computed(vn("_v"),
                                                  vn("_v.lerp_pre_compute"),
                                                  vn("vg"), vn("v"))
            v = vn("_v")

        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [vn("r"), "PBTHN"],
                                  [vn("r.head")]))
        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [vn("w"), "PBTHN"],
                                  [vn("w.head")]))
        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [vn("k"), "PBTHN"],
                                  [vn("k.head")]))
        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [v, "PBTHN"], [vn("v.head")]))
        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [vn("a"), "PBTHN"],
                                  [vn("a.head")]))
        tmix_nodes.append(
            onnx.helper.make_node("Mul",
                                  [vn("kk.head"), vn("a.head")],
                                  [vn("b.head")]))

        tmix_nodes += self._wkv(wkv_state, vn("r.head"), vn("w.head"),
                                vn("k.head"), vn("v.head"), vn("kk.head"),
                                vn("b.head"), wkv_state_next, vn("wkv_out"))

        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [vn("wkv_out"), "GC"],
                                  [vn("wkv_out.group")]))
        tmix_nodes += self._group_norm(vn("wkv_out.group"),
                                       pn("ln_x.weight"),
                                       pn("ln_x.bias"),
                                       vn("wkv_out.gn.group"),
                                       epsilon=64e-5,
                                       num_groups=self.hparam.H)
        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [vn("wkv_out.gn.group"), "PBTC"],
                                  [vn("wkv_out.gn")]))
        tmix_nodes += self._mul(pn("r_k"),
                                vn("k.head"),
                                vn("u.k.head"),
                                is_a_param=True)
        tmix_nodes.append(
            onnx.helper.make_node("Einsum",
                                  [vn("r.head"), vn("u.k.head")],
                                  [vn("_u.rk.head")],
                                  equation="...i,...i->..."))
        tmix_nodes.append(
            onnx.helper.make_node("Unsqueeze", [vn("_u.rk.head"), "-1"],
                                  [vn("u.rk.head")]))
        tmix_nodes.append(
            onnx.helper.make_node(
                "Mul", [vn("u.rk.head"), vn("v.head")], [vn("u.head")]))
        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [vn("u.head"), "PBTC"],
                                  [vn("rkv")]))
        tmix_nodes.append(
            onnx.helper.make_node(
                "Add", [vn("wkv_out.gn"), vn("rkv")], [vn("_rwkv")]))
        tmix_nodes.append(
            onnx.helper.make_node("Mul", [vn("_rwkv"), vn("g")], [vn("rwkv")]))
        tmix_nodes += self._linear_transposed(vn("rwkv"), pn("output.weight"),
                                              output)
        return tmix_nodes

    def _cmix(self, layer_id: int, x: str, x_last: str, output: str,
              x_next: str) -> List[onnx.NodeProto]:
        vn: Callable[
            [str],
            str] = lambda name: f"buf.rwkv7.blocks.{layer_id}.ffn..{name}"
        pn: Callable[[str],
                     str] = lambda name: f"rwkv7.blocks.{layer_id}.ffn.{name}"

        cmix_nodes: list[onnx.NodeProto] = []
        cmix_nodes += self._time_shift(x, x_last, vn("x.shifted"), x_next)
        cmix_nodes.append(
            onnx.helper.make_node("Sub", [vn("x.shifted"), x],
                                  [vn("x.lerp_pre_compute")]))
        cmix_nodes += self._lerp_pre_computed(x,
                                              vn("x.lerp_pre_compute"),
                                              pn("x_k"),
                                              vn("x.lerp"),
                                              is_x_param=True)
        cmix_nodes += self._linear_transposed(vn("x.lerp"), pn("key.weight"),
                                              vn("key"))
        cmix_nodes.append(
            onnx.helper.make_node("Relu", [vn("key")], [vn("relu")]))
        cmix_nodes.append(
            onnx.helper.make_node("Mul", [vn("relu"), vn("relu")],
                                  [vn("relu_squared")]))

        cmix_nodes += self._linear_transposed(vn("relu_squared"),
                                              pn("value.weight"), output)
        return cmix_nodes

    def _sampling(self, logit: str, occurence: str | None, output: str,
                  occurence_next: str) -> List[onnx.NodeProto]:
        sn: Callable[[str], str] = lambda x: f"sampling.{x}"

        if self.sampling_else is None and self.sampling_then is None:
            P: list[str] = ["P"] if self.args.evolution_strategy else []
            sampling_graph_outputs: list[onnx.ValueInfoProto] = []
            y_type_proto: onnx.TypeProto = onnx.helper.make_tensor_type_proto(
                onnx.TensorProto.INT64, P + ["B"])
            occurence_next_type_proto: onnx.TypeProto = onnx.helper.make_tensor_type_proto(
                onnx.TensorProto.FLOAT, P + ["B", self.hparam.V])
            sampling_graph_outputs.append(
                onnx.helper.make_value_info(
                    "y", onnx.helper.make_optional_type_proto(y_type_proto)))
            sampling_graph_outputs.append(
                onnx.helper.make_value_info(
                    "occurence_next",
                    onnx.helper.make_optional_type_proto(
                        occurence_next_type_proto)))

            sampling_else_nodes: list[onnx.NodeProto] = []
            sampling_else_nodes.append(
                onnx.helper.make_node("Optional", [], ["y"],
                                      type=y_type_proto))
            sampling_else_nodes.append(
                onnx.helper.make_node("Optional", [], ["occurence_next"],
                                      type=occurence_next_type_proto))
            self.sampling_else = onnx.helper.make_graph(
                sampling_else_nodes, "sampling_else", [],
                sampling_graph_outputs)

            sampling_then_nodes: list[onnx.NodeProto] = []
            sampling_then_nodes.append(
                onnx.helper.make_node("OptionalGetElement", [occurence],
                                      [sn("occurence")]))
            sampling_then_nodes.append(
                onnx.helper.make_node("Greater", [sn("occurence"), "0.0f"],
                                      [sn("presence")]))
            sampling_then_nodes.append(
                onnx.helper.make_node(
                    "Where", [sn("presence"), "alpha_presence", "0.0f"],
                    [sn("presence_penalty")]))
            sampling_then_nodes.append(
                onnx.helper.make_node("Mul",
                                      [sn("occurence"), "alpha_frequency"],
                                      [sn("frequency_penalty")]))
            sampling_then_nodes.append(
                onnx.helper.make_node(
                    "Add", [sn("presence_penalty"),
                            sn("frequency_penalty")], [sn("penalty")]))
            sampling_then_nodes.append(
                onnx.helper.make_node("Squeeze", [logit, "1"], [sn("_logit")]))
            sampling_then_nodes.append(
                onnx.helper.make_node(
                    "Sub", [sn("_logit"), sn("penalty")], [sn("logit")]))
            sampling_then_nodes.append(
                onnx.helper.make_node("Softmax", [sn("logit")],
                                      [sn("logit_prob")]))

            sampling_then_nodes.append(
                onnx.helper.make_node(
                    "TopK", [sn("logit_prob"), "V"],
                    [sn("logit_prob_sorted"),
                     sn("logit_prob_idx")]))

            sampling_then_nodes.append(
                onnx.helper.make_node("CumSum",
                                      [sn("logit_prob_sorted"), "-1"],
                                      [sn("logit_prob_cumsum")],
                                      exclusive=1))
            sampling_then_nodes.append(
                onnx.helper.make_node("GreaterOrEqual",
                                      [sn("logit_prob_cumsum"), "topp"],
                                      [sn("_topp")]))
            sampling_then_nodes.append(
                onnx.helper.make_node(
                    "Where", [sn("_topp"), "0.0f",
                              sn("logit_prob_sorted")], [sn("topp")]))
            sampling_then_nodes.append(
                onnx.helper.make_node("ReduceSum", [sn("topp"), "-1"],
                                      [sn("topp_sum")]))
            sampling_then_nodes.append(
                onnx.helper.make_node("Div",
                                      [sn("topp"), sn("topp_sum")],
                                      [sn("topp_scaled")]))

            topk_scaled: str = sn("topp_scaled")
            if self.args.topk > 0:
                topk_scaled = sn("topk_scaled")

                sampling_then_nodes.append(
                    onnx.helper.make_node(
                        "TopK", [sn("topp_scaled"), "topk"],
                        [sn("topk"), sn("topk_idx")]))
                sampling_then_nodes.append(
                    onnx.helper.make_node("ReduceSum", [sn("topk"), "-1"],
                                          [sn("topk_sum")]))
                sampling_then_nodes.append(
                    onnx.helper.make_node(
                        "Div", [sn("topk"), sn("topk_sum")], [topk_scaled]))

            temp_scaled: str = topk_scaled
            if self.args.temperature != 1.0:
                temp_scaled = sn("temp_scaled")
                sampling_then_nodes.append(
                    onnx.helper.make_node("Reciprocal", ["temperature"],
                                          [sn("temperature")]))
                sampling_then_nodes.append(
                    onnx.helper.make_node(
                        "Pow", [topk_scaled, sn("temperature")],
                        [sn("logit_temp")]))
                sampling_then_nodes.append(
                    onnx.helper.make_node("ReduceSum",
                                          [sn("logit_temp"), "-1"],
                                          [sn("logit_temp_sum")]))
                sampling_then_nodes.append(
                    onnx.helper.make_node(
                        "Div", [sn("logit_temp"),
                                sn("logit_temp_sum")], [temp_scaled]))

            sampling_then_nodes.append(
                onnx.helper.make_node("Log", [temp_scaled], [sn("log_prob")]))
            sampling_then_nodes.append(
                onnx.helper.make_node("Multinomial", [sn("log_prob")],
                                      [sn("_sampled_idx")],
                                      dtype=onnx.TensorProto.INT64))
            sampling_then_nodes.append(
                onnx.helper.make_node(
                    "GatherElements",
                    [sn("logit_prob_idx"),
                     sn("_sampled_idx")], [sn("sampled_idx")],
                    axis=1))
            sampling_then_nodes.append(
                onnx.helper.make_node("Squeeze", [sn("sampled_idx"), "1"],
                                      ["_y"]))
            sampling_then_nodes.append(
                onnx.helper.make_node("Optional", ["_y"], ["y"]))

            sampling_then_nodes.append(
                onnx.helper.make_node("Mul", [sn("occurence"), "alpha_decay"],
                                      [sn("occurence_decay")]))
            sampling_then_nodes.append(
                onnx.helper.make_node("OneHot", ["_y", "V", "(0.0, 1.0)"],
                                      [sn("onehot")]))
            sampling_then_nodes.append(
                onnx.helper.make_node(
                    "Where",
                    ["penalty_target", sn("onehot"), "0.0f"],
                    [sn("occurence_generated")]))
            sampling_then_nodes.append(
                onnx.helper.make_node(
                    "Add", [sn("occurence_decay"),
                            sn("occurence_generated")],
                    [sn("occurence_next")]))
            sampling_then_nodes.append(
                onnx.helper.make_node("Optional", [sn("occurence_next")],
                                      ["occurence_next"]))

            self.sampling_then = onnx.helper.make_graph(
                sampling_then_nodes, "sampling_then", [],
                sampling_graph_outputs)

        assert self.sampling_else is not None and self.sampling_then is not None

        sampling_nodes: list[onnx.NodeProto] = []
        sampling_nodes.append(
            onnx.helper.make_node("Equal", ["T", "1"], [sn("is_generation")]))
        sampling_nodes.append(
            onnx.helper.make_node("OptionalHasElement", [occurence],
                                  [sn("occurence_available")]))
        sampling_nodes.append(
            onnx.helper.make_node(
                "And", [sn("is_generation"),
                        sn("occurence_available")], [sn("do_sampling")]))
        sampling_nodes.append(
            onnx.helper.make_node("If", [sn("do_sampling")],
                                  [output, occurence_next],
                                  else_branch=self.sampling_else,
                                  then_branch=self.sampling_then))
        return sampling_nodes

    def generate(self, parameters: Dict[str, np.ndarray]) -> onnx.ModelProto:
        model_nodes: list[onnx.NodeProto] = []
        model_inputs: list[onnx.ValueInfoProto] = []
        model_outputs: list[onnx.ValueInfoProto] = []
        P_shape: list[str] = ["P"] if self.args.evolution_strategy else []
        P: str = "P" if self.args.evolution_strategy else ""

        model_inputs.append(
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.INT64,
                                               ["B", "T"]))
        model_nodes += self._rt_constants_nodes()
        if self.args.evolution_strategy:
            emb_population_outputs: list[onnx.ValueInfoProto] = []
            emb_population_outputs.append(
                onnx.helper.make_tensor_value_info(
                    "x.0", self.hparam.elem_type,
                    ["P", "B", "T", self.hparam.C]))

            emb_population_else_nodes: list[onnx.NodeProto] = []

            emb_population_else_nodes.append(
                onnx.helper.make_node("Unsqueeze", ["_x.0", "0"], ["x.0"]))
            emb_population_else: onnx.GraphProto = onnx.helper.make_graph(
                emb_population_else_nodes, "emb_population_else", [],
                emb_population_outputs)
            emb_population_then_nodes: list[onnx.NodeProto] = []
            emb_population_then_nodes.append(
                onnx.helper.make_node("OptionalGetElement",
                                      ["rwkv7.emb.weight.population.a.opt"],
                                      ["rwkv7.emb.weight.population.a"]))
            emb_population_then_nodes.append(
                onnx.helper.make_node("OptionalGetElement",
                                      ["rwkv7.emb.weight.population.b.opt"],
                                      ["rwkv7.emb.weight.population.b"]))
            emb_population_then_nodes.append(
                onnx.helper.make_node("Gather",
                                      ["rwkv7.emb.weight.population.a", "x"],
                                      ["_x.0.a"],
                                      axis=-2))
            emb_population_then_nodes.append(
                onnx.helper.make_node(
                    "MatMul", ["_x.0.a", "rwkv7.emb.weight.population.b"],
                    ["_x.0.population"]))
            emb_population_then_nodes.append(
                onnx.helper.make_node("Add", ["_x.0", "_x.0.population"],
                                      ["x.0"]))
            emb_population_then: onnx.GraphProto = onnx.helper.make_graph(
                emb_population_then_nodes, "emb_population_then", [],
                emb_population_outputs)

            model_nodes.append(
                onnx.helper.make_node("Gather", ["rwkv7.emb.weight", "x"],
                                      ["_x.0"]))
            model_nodes.append(
                onnx.helper.make_node(
                    "OptionalHasElement",
                    ["rwkv7.emb.weight.population.a.opt"],
                    ["rwkv7.emb.weight.population.a.has_element"]))
            model_nodes.append(
                onnx.helper.make_node(
                    "OptionalHasElement",
                    ["rwkv7.emb.weight.population.b.opt"],
                    ["rwkv7.emb.weight.population.b.has_element"]))
            model_nodes.append(
                onnx.helper.make_node("And", [
                    "rwkv7.emb.weight.population.a.has_element",
                    "rwkv7.emb.weight.population.b.has_element"
                ], ["rwkv7.emb.weight.population.has_element"]))
            model_nodes.append(
                onnx.helper.make_node(
                    "If", ["rwkv7.emb.weight.population.has_element"], ["x.0"],
                    else_branch=emb_population_else,
                    then_branch=emb_population_then))
        else:
            model_nodes.append(
                onnx.helper.make_node("Gather", ["rwkv7.emb.weight", "x"],
                                      ["x.0"]))

        nlayers: int = 0
        while f"rwkv7.blocks.{nlayers}.ln1.weight" in parameters.keys():
            bn: Callable[[str], str] = lambda n: f"rwkv7.blocks.{nlayers}.{n}"
            tn: Callable[[str],
                         str] = lambda n: f"rwkv7.blocks.{nlayers}.att.{n}"
            cn: Callable[[str],
                         str] = lambda n: f"rwkv7.blocks.{nlayers}.ffn.{n}"

            x_tmix_type: onnx.TypeProto = onnx.helper.make_tensor_type_proto(
                self.hparam.elem_type, P_shape + ["B", 1, self.hparam.C])
            wkv_state_type: onnx.TypeProto = onnx.helper.make_tensor_type_proto(
                self.hparam.wkv_elem_type,
                P_shape + ["B", self.hparam.H, self.hparam.N, self.hparam.N])
            x_cmix_type: onnx.TypeProto = onnx.helper.make_tensor_type_proto(
                self.hparam.elem_type, P_shape + ["B", 1, self.hparam.C])

            tmix_last_value_info: onnx.ValueInfoProto
            wkv_state_value_info: onnx.ValueInfoProto
            cmix_last_value_info: onnx.ValueInfoProto
            tmix_next_value_info: onnx.ValueInfoProto
            wkv_next_value_info: onnx.ValueInfoProto
            cmix_next_value_info: onnx.ValueInfoProto
            if self.args.optional_state:
                tmix_last_value_info = onnx.helper.make_value_info(
                    f"x_tmix_last_{nlayers}_opt",
                    onnx.helper.make_optional_type_proto(x_tmix_type))
                wkv_state_value_info = onnx.helper.make_value_info(
                    f"wkv_state_{nlayers}_opt",
                    onnx.helper.make_optional_type_proto(wkv_state_type))
                cmix_last_value_info = onnx.helper.make_value_info(
                    f"x_cmix_last_{nlayers}_opt",
                    onnx.helper.make_optional_type_proto(x_cmix_type))

                model_nodes += self._fallback_nodes(
                    f"x_tmix_last_{nlayers}_opt",
                    f"rwkv7.blocks.{nlayers}.att.ts_state", P + "B1C",
                    x_tmix_type, f"x_tmix_last_{nlayers}")
                model_nodes += self._fallback_nodes(
                    f"wkv_state_{nlayers}_opt",
                    f"rwkv7.blocks.{nlayers}.att.time_state", P + "BHNN",
                    wkv_state_type, f"wkv_state_{nlayers}")
                model_nodes += self._fallback_nodes(
                    f"x_cmix_last_{nlayers}_opt",
                    f"rwkv7.blocks.{nlayers}.ffn.ts_state", P + "B1C",
                    x_cmix_type, f"x_cmix_last_{nlayers}")
            else:
                tmix_last_value_info = onnx.helper.make_value_info(
                    f"x_tmix_last_{nlayers}", x_tmix_type)
                wkv_state_value_info = onnx.helper.make_value_info(
                    f"wkv_state_{nlayers}", wkv_state_type)
                cmix_last_value_info = onnx.helper.make_value_info(
                    f"x_cmix_last_{nlayers}", x_cmix_type)
            tmix_next_value_info = onnx.helper.make_value_info(
                f"x_tmix_next_{nlayers}", x_tmix_type)
            wkv_next_value_info = onnx.helper.make_value_info(
                f"wkv_next_{nlayers}", wkv_state_type)
            cmix_next_value_info = onnx.helper.make_value_info(
                f"x_cmix_next_{nlayers}", x_cmix_type)

            model_inputs.append(tmix_last_value_info)
            model_inputs.append(wkv_state_value_info)
            model_inputs.append(cmix_last_value_info)

            model_outputs.append(tmix_next_value_info)
            model_outputs.append(wkv_next_value_info)
            model_outputs.append(cmix_next_value_info)

            model_nodes += self._layer_norm(f"x.{nlayers}", bn("ln1.weight"),
                                            bn("ln1.bias"), f"x.{nlayers}.ln1")
            model_nodes += self._tmix(nlayers, f"x.{nlayers}.ln1", "v_first",
                                      f"x_tmix_last_{nlayers}",
                                      f"wkv_state_{nlayers}",
                                      f"x.{nlayers}._tmix",
                                      f"x_tmix_next_{nlayers}",
                                      f"wkv_next_{nlayers}")
            model_nodes.append(
                onnx.helper.make_node("Add",
                                      [f"x.{nlayers}", f"x.{nlayers}._tmix"],
                                      [f"x.{nlayers}.tmix"]))
            model_nodes += self._layer_norm(f"x.{nlayers}.tmix",
                                            bn("ln2.weight"), bn("ln2.bias"),
                                            f"x.{nlayers}.ln2")
            if self.hparam.D_de is None:
                model_nodes += self._cmix(nlayers, f"x.{nlayers}.ln2",
                                          f"x_cmix_last_{nlayers}",
                                          f"x.{nlayers}._cmix",
                                          f"x_cmix_next_{nlayers}")
            else:
                raise NotImplementedError("RWKV-7a is not implemented yet.")
            model_nodes.append(
                onnx.helper.make_node(
                    "Add", [f"x.{nlayers}.tmix", f"x.{nlayers}._cmix"],
                    [f"x.{nlayers}.cmix"]))
            model_nodes.append(
                onnx.helper.make_node("Identity", [f"x.{nlayers}.cmix"],
                                      [f"x.{nlayers + 1}"]))
            nlayers += 1
        model_nodes += self._layer_norm(f"x.{nlayers}", "rwkv7.ln_out.weight",
                                        "rwkv7.ln_out.bias", "x.ln_out")
        model_nodes += self._linear_transposed("x.ln_out",
                                               "head.weight",
                                               "x.head",
                                               auto_quantize=False)
        model_nodes.append(
            onnx.helper.make_node("Cast", ["x.head"], ["x.head.casted"],
                                  to=onnx.TensorProto.FLOAT))
        model_nodes.append(
            onnx.helper.make_node("Identity", ["x.head.casted"], ["head"]))

        model_outputs.append(
            onnx.helper.make_tensor_value_info(
                "head", onnx.TensorProto.FLOAT,
                P_shape + ["B", "T", self.hparam.V]))
        if self.args.sampling:
            model_inputs.append(
                onnx.helper.make_value_info(
                    "occurence",
                    onnx.helper.make_optional_type_proto(
                        onnx.helper.make_tensor_type_proto(
                            onnx.TensorProto.FLOAT,
                            P_shape + ["B", self.hparam.V]))))
            model_outputs.append(
                onnx.helper.make_value_info(
                    "y",
                    onnx.helper.make_optional_type_proto(
                        onnx.helper.make_tensor_type_proto(
                            onnx.TensorProto.INT64, P_shape + ["B"]))))
            model_outputs.append(
                onnx.helper.make_value_info(
                    "occurence_next",
                    onnx.helper.make_optional_type_proto(
                        onnx.helper.make_tensor_type_proto(
                            onnx.TensorProto.FLOAT,
                            P_shape + ["B", self.hparam.V]))))
            model_nodes += self._sampling("x.head.casted", "occurence", "y",
                                          "occurence_next")

        if self.args.evolution_strategy:
            lora_a: Callable[[str], str] = lambda name: f"{name}.a"
            lora_b: Callable[[str], str] = lambda name: f"{name}.b"

            P11C: onnx.TypeProto = onnx.helper.make_tensor_type_proto(
                self.hparam.elem_type, ["P", 1, 1, self.hparam.C])
            P1Cr: onnx.TypeProto = onnx.helper.make_tensor_type_proto(
                self.hparam.elem_type,
                ["P", 1, self.hparam.C, self.args.evolution_strategy_rank])
            P1rC: onnx.TypeProto = onnx.helper.make_tensor_type_proto(
                self.hparam.elem_type,
                ["P", 1, self.args.evolution_strategy_rank, self.hparam.C])

            vi: Callable[
                [str, onnx.TypeProto], onnx.
                ValueInfoProto] = lambda name, type_proto: onnx.helper.make_value_info(
                    name, onnx.helper.make_optional_type_proto(type_proto))

            # (m, n) = (m, r) (r, n), r = self.args.evolution_strategy_rank
            def lora_weights(
                    name: str,
                    m: int,
                    n: int,
                    auto_quant: bool = True) -> List[onnx.ValueInfoProto]:
                infos: list[onnx.ValueInfoProto] = []
                t: int = self.hparam.elem_type if self.quant_dtype is None or not auto_quant else self.hparam.quant_elem_type
                infos.append(
                    vi(
                        f"{name}.population.a.opt",
                        onnx.helper.make_tensor_type_proto(
                            t,
                            ["P", 1, m, self.args.evolution_strategy_rank])))
                infos.append(
                    vi(
                        f"{name}.population.b.opt",
                        onnx.helper.make_tensor_type_proto(
                            t,
                            ["P", 1, self.args.evolution_strategy_rank, n])))
                if auto_quant and (self.quant_dtype == np.int8):
                    infos.append(
                        vi(
                            f"{name}.population.a.a.opt",
                            onnx.helper.make_tensor_type_proto(
                                self.hparam.elem_type, [])))
                    infos.append(
                        vi(
                            f"{name}.population.a.b.opt",
                            onnx.helper.make_tensor_type_proto(
                                self.hparam.quant_elem_type, [])))
                    infos.append(
                        vi(
                            f"{name}.population.b.a.opt",
                            onnx.helper.make_tensor_type_proto(
                                self.hparam.elem_type, [])))
                    infos.append(
                        vi(
                            f"{name}.population.b.b.opt",
                            onnx.helper.make_tensor_type_proto(
                                self.hparam.quant_elem_type, [])))
                return infos

            model_inputs.append(
                vi(
                    f"rwkv7.emb.weight.population.a.opt",
                    onnx.helper.make_tensor_type_proto(self.hparam.elem_type, [
                        "P", self.hparam.V, self.args.evolution_strategy_rank
                    ])))
            model_inputs.append(
                vi(
                    f"rwkv7.emb.weight.population.b.opt",
                    onnx.helper.make_tensor_type_proto(self.hparam.elem_type, [
                        "P", 1, self.args.evolution_strategy_rank,
                        self.hparam.C
                    ])))
            # Skip ln0.* because they are pre-computed.
            model_inputs.append(
                vi(
                    "P",
                    onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT64,
                                                       None)))
            for layer_id in range(self.hparam.nlayers):
                pfbpn: Callable[
                    [str],
                    str] = lambda name: f"rwkv7.blocks.{layer_id}.{name}"
                bpn: Callable[
                    [str],
                    str] = lambda name: f"rwkv7.blocks.{layer_id}.{name}.population.opt"
                bpn_a: Callable[
                    [str],
                    str] = lambda name: f"rwkv7.blocks.{layer_id}.{name}.population.a.opt"
                bpn_b: Callable[
                    [str],
                    str] = lambda name: f"rwkv7.blocks.{layer_id}.{name}.population.b.opt"
                tpn: Callable[[str], str] = lambda name: bpn(f"att.{name}")
                pftpn: Callable[[str], str] = lambda name: pfbpn(f"att.{name}")
                tpn_a: Callable[[str], str] = lambda name: bpn_a(f"att.{name}")
                tpn_b: Callable[[str], str] = lambda name: bpn_b(f"att.{name}")
                pfcpn: Callable[[str], str] = lambda name: pfbpn(f"ffn.{name}")
                cpn: Callable[[str], str] = lambda name: bpn(f"ffn.{name}")
                cpn_a: Callable[[str], str] = lambda name: bpn_a(f"ffn.{name}")
                cpn_b: Callable[[str], str] = lambda name: bpn_b(f"ffn.{name}")

                model_inputs.append(vi(bpn("ln1.weight"), P11C))
                model_inputs.append(vi(bpn("ln1.bias"), P11C))
                model_inputs.append(vi(bpn("ln2.weight"), P11C))
                model_inputs.append(vi(bpn("ln2.bias"), P11C))

                model_inputs.append(vi(tpn("x_r"), P11C))
                model_inputs.append(vi(tpn("x_w"), P11C))
                model_inputs.append(vi(tpn("x_k"), P11C))
                model_inputs.append(vi(tpn("x_v"), P11C))
                model_inputs.append(vi(tpn("x_a"), P11C))
                model_inputs.append(vi(tpn("x_g"), P11C))
                model_inputs.append(vi(tpn("w0"), P11C))
                model_inputs += lora_weights(pftpn("w1"), self.hparam.C,
                                             self.hparam.D_w)
                model_inputs += lora_weights(pftpn("w2"), self.hparam.D_w,
                                             self.hparam.C)
                model_inputs.append(vi(tpn("a0"), P11C))
                model_inputs += lora_weights(pftpn("a1"), self.hparam.C,
                                             self.hparam.D_a)
                model_inputs += lora_weights(pftpn("a2"), self.hparam.D_a,
                                             self.hparam.C)
                if layer_id > 0:
                    model_inputs.append(vi(tpn("v0"), P11C))
                    model_inputs += lora_weights(pftpn("v1"), self.hparam.C,
                                                 self.hparam.D_v)
                    model_inputs += lora_weights(pftpn("v2"), self.hparam.D_v,
                                                 self.hparam.C)
                model_inputs += lora_weights(pftpn("g1"), self.hparam.C,
                                             self.hparam.D_g)
                model_inputs += lora_weights(pftpn("g2"), self.hparam.D_g,
                                             self.hparam.C)
                model_inputs.append(vi(tpn("k_k"), P11C))
                model_inputs.append(vi(tpn("k_a"), P11C))
                model_inputs.append(
                    vi(
                        tpn("r_k"),
                        onnx.helper.make_tensor_type_proto(
                            self.hparam.elem_type,
                            ["P", 1, 1, self.hparam.H, self.hparam.N])))
                model_inputs += lora_weights(pftpn("receptance.weight"),
                                             self.hparam.C, self.hparam.C)
                model_inputs += lora_weights(pftpn("key.weight"),
                                             self.hparam.C, self.hparam.C)
                model_inputs += lora_weights(pftpn("value.weight"),
                                             self.hparam.C, self.hparam.C)
                model_inputs += lora_weights(pftpn("output.weight"),
                                             self.hparam.C, self.hparam.C)
                model_inputs.append(vi(tpn("ln_x.weight"), P11C))
                model_inputs.append(vi(tpn("ln_x.bias"), P11C))

                model_inputs.append(vi(cpn("x_k"), P11C))
                model_inputs += lora_weights(pfcpn("key.weight"),
                                             self.hparam.C, self.hparam.C * 4)
                model_inputs += lora_weights(pfcpn("value.weight"),
                                             self.hparam.C * 4, self.hparam.C)
            model_inputs.append(vi("rwkv7.ln_out.weight.population.opt", P11C))
            model_inputs.append(vi("rwkv7.ln_out.bias.population.opt", P11C))
            model_inputs += lora_weights("head.weight",
                                         self.hparam.C,
                                         self.hparam.V,
                                         auto_quant=False)

        model_initializer: list[onnx.TensorProto] = []
        model_initializer += self._constants_tensors()
        for k, v in self._cast_parameters(parameters).items():
            model_initializer.append(onnx.numpy_helper.from_array(v, k))
        model_graph: onnx.GraphProto = onnx.helper.make_graph(
            model_nodes,
            "RWKV7-LM",
            model_inputs,
            model_outputs,
            initializer=model_initializer)
        model: onnx.ModelProto = onnx.helper.make_model(
            model_graph, opset_imports=(onnx.helper.make_opsetid("", 21), ))
        return model


def main() -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "fp16", "fp16-mixed", "bf16", "bf16-mixed"],
        help="data type.")
    parser.add_argument(
        "--quantize",
        default="auto",
        choices=["auto", "i8"],
        help="data type for quantizing. auto to no quantization.")
    parser.add_argument("--quantize-type",
                        default="simple",
                        choices=["simple"])
    parser.add_argument("-s",
                        "--sampling",
                        action="store_true",
                        help="Sample token.")
    parser.add_argument("--alpha_presence",
                        default=2.0,
                        type=float,
                        help="Presence penalty.")
    parser.add_argument("--alpha_frequency",
                        default=0.1,
                        type=float,
                        help="Frequency penalty")
    parser.add_argument("--alpha_decay",
                        default=0.99,
                        type=float,
                        help="Frequency penalty decay")
    parser.add_argument("--topp", default=0.5, type=float, help="p for TopP.")
    parser.add_argument("--topk",
                        default=-1,
                        type=int,
                        help="k for TopK. -1 to disable")
    parser.add_argument("--temperature",
                        default=1.0,
                        type=float,
                        help="Temperature")
    parser.add_argument(
        "--optional-state",
        action="store_true",
        help=
        "Use optional(tensor(dtype)) instead of tensor(dtype) for token shift and state input."
    )
    parser.add_argument(
        "--evolution-strategy",
        action="store_true",
        help="Add optional input to do EGGROLL like evolution strategy.")
    parser.add_argument("--evolution-strategy-rank",
                        default=1,
                        type=int,
                        help="Tensor rank for evolution strategy.")
    parser.add_argument("onnx_file",
                        type=str,
                        help="The ONNX file name to save model.")
    parser.add_argument("st_file",
                        nargs="*",
                        type=str,
                        help="Safetensors file(s) for model parameters.")

    args: argparse.Namespace = parser.parse_args()
    parameters: dict[str, np.ndarray] = {}
    try:
        for path in args.st_file:
            for file in glob.glob(path):
                parameters = parameters | safetensors.load_file(file)
    except FileNotFoundError as e:
        print(f"Safetensor file {args.st_file} is not found: {e}")
        return 1

    model: onnx.ModelProto = Model(
        args, HyperParameter.from_parameters(parameters)).generate(parameters)
    onnx.save_model(model,
                    args.onnx_file,
                    save_as_external_data=True,
                    location=f"{args.onnx_file}.data")
    onnx.shape_inference.infer_shapes_path(args.onnx_file,
                                           check_type=True,
                                           strict_mode=True,
                                           data_prop=True)
    onnx.checker.check_model(args.onnx_file, full_check=True)
    return 0


if __name__ == "__main__":
    exit(main())
