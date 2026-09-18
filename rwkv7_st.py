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
    D: int | None  # dim for deepembd
    nlayers: int
    elem_type: int
    wkv_elem_type: int
    quant_elem_type: int | None  # None if no quantization

    @staticmethod
    def from_parameters(parameters: Dict[str, np.ndarray]) -> HyperParameter:
        V, C = parameters["rwkv7.emb.weight"].shape
        H, N = parameters["rwkv7.blocks.0.att.r_k"].shape
        D = parameters["rwkv7.blocks.0.ffn.s1"].shape[
            1] if "rwkv7.blocks.0.ffn.s1" in parameters.keys() else None
        nlayers: int = 0
        while f"rwkv7.blocks.{nlayers}.att.r_k" in parameters.keys():
            nlayers += 1
        dtype: int = DTYPE_TO_ELEM_TYPE[parameters["rwkv7.emb.weight"].dtype]
        return HyperParameter(V, C, H, N, D, nlayers, dtype, dtype, None)


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
            self.dtype = ELEM_TYPE_TO_DTYPE[self.hparam.wkv_elem_type]
        else:
            self.hparam.wkv_elem_type = DTYPE_TO_ELEM_TYPE[self.wkv_dtype]

        for layer in range(self.hparam.nlayers):
            self.quant_filter.add(f"rwkv7.blocks.{layer}.att.r_k")
        self.wkv_graph: onnx.GraphProto = None
        self.sampling_else: onnx.GraphProto = None
        self.sampling_then: onnx.GraphProto = None

        self.dynamic_quantized: set[str] = set()

    def _dynamic_quantize_hparam(self, x: str, x_scale: str, x_zero_point: str,
                                 qmin: str, qmax: str, qrange: str,
                                 to: int) -> List[onnx.NodeProto]:
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
                quantizeB: bool = False) -> List[onnx.NodeProto]:
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
                                      [f"{output_value}.a"]))
            matmul_nodes.append(
                onnx.helper.make_node("MatMulInteger",
                                      [A_T, B_T, f"{A}.b", f"{B}.b"],
                                      [f"{output_value}.quantized"]))
            matmul_nodes.append(
                onnx.helper.make_node("DequantizeLinear", [
                    f"{output_value}.quantized",
                    f"{output_value}.a",
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
        if self.hparam.D is not None:
            tensors.append(
                onnx.numpy_helper.from_array(
                    np.array((self.hparam.D, ), dtype=np.int64), "D"))
        tensors.append(
            onnx.helper.make_tensor("eps", self.hparam.elem_type, [], 1e-12))
        tensors.append(
            onnx.helper.make_tensor("0.0", onnx.TensorProto.FLOAT, [], 0.0))
        tensors.append(
            onnx.helper.make_tensor("1.0", self.hparam.elem_type, [], 1.0))
        tensors.append(
            onnx.helper.make_tensor("-exp(-0.5)", self.hparam.elem_type, [],
                                    -np.exp(-0.5)))
        # tensors.append(onnx.helper.make_tensor("INT8_MIN", self.hparam.elem_type, [], self.dtype.type(np.iinfo(np.int8).min)))
        # tensors.append(onnx.helper.make_tensor("INT8_MAX", self.hparam.elem_type, [], self.dtype.type(np.iinfo(np.int8).max)))
        # tensors.append(onnx.helper.make_tensor("INT8_RANGE", self.hparam.elem_type, [], self.dtype.type(np.iinfo(np.int8).max - np.iinfo(np.int8).min)))
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
            onnx.numpy_helper.from_array(np.array((2, ), dtype=np.int64), "2"))
        tensors.append(
            onnx.numpy_helper.from_array(np.array((3, ), dtype=np.int64), "3"))
        tensors.append(
            onnx.numpy_helper.from_array(np.array((-1, ), dtype=np.int64),
                                         "-1"))
        tensors.append(
            onnx.numpy_helper.from_array(
                np.array((False, True), dtype=np.bool), "(False, True)"))
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
        if self.hparam.D is not None:
            nodes.append(
                onnx.helper.make_node("Concat", ["B", "T", "D"], ["BTD"],
                                      axis=0))
            nodes.append(
                onnx.helper.make_node("Concat", ["B", "T", "1", "D"], ["BT1D"],
                                      axis=0))
            nodes.append(
                onnx.helper.make_node("Concat", ["B", "T", "D", "D"], ["BTDD"],
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
                   output_value: Optional[str],
                   axis: str = "-1",
                   eps: str = "eps") -> List[onnx.NodeProto]:
        if output_value is None:
            output_value = f"{value}.normalized"

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

    # xW^T
    def _linear_transposed(self,
                           x: str,
                           weight: str,
                           output_value: Optional[str],
                           auto_quantize: bool = True) -> List[onnx.NodeProto]:
        if output_value is None:
            output_value = f"{x}@{weight}.T"

        return self._matmul(x,
                            weight,
                            output_value,
                            auto_quantize=auto_quantize,
                            quantizeA=True)

    def _time_shift(self,
                    x: str,
                    x_last: str,
                    shifted_value: Optional[str] = None,
                    next_value: Optional[str] = None) -> List[onnx.NodeProto]:
        if shifted_value is None:
            shifted_value = f"{x}.shifted"
        if next_value is None:
            next_value = f"{x}.next"

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

    # lerp(a, b, x) = a + (b - a) * x
    # ba = b - a
    def _lerp_pre_computed(self,
                           a: str,
                           ba: str,
                           x: str,
                           output_value: Optional[str] = None
                           ) -> List[onnx.NodeProto]:
        if output_value is None:
            output_value = f"{x}.lerp"

        lerp_nodes: list[onnx.NodeProto] = []
        lerp_nodes.append(
            onnx.helper.make_node("Mul", [ba, x], [f"{x}.lerp.{ba}"]))
        lerp_nodes.append(
            onnx.helper.make_node("Add", [a, f"{x}.lerp.{ba}"],
                                  [output_value]))
        return lerp_nodes

    def _loramlp(self,
                 x: str,
                 A: str,
                 B: str,
                 bias: str | None,
                 f: str,
                 output_value: Optional[str] = None) -> List[onnx.NodeProto]:
        if output_value is None:
            output_value = f"{x}.loramlp.{f}"

        loramlp_nodes: list[onnx.NodeProto] = []
        loramlp_nodes += self._matmul(x,
                                      A,
                                      f"{x}.loramlp.{f}._{A}",
                                      quantizeA=True)
        loramlp_nodes.append(
            onnx.helper.make_node(f, [f"{x}.loramlp.{f}._{A}"],
                                  [f"{x}.loramlp.{f}.{A}"]))
        if bias is None:
            loramlp_nodes += self._matmul(f"{x}.loramlp.{f}.{A}",
                                          B,
                                          output_value,
                                          quantizeA=True)
        else:
            loramlp_nodes += self._matmul(f"{x}.loramlp.{f}.{A}",
                                          B,
                                          f"{x}.loramlp.{f}.{A}.{B}",
                                          quantizeA=True)
            loramlp_nodes.append(
                onnx.helper.make_node("Add",
                                      [f"{x}.loramlp.{f}.{A}.{B}", bias],
                                      [output_value]))
        return loramlp_nodes

    def _wkv(self, wkv_state: str, r: str, w: str, k: str, v: str, a: str,
             b: str, output_state: str,
             output_value: str) -> List[onnx.NodeProto]:
        if self.wkv_graph is None:
            wkv_graph_inputs: list[onnx.ValueInfoProto] = []
            wkv_graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    "wkv_state", self.hparam.wkv_elem_type,
                    ["B", self.hparam.H, self.hparam.N, self.hparam.N]))
            wkv_graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    "w", self.hparam.wkv_elem_type,
                    ["B", self.hparam.H, 1, self.hparam.N]))
            wkv_graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    "a", self.hparam.wkv_elem_type,
                    ["B", self.hparam.H, self.hparam.N]))
            wkv_graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    "b", self.hparam.wkv_elem_type,
                    ["B", self.hparam.H, self.hparam.N]))
            wkv_graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    "v", self.hparam.wkv_elem_type,
                    ["B", self.hparam.H, self.hparam.N]))
            wkv_graph_inputs.append(
                onnx.helper.make_tensor_value_info(
                    "k", self.hparam.wkv_elem_type,
                    ["B", self.hparam.H, self.hparam.N]))

            wkv_graph_outputs: list[onnx.ValueInfoProto] = []
            wkv_graph_outputs.append(
                onnx.helper.make_tensor_value_info(
                    "wkv_next", self.hparam.wkv_elem_type,
                    ["B", self.hparam.H, self.hparam.N, self.hparam.N]))
            wkv_graph_outputs.append(
                onnx.helper.make_tensor_value_info(
                    "wkv_out", self.hparam.wkv_elem_type,
                    ["B", self.hparam.H, self.hparam.N, self.hparam.N]))

            wkv_graph_nodes: list[onnx.NodeProto] = []
            wkv_graph_nodes.append(
                onnx.helper.make_node("Einsum", ["wkv_state", "a"],
                                      ["wkv_state.a"],
                                      equation="bhij,bhj->bhi"))
            wkv_graph_nodes.append(
                onnx.helper.make_node("Einsum", ["wkv_state.a", "b"],
                                      ["wkv_state.a.b"],
                                      equation="bhi,bhj->bhij"))
            wkv_graph_nodes.append(
                onnx.helper.make_node("Einsum", ["v", "k"], ["v.k"],
                                      equation="bhi,bhj->bhij"))
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
            onnx.helper.make_node("Unsqueeze", [f"{w}._casted", "3"],
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
            onnx.helper.make_node("Scan", [
                wkv_state, f"{w}.casted", f"{a}.casted", f"{b}.casted",
                f"{v}.casted", f"{k}.casted"
            ], [output_state, f"_{output_value}.casted"],
                                  body=self.wkv_graph,
                                  num_scan_inputs=5,
                                  scan_input_axes=[1, 1, 1, 1, 1],
                                  scan_output_axes=[1]))

        wkv_nodes.append(
            onnx.helper.make_node("Einsum",
                                  [f"_{output_value}.casted", f"{r}.casted"],
                                  [f"{output_value}.casted"],
                                  equation="bthij,bthj->bthi"))
        wkv_nodes.append(
            onnx.helper.make_node("Cast", [f"{output_value}.casted"],
                                  [output_value],
                                  to=self.hparam.elem_type))
        return wkv_nodes

    def _tmix(self, layer_id: int, x: str, v_first: str, x_last: str,
              wkv_state: str, x_r: str, x_w: str, x_k: str, x_v: str, x_a: str,
              x_g: str, w1: str, w2: str, w0: str, a1: str, a2: str, a0: str,
              v1: str, v2: str, v0: str, g1: str, g2: str, k_k: str, k_a: str,
              r_k: str, receptance_weight: str, key_weight: str,
              value_weight: str, output_weight: str, ln_x_weight: str,
              ln_x_bias: str, output: str, x_next: str,
              wkv_state_next: str) -> List[onnx.NodeProto]:
        tn: Callable[[str], str] = lambda x: f"buf.att.{layer_id}.{x}"

        tmix_nodes: list[onnx.NodeProto] = []
        tmix_nodes += self._time_shift(x, x_last, tn("x.shifted"), x_next)
        tmix_nodes.append(
            onnx.helper.make_node("Sub", [tn("x.shifted"), x],
                                  [tn("x.lerp_pre_compute")]))
        tmix_nodes += self._lerp_pre_computed(x, tn("x.lerp_pre_compute"), x_r,
                                              tn("xr"))
        tmix_nodes += self._lerp_pre_computed(x, tn("x.lerp_pre_compute"), x_w,
                                              tn("xw"))
        tmix_nodes += self._lerp_pre_computed(x, tn("x.lerp_pre_compute"), x_k,
                                              tn("xk"))
        tmix_nodes += self._lerp_pre_computed(x, tn("x.lerp_pre_compute"), x_v,
                                              tn("xv"))
        tmix_nodes += self._lerp_pre_computed(x, tn("x.lerp_pre_compute"), x_a,
                                              tn("xa"))
        tmix_nodes += self._lerp_pre_computed(x, tn("x.lerp_pre_compute"), x_g,
                                              tn("xg"))

        tmix_nodes += self._linear_transposed(tn("xr"), receptance_weight,
                                              tn("r"))
        tmix_nodes += self._loramlp(tn("xw"), w1, w2, w0, "Tanh", tn("_w"))
        tmix_nodes += self._linear_transposed(tn("xk"), key_weight, tn("_k"))
        tmix_nodes += self._linear_transposed(tn("xv"), value_weight, tn("_v"))
        tmix_nodes += self._loramlp(tn("xa"), a1, a2, a0, "Identity", tn("_a"))
        tmix_nodes.append(
            onnx.helper.make_node("Sigmoid", [tn("_a")], [tn("a")]))
        tmix_nodes += self._loramlp(tn("xg"), g1, g2, None, "Sigmoid", tn("g"))
        tmix_nodes.append(
            onnx.helper.make_node("Mul", [tn("_k"), k_k], [tn("_kk")]))
        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [tn("_kk"), "BTHN"],
                                  [tn("_kk.head")]))
        tmix_nodes += self._normalize(tn("_kk.head"), tn("kk.head"))
        tmix_nodes.append(
            onnx.helper.make_node("Sub", [tn("a"), "1.0"],
                                  [tn("a.lerp_pre_compute")]))
        tmix_nodes += self._lerp_pre_computed("1.0", tn("a.lerp_pre_compute"),
                                              k_a, tn("a.lerp"))
        tmix_nodes.append(
            onnx.helper.make_node("Mul", [tn("_k"), tn("a.lerp")], [tn("k")]))
        tmix_nodes.append(
            onnx.helper.make_node("Sigmoid", [tn("_w")], [tn("w.sigmoid")]))
        tmix_nodes.append(
            onnx.helper.make_node("Mul",
                                  ["-exp(-0.5)", tn("w.sigmoid")],
                                  [tn("w.sigmoid.clipped")]))
        tmix_nodes.append(
            onnx.helper.make_node("Exp", [tn("w.sigmoid.clipped")], [tn("w")]))

        if layer_id == 0:
            tmix_nodes.append(
                onnx.helper.make_node("Identity", [tn("_v")], [tn("v")]))
            tmix_nodes.append(
                onnx.helper.make_node("Identity", [tn("v")], [v_first]))
        else:
            assert v1 is not None and v2 is not None
            tmix_nodes += self._loramlp(tn("xv"), v1, v2, v0, "Identity",
                                        tn("_vg"))
            tmix_nodes.append(
                onnx.helper.make_node("Sigmoid", [tn("_vg")], [tn("vg")]))
            tmix_nodes.append(
                onnx.helper.make_node("Sub", [v_first, tn("_v")],
                                      [tn("_v.lerp_pre_compute")]))
            tmix_nodes += self._lerp_pre_computed(tn("_v"),
                                                  tn("_v.lerp_pre_compute"),
                                                  tn("vg"), tn("v"))

        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [tn("r"), "BTHN"],
                                  [tn("r.head")]))
        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [tn("w"), "BTHN"],
                                  [tn("w.head")]))
        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [tn("k"), "BTHN"],
                                  [tn("k.head")]))
        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [tn("v"), "BTHN"],
                                  [tn("v.head")]))
        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [tn("a"), "BTHN"],
                                  [tn("a.head")]))
        tmix_nodes.append(
            onnx.helper.make_node("Mul",
                                  [tn("kk.head"), tn("a.head")],
                                  [tn("b.head")]))

        tmix_nodes += self._wkv(wkv_state, tn("r.head"), tn("w.head"),
                                tn("k.head"), tn("v.head"), tn("kk.head"),
                                tn("b.head"), wkv_state_next, tn("wkv_out"))

        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [tn("wkv_out"), "GC"],
                                  [tn("wkv_out.group")]))
        tmix_nodes.append(
            onnx.helper.make_node(
                "GroupNormalization",
                [tn("wkv_out.group"), ln_x_weight, ln_x_bias],
                [tn("wkv_out.gn.group")],
                epsilon=64e-5,
                num_groups=self.hparam.H))
        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [tn("wkv_out.gn.group"), "BTC"],
                                  [tn("wkv_out.gn")]))
        tmix_nodes.append(
            onnx.helper.make_node("Mul", [r_k, tn("k.head")],
                                  [tn("u.k.head")]))
        tmix_nodes.append(
            onnx.helper.make_node("Einsum",
                                  [tn("r.head"), tn("u.k.head")],
                                  [tn("_u.rk.head")],
                                  equation="bthi,bthi->bth"))
        tmix_nodes.append(
            onnx.helper.make_node("Unsqueeze", [tn("_u.rk.head"), "-1"],
                                  [tn("u.rk.head")]))
        tmix_nodes.append(
            onnx.helper.make_node(
                "Mul", [tn("u.rk.head"), tn("v.head")], [tn("u.head")]))
        tmix_nodes.append(
            onnx.helper.make_node("Reshape", [tn("u.head"), "BTC"],
                                  [tn("rkv")]))
        tmix_nodes.append(
            onnx.helper.make_node(
                "Add", [tn("wkv_out.gn"), tn("rkv")], [tn("_rwkv")]))
        tmix_nodes.append(
            onnx.helper.make_node("Mul", [tn("_rwkv"), tn("g")], [tn("rwkv")]))
        tmix_nodes += self._linear_transposed(tn("rwkv"), output_weight,
                                              output)
        return tmix_nodes

    def _cmix(self, layer_id: int, x: str, x_last: str, x_k: str,
              key_weight: str, value_weight: str, output: str,
              x_next: str) -> List[onnx.NodeProto]:
        cn: Callable[[str], str] = lambda x: f"buf.ffn.{layer_id}.{x}"

        cmix_nodes: list[onnx.NodeProto] = []
        cmix_nodes += self._time_shift(x, x_last, cn("x.shifted"), x_next)
        cmix_nodes.append(
            onnx.helper.make_node("Sub", [cn("x.shifted"), x],
                                  [cn("x.lerp_pre_compute")]))
        cmix_nodes += self._lerp_pre_computed(x, cn("x.lerp_pre_compute"), x_k,
                                              cn("x.lerp"))
        cmix_nodes += self._linear_transposed(cn("x.lerp"), key_weight,
                                              cn("key"))
        cmix_nodes.append(
            onnx.helper.make_node("Relu", [cn("key")], [cn("relu")]))
        cmix_nodes.append(
            onnx.helper.make_node("Mul", [cn("relu"), cn("relu")],
                                  [cn("relu_squared")]))

        cmix_nodes += self._linear_transposed(cn("relu_squared"), value_weight,
                                              output)
        return cmix_nodes

    def _sampling(self, logit: str, occurence: str | None, output: str,
                  occurence_next: str) -> List[onnx.NodeProto]:
        sn: Callable[[str], str] = lambda x: f"sampling.{x}"

        if self.sampling_else is None and self.sampling_then is None:
            sampling_graph_outputs: list[onnx.ValueInfoProto] = []
            y_type_proto: onnx.TypeProto = onnx.helper.make_tensor_type_proto(
                onnx.TensorProto.INT64, ["B"])
            occurence_next_type_proto: onnx.TypeProto = onnx.helper.make_tensor_type_proto(
                onnx.TensorProto.FLOAT, ["B", self.hparam.V])
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
                onnx.helper.make_node("Greater", [sn("occurence"), "0.0"],
                                      [sn("presence")]))
            sampling_then_nodes.append(
                onnx.helper.make_node(
                    "Where", [sn("presence"), "alpha_presence", "0.0"],
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
                    "Where", [sn("_topp"), "0.0",
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
                    ["penalty_target", sn("onehot"), "0.0"],
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

        model_inputs.append(
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.INT64,
                                               ["B", "T"]))
        model_nodes += self._rt_constants_nodes()
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
                self.hparam.elem_type, ["B", 1, self.hparam.C])
            wkv_state_type: onnx.TypeProto = onnx.helper.make_tensor_type_proto(
                self.hparam.wkv_elem_type,
                ["B", self.hparam.H, self.hparam.N, self.hparam.N])
            x_cmix_type: onnx.TypeProto = onnx.helper.make_tensor_type_proto(
                self.hparam.elem_type, ["B", 1, self.hparam.C])

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
                    f"rwkv7.blocks.{nlayers}.att.ts_state", "B1C", x_tmix_type,
                    f"x_tmix_last_{nlayers}")
                model_nodes += self._fallback_nodes(
                    f"wkv_state_{nlayers}_opt",
                    f"rwkv7.blocks.{nlayers}.att.time_state", "BHNN",
                    wkv_state_type, f"wkv_state_{nlayers}")
                model_nodes += self._fallback_nodes(
                    f"x_cmix_last_{nlayers}_opt",
                    f"rwkv7.blocks.{nlayers}.ffn.ts_state", "B1C", x_cmix_type,
                    f"x_cmix_last_{nlayers}")
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

            model_nodes.append(
                onnx.helper.make_node(
                    "LayerNormalization",
                    [f"x.{nlayers}",
                     bn("ln1.weight"),
                     bn("ln1.bias")], [f"x.{nlayers}.ln1"]))
            model_nodes += self._tmix(
                nlayers, f"x.{nlayers}.ln1", "v_first",
                f"x_tmix_last_{nlayers}", f"wkv_state_{nlayers}", tn("x_r"),
                tn("x_w"), tn("x_k"), tn("x_v"), tn("x_a"),
                tn("x_g"), tn("w1"), tn("w2"), tn("w0"), tn("a1"), tn("a2"),
                tn("a0"), tn("v1"), tn("v2"), tn("v0"), tn("g1"), tn("g2"),
                tn("k_k"), tn("k_a"), tn("r_k"), tn("receptance.weight"),
                tn("key.weight"), tn("value.weight"), tn("output.weight"),
                tn("ln_x.weight"), tn("ln_x.bias"), f"x.{nlayers}._tmix",
                f"x_tmix_next_{nlayers}", f"wkv_next_{nlayers}")
            model_nodes.append(
                onnx.helper.make_node("Add",
                                      [f"x.{nlayers}", f"x.{nlayers}._tmix"],
                                      [f"x.{nlayers}.tmix"]))
            model_nodes.append(
                onnx.helper.make_node(
                    "LayerNormalization",
                    [f"x.{nlayers}.tmix",
                     bn("ln2.weight"),
                     bn("ln2.bias")], [f"x.{nlayers}.ln2"]))
            if self.hparam.D is None:
                model_nodes += self._cmix(nlayers, f"x.{nlayers}.ln2",
                                          f"x_cmix_last_{nlayers}", cn("x_k"),
                                          cn("key.weight"), cn("value.weight"),
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
        model_nodes.append(
            onnx.helper.make_node(
                "LayerNormalization",
                [f"x.{nlayers}", "rwkv7.ln_out.weight", "rwkv7.ln_out.bias"],
                ["x.ln_out"]))
        model_nodes += self._linear_transposed("x.ln_out",
                                               "head.weight",
                                               "x.head",
                                               auto_quantize=False)
        model_nodes.append(
            onnx.helper.make_node("Cast", ["x.head"], ["x.head.casted"],
                                  to=onnx.TensorProto.FLOAT))
        model_nodes.append(
            onnx.helper.make_node("Identity", ["x.head.casted"], ["head"]))
        if self.args.sampling:
            model_inputs.append(
                onnx.helper.make_value_info(
                    "occurence",
                    onnx.helper.make_optional_type_proto(
                        onnx.helper.make_tensor_type_proto(
                            onnx.TensorProto.FLOAT, ["B", self.hparam.V]))))
            model_outputs.append(
                onnx.helper.make_value_info(
                    "y",
                    onnx.helper.make_optional_type_proto(
                        onnx.helper.make_tensor_type_proto(
                            onnx.TensorProto.INT64, ["B"]))))
            model_outputs.append(
                onnx.helper.make_value_info(
                    "occurence_next",
                    onnx.helper.make_optional_type_proto(
                        onnx.helper.make_tensor_type_proto(
                            onnx.TensorProto.FLOAT, ["B", self.hparam.V]))))
            model_nodes += self._sampling("x.head.casted", "occurence", "y",
                                          "occurence_next")

        model_outputs.append(
            onnx.helper.make_tensor_value_info("head", onnx.TensorProto.FLOAT,
                                               ["B", "T", self.hparam.V]))

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
