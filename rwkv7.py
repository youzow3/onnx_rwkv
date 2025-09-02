import onnx
import numpy as np
import torch
import argparse

import pathlib


__domain: str = "rwkv7"
__opset_imports: list[onnx.OperatorSetIdProto] = [
        onnx.helper.make_opsetid("", 21),
        onnx.helper.make_opsetid(__domain, 1)]


def make_tensor(tensor: torch.Tensor, name: str | None) -> onnx.TensorProto:
    return onnx.numpy_helper.from_array(
            tensor.to(torch.float).numpy(), name=name)


def make_normalize() -> onnx.FunctionProto:
    eps: onnx.NodeProto = onnx.helper.make_node(
            "Constant", [], ["eps"], value_float=1e-12)
    axes: onnx.NodeProto = onnx.helper.make_node(
            "Constant", [], ["axes"], value_ints=[-1])
    x: onnx.NodeProto = onnx.helper.make_node(
            "Cast", ["x_"], ["x"], to=onnx.TensorProto.FLOAT)
    normalize: list[onnx.NodeProto] = [
            onnx.helper.make_node("ReduceL2", ["x", "axes"], ["x_l2"]),
            onnx.helper.make_node("Max", ["x_l2", "eps"], ["x_l2_eps"]),
            onnx.helper.make_node("Div", ["x", "x_l2_eps"], ["normalize_"]),
            onnx.helper.make_node(
                "CastLike", ["normalize_", "x_"], ["normalize"])]

    return onnx.helper.make_function(__domain, "normalize",
                                     ["x_"], ["normalize"],
                                     [eps, axes, x] + normalize,
                                     __opset_imports)


def make_linear() -> onnx.FunctionProto:
    weight_t: onnx.NodeProto = onnx.helper.make_node(
            "Transpose", ["weight"], ["weight.T"])
    x_weight_t: onnx.NodeProto = onnx.helper.make_node(
            "MatMul", ["x", "weight.T"], ["x@weight.T"])
    return onnx.helper.make_function(
                __domain, "linear", ["x", "weight"], ["x@weight.T"],
                [weight_t, x_weight_t], __opset_imports)


def make_time_shift() -> onnx.FunctionProto:
    constants: list[onnx.NodeProto] = [
            onnx.helper.make_node("Constant", [], ["zero"], value_ints=[0]),
            onnx.helper.make_node("Constant", [], ["one"], value_ints=[1]),
            onnx.helper.make_node("Constant", [], ["two"], value_ints=[2]),
            onnx.helper.make_node("Constant", [], ["three"], value_ints=[3]),
            onnx.helper.make_node(
                "Constant", [], ["max"], value_ints=[np.iinfo(np.int64).max]),
            onnx.helper.make_node("Constant", [], ["x_end"], value_ints=[-1]),
            onnx.helper.make_node("Constant", [], ["T_axes"], value_ints=[1])]
    broadcast: list[onnx.NodeProto] = [
            onnx.helper.make_node("Shape", ["x"], ["x_shape"]),
            onnx.helper.make_node(
                "Slice", ["x_shape", "zero", "one", "zero"], ["B"]),
            onnx.helper.make_node(
                "Slice", ["x_shape", "two", "three", "zero"], ["C"]),
            onnx.helper.make_node(
                "Concat", ["B", "one", "C"], ["expand_shape"], axis=0),
            onnx.helper.make_node(
                "Expand", ["x_last_", "expand_shape"], ["x_last"])]
    x_shift: list[onnx.NodeProto] = [
            onnx.helper.make_node(
                "Slice", ["x", "zero", "x_end", "T_axes"], ["x_shift_"]),
            onnx.helper.make_node(
                "Concat", ["x_last", "x_shift_"], ["x_shift"], axis=1)]
    x_next: onnx.NodeProto = onnx.helper.make_node(
            "Slice", ["x", "x_end", "max", "T_axes"], ["x_next"])

    return onnx.helper.make_function(
            __domain, "time_shift", ["x", "x_last_"], ["x_shift", "x_next"],
            constants + broadcast + x_shift + [x_next], __opset_imports)


def make_lerp() -> onnx.FunctionProto:
    lerp: list[onnx.NodeProto] = [
            onnx.helper.make_node("Sub", ["b", "a"], ["b_sub_a"]),
            onnx.helper.make_node("Mul", ["b_sub_a", "x"], ["b_sub_a_x"]),
            onnx.helper.make_node("Add", ["a", "b_sub_a_x"], ["lerp"])]
    return onnx.helper.make_function(__domain, "lerp",
                                     ["a", "b", "x"], ["lerp"],
                                     lerp, __opset_imports)


def make_loramlp() -> list[onnx.FunctionProto]:
    xA: onnx.NodeProto = onnx.helper.make_node("MatMul", ["x", "A"], ["xA"])
    identity_xA: onnx.NodeProto = onnx.helper.make_node(
            "Identity", ["xA"], ["f_xA"])
    sigmoid_xA: onnx.NodeProto = onnx.helper.make_node(
            "Sigmoid", ["xA"], ["f_xA"])
    tanh_xA: onnx.NodeProto = onnx.helper.make_node("Tanh", ["xA"], ["f_xA"])
    f_xAB: onnx.NodeProto = onnx.helper.make_node(
            "MatMul", ["f_xA", "B"], ["f_xAB"])

    f_xAB_identity: onnx.NodeProto = onnx.helper.make_node(
            "Identity", ["f_xAB"], ["loramlp"])
    f_xAB_bias: onnx.NodeProto = onnx.helper.make_node(
            "Add", ["f_xAB", "bias"], ["loramlp"])

    return [
            onnx.helper.make_function(
                __domain, "loramlp_identity_bias", ["x", "A", "B", "bias"],
                ["loramlp"], [xA, identity_xA, f_xAB, f_xAB_bias],
                __opset_imports),
            onnx.helper.make_function(
                __domain, "loramlp_sigmoid_bias", ["x", "A", "B", "bias"],
                ["loramlp"], [xA, sigmoid_xA, f_xAB, f_xAB_bias],
                __opset_imports),
            onnx.helper.make_function(
                __domain, "loramlp_tanh_bias", ["x", "A", "B", "bias"],
                ["loramlp"], [xA, tanh_xA, f_xAB, f_xAB_bias],
                __opset_imports),
            onnx.helper.make_function(
                __domain, "loramlp_identity", ["x", "A", "B"],
                ["loramlp"], [xA, identity_xA, f_xAB, f_xAB_identity],
                __opset_imports),
            onnx.helper.make_function(
                __domain, "loramlp_sigmoid", ["x", "A", "B"],
                ["loramlp"], [xA, sigmoid_xA, f_xAB, f_xAB_identity],
                __opset_imports),
            onnx.helper.make_function(
                __domain, "loramlp_tanh", ["x", "A", "B"],
                ["loramlp"], [xA, tanh_xA, f_xAB, f_xAB_identity],
                __opset_imports),
            ]


def make_wkv7(dtype: int) -> onnx.FunctionProto:
    #
    # WKV Operation
    #
    wkv_state_value_info: onnx.ValueInfoProto
    wkv_state_value_info = onnx.helper.make_tensor_value_info(
            "wkv_state", dtype, ["B", "H", "N", "N"])
    w_value_info: onnx.ValueInfoProto = onnx.helper.make_tensor_value_info(
            "w", dtype, ["B", "H", "N"])
    ab_value_info: onnx.ValueInfoProto = onnx.helper.make_tensor_value_info(
            "ab", dtype, ["B", "H", "N", "N"])
    vk_value_info: onnx.ValueInfoProto = onnx.helper.make_tensor_value_info(
            "vk", dtype, ["B", "H", "N", "N"])

    wkv_next_value_info: onnx.ValueInfoProto = onnx.helper.make_tensor_value_info(
            "wkv_next", dtype, ["B", "H", "N", "N"])
    wkv_out_value_info: onnx.ValueInfoProto = onnx.helper.make_tensor_value_info(
            "wkv_out", dtype, ["B", "H", "N", "N"])

    diag_w_unsqueeze: onnx.NodeProto = onnx.helper.make_node(
            "Constant", [], ["diag_w_unsqueeze"], value_ints=[2])
    diag_w: onnx.NodeProto = onnx.helper.make_node(
            "Unsqueeze", ["w", "diag_w_unsqueeze"], ["diag_w"])
    wkv_w: onnx.NodeProto = onnx.helper.make_node(
            "Mul", ["wkv_state", "diag_w"], ["wkv_w"])
    wkv_ab: onnx.NodeProto = onnx.helper.make_node(
            "MatMul", ["wkv_state", "ab"], ["wkv_ab"])
    wkv_next: list[onnx.NodeProto] = [
            onnx.helper.make_node("Sub", ["wkv_w", "wkv_ab"], ["wkv_w_ab"]),
            onnx.helper.make_node("Add", ["wkv_w_ab", "vk"], ["wkv_next"])]
    wkv_out: onnx.NodeProto = onnx.helper.make_node(
            "Identity", ["wkv_next"], ["wkv_out"])

    wkv_loop_graph: onnx.GraphProto = onnx.helper.make_graph(
            [diag_w_unsqueeze, diag_w, wkv_w, wkv_ab] + wkv_next + [wkv_out],
            "wkv_loop",
            [wkv_state_value_info, w_value_info, ab_value_info, vk_value_info],
            [wkv_next_value_info, wkv_out_value_info])

    onnx.checker.check_graph(wkv_loop_graph)

    #
    # WKV preparation
    #
    constants: list[onnx.NodeProto] = [
            onnx.helper.make_node("Constant", [], ["[0]"], value_ints=[0]),
            onnx.helper.make_node("Constant", [], ["[1]"], value_ints=[1]),
            onnx.helper.make_node("Constant", [], ["[3]"], value_ints=[3]),
            onnx.helper.make_node("Constant", [], ["[4]"], value_ints=[4])]
    broadcast: list[onnx.NodeProto] = [
            onnx.helper.make_node("Shape", ["r"], ["BTHN"]),
            onnx.helper.make_node("Shape", ["wkv_state_"], ["1HNN"]),
            onnx.helper.make_node("Slice", ["BTHN", "[0]", "[1]"], ["B"]),
            onnx.helper.make_node("Slice", ["1HNN", "[1]", "[4]"], ["HNN"]),
            onnx.helper.make_node("Concat", ["B", "HNN"], ["BHNN"], axis=0),
            onnx.helper.make_node(
                "Expand", ["wkv_state_", "BHNN"], ["wkv_state"])]
    ab: list[onnx.NodeProto] = [
            onnx.helper.make_node("Unsqueeze", ["a", "[4]"], ["a_t"]),
            onnx.helper.make_node("Unsqueeze", ["b", "[3]"], ["b_t"]),
            onnx.helper.make_node("MatMul", ["a_t", "b_t"], ["ab"])]
    vk: list[onnx.NodeProto] = [
            onnx.helper.make_node("Unsqueeze", ["v", "[4]"], ["v_t"]),
            onnx.helper.make_node("Unsqueeze", ["k", "[3]"], ["k_t"]),
            onnx.helper.make_node("MatMul", ["v_t", "k_t"], ["vk"])]

    w_float: onnx.NodeProto = onnx.helper.make_node(
            "Cast", ["w"], ["w_float"], to=dtype)
    ab_float: onnx.NodeProto = onnx.helper.make_node(
            "Cast", ["ab"], ["ab_float"], to=dtype)
    vk_float: onnx.NodeProto = onnx.helper.make_node(
            "Cast", ["vk"], ["vk_float"], to=dtype)

    wkv_loop: onnx.NodeProto = onnx.helper.make_node(
            "Scan", ["wkv_state", "w_float", "ab_float", "vk_float"],
            ["wkv_state_next", "wkv_out"],
            body=wkv_loop_graph, num_scan_inputs=3,
            scan_input_axes=[1, 1, 1], scan_output_axes=[1])

    wkv_out_casted: onnx.NodeProto = onnx.helper.make_node(
            "CastLike", ["wkv_out", "w"], ["wkv_out_casted"])

    rwkv: [onnx.NodeProto] = [
            onnx.helper.make_node("Constant", [], ["4"], value_ints=[4]),
            onnx.helper.make_node("Unsqueeze", ["r", "4"], ["r_unsqueeze"]),
            onnx.helper.make_node(
                "MatMul", ["wkv_out_casted", "r_unsqueeze"], ["rwkv"])]

    return onnx.helper.make_function(
            __domain, "wkv7",
            ["wkv_state_", "r", "w", "k", "v", "a", "b"],
            ["wkv_state_next", "rwkv"],
            constants + broadcast + ab + vk + [
                w_float, ab_float, vk_float, wkv_loop, wkv_out_casted] + rwkv,
            __opset_imports)


def make_time_mix(dim: int, head_size: int, dim_att: int
                  ) -> list[onnx.FunctionProto]:
    assert isinstance(dim, int)
    assert isinstance(head_size, int)
    assert isinstance(dim_att, int)

    n_head: int = dim_att // head_size
    assert dim_att % n_head == 0

    constants: list[onnx.NodeProto] = [
            onnx.helper.make_node("Constant", [], ["H"], value_ints=[n_head]),
            onnx.helper.make_node(
                "Constant", [], ["N"], value_ints=[head_size]),
            onnx.helper.make_node("Constant", [], ["[-1]"], value_ints=[-1]),
            onnx.helper.make_node("Constant", [], ["(1.0)"], value_float=1.0),
            onnx.helper.make_node("CastLike", ["(1.0)", "x"], ["1.0"]),
            onnx.helper.make_node(
                "Constant", [], ["(-exp(-0.5))"], value_float=-np.exp(-0.5)),
            onnx.helper.make_node(
                "CastLike", ["(-exp(-0.5))", "x"], ["-exp(-0.5)"]),
            onnx.helper.make_node("Constant", [], ["[0]"], value_ints=[0]),
            onnx.helper.make_node("Constant", [], ["[1]"], value_ints=[1]),
            onnx.helper.make_node("Constant", [], ["[2]"], value_ints=[2]),
            onnx.helper.make_node("Constant", [], ["[3]"], value_ints=[3]),
            onnx.helper.make_node("Shape", ["x"], ["BTC"]),
            onnx.helper.make_node("Slice", ["BTC", "[0]", "[1]"], ["B"]),
            onnx.helper.make_node("Slice", ["BTC", "[1]", "[2]"], ["T"]),
            onnx.helper.make_node("Slice", ["BTC", "[2]", "[3]"], ["C"]),
            onnx.helper.make_node(
                "Concat", ["B", "T", "H", "N"], ["BTHN"], axis=0),
            onnx.helper.make_node("Mul", ["B", "T"], ["G"]),
            onnx.helper.make_node("Concat", ["G", "C"], ["GC"], axis=0)]

    token_shift: list[onnx.NodeProto] = [
            onnx.helper.make_node(
                "time_shift", ["x", "x_last"], ["x_shift", "x_next"],
                domain=__domain),
            onnx.helper.make_node(
                "lerp", ["x", "x_shift", "x_r"], ["xr"], domain=__domain),
            onnx.helper.make_node(
                "lerp", ["x", "x_shift", "x_w"], ["xw"], domain=__domain),
            onnx.helper.make_node(
                "lerp", ["x", "x_shift", "x_k"], ["xk"], domain=__domain),
            onnx.helper.make_node(
                "lerp", ["x", "x_shift", "x_v"], ["xv"], domain=__domain),
            onnx.helper.make_node(
                "lerp", ["x", "x_shift", "x_a"], ["xa"], domain=__domain),
            onnx.helper.make_node(
                "lerp", ["x", "x_shift", "x_g"], ["xg"], domain=__domain)]

    weight_prepare: list[onnx.NodeProto] = [
            onnx.helper.make_node(
                "linear", ["xr", "receptance.weight"], ["r"], domain=__domain),
            onnx.helper.make_node(
                "loramlp_tanh_bias", ["xw", "w1", "w2", "w0"], ["w_"],
                domain=__domain),
            onnx.helper.make_node(
                "linear", ["xk", "key.weight"], ["k_"], domain=__domain),
            onnx.helper.make_node(
                "linear", ["xv", "value.weight"], ["v_"], domain=__domain),
            onnx.helper.make_node(
                "loramlp_identity_bias", ["xa", "a1", "a2", "a0"], ["a_"],
                domain=__domain),
            onnx.helper.make_node("Sigmoid", ["a_"], ["a"]),
            onnx.helper.make_node(
                "loramlp_sigmoid", ["xg", "g1", "g2"], ["g"], domain=__domain),

            onnx.helper.make_node("Mul", ["k_", "k_k"], ["k_*k_k"]),
            onnx.helper.make_node(
                "Reshape", ["k_*k_k", "BTHN"], ["k_*k_k:head"]),
            onnx.helper.make_node("normalize", ["k_*k_k:head"], ["kk:head"],
                                  domain=__domain),
            onnx.helper.make_node(
                "lerp", ["1.0", "a", "k_a"], ["lerp(1, a, alpha)"],
                domain=__domain),
            onnx.helper.make_node(
                "Mul", ["k_", "lerp(1, a, alpha)"], ["k"]),

            onnx.helper.make_node("Sigmoid", ["w_"], ["sigmoid(w_)"]),
            onnx.helper.make_node("Mul", ["-exp(-0.5)", "sigmoid(w_)"],
                                  ["-exp(-0.5)*sigmoid(w_)"]),
            onnx.helper.make_node("Exp", ["-exp(-0.5)*sigmoid(w_)"], ["w"])]

    v_l0: onnx.NodeProto = onnx.helper.make_node("Identity", ["v_"], ["v"])
    v_l: list[onnx.NodeProto] = [
            onnx.helper.make_node(
                "loramlp_identity_bias", ["xv", "v1", "v2", "v0"], ["vg_"],
                domain=__domain),
            onnx.helper.make_node("Sigmoid", ["vg_"], ["vg"]),
            onnx.helper.make_node(
                "lerp", ["v_", "v_first", "vg"], ["v"], domain=__domain)]

    head_prepare: list[onnx.NodeProto] = [
            onnx.helper.make_node("Reshape", ["r", "BTHN"], ["r:head"]),
            onnx.helper.make_node("Reshape", ["w", "BTHN"], ["w:head"]),
            onnx.helper.make_node("Reshape", ["k", "BTHN"], ["k:head"]),
            onnx.helper.make_node("Reshape", ["v", "BTHN"], ["v:head"]),
            onnx.helper.make_node("Reshape", ["a", "BTHN"], ["a:head"]),
            onnx.helper.make_node("Mul", ["kk:head", "a:head"], ["b:head"])]

    wkv: onnx.NodeProto = onnx.helper.make_node(
            "wkv7",
            ["wkv_state",
             "r:head", "w:head", "k:head", "v:head", "kk:head", "b:head"],
            ["wkv_state_next", "wkv_out"], domain=__domain)

    readout: list[onnx.NodeProto] = [
            onnx.helper.make_node(
                "Reshape", ["wkv_out", "GC"], ["wkv_out:group"]),
            onnx.helper.make_node(
                "GroupNormalization",
                ["wkv_out:group", "ln_x.weight", "ln_x.bias"],
                ["wkv_out_gn:group"], epsilon=64e-5, num_groups=n_head),
            onnx.helper.make_node(
                "Reshape", ["wkv_out_gn:group", "BTC"], ["wkv_out_gn"]),
            onnx.helper.make_node("Mul", ["r", "k"], ["r*k"]),
            onnx.helper.make_node("Reshape", ["r_k", "[-1]"], ["r_k:flat"]),
            onnx.helper.make_node("Mul", ["r*k", "r_k:flat"], ["r*k*r_k"]),
            onnx.helper.make_node(
                "Reshape", ["r*k*r_k", "BTHN"], ["r*k*r_k:head"]),
            onnx.helper.make_node(
                "ReduceSum", ["r*k*r_k:head", "[-1]"], ["rk"]),
            onnx.helper.make_node("Mul", ["rk", "v:head"], ["rkv:head"]),
            onnx.helper.make_node("Reshape", ["rkv:head", "BTC"], ["rkv"]),
            onnx.helper.make_node(
                "Add", ["wkv_out_gn", "rkv"], ["wkv_out_gn+rkv"]),
            onnx.helper.make_node(
                "Mul", ["wkv_out_gn+rkv", "g"], ["(wkv_out_gn+rkv)*g"]),
            onnx.helper.make_node(
                "linear", ["(wkv_out_gn+rkv)*g", "output.weight"], ["output"],
                domain=__domain)]

    return [
            onnx.helper.make_function(
                __domain, "time_mix0",
                ["x", "x_last", "wkv_state",
                 "x_r", "x_w", "x_k", "x_v", "x_a", "x_g",
                 "w1", "w2", "w0", "a1", "a2", "a0", "g1", "g2",
                 "k_k", "k_a", "r_k",
                 "receptance.weight", "key.weight",
                 "value.weight", "output.weight", "ln_x.weight", "ln_x.bias"],
                ["output", "v", "x_next", "wkv_state_next"],
                constants + token_shift + weight_prepare + [v_l0] +
                head_prepare + [wkv] + readout, __opset_imports),
            onnx.helper.make_function(
                __domain, "time_mix",
                ["x", "v_first", "x_last", "wkv_state",
                 "x_r", "x_w", "x_k", "x_v", "x_a", "x_g",
                 "w1", "w2", "w0", "a1", "a2", "a0",
                 "v1", "v2", "v0", "g1", "g2",
                 "k_k", "k_a", "r_k",
                 "receptance.weight", "key.weight",
                 "value.weight", "output.weight", "ln_x.weight", "ln_x.bias"],
                ["output", "x_next", "wkv_state_next"],
                constants + token_shift + weight_prepare + v_l +
                head_prepare + [wkv] + readout, __opset_imports)]


def make_channel_mix(C: int) -> onnx.FunctionProto:
    deepemb_constants: list[onnx.NodeProto] = [
            onnx.helper.make_node("Constant", [], ["[0]"], value_ints=[0]),
            onnx.helper.make_node("Constant", [], ["[1]"], value_ints=[1]),
            onnx.helper.make_node("Constant", [], ["[2]"], value_ints=[2]),
            onnx.helper.make_node("Constant", [], ["[32]"], value_ints=[32])
    ]

    deepemb_shape: list[onnx.NodeProto] = [
        onnx.helper.make_node("Shape", ["x"], ["BTC"]),
        onnx.helper.make_node("Slice", ["BTC", "[0]", "[1]"], ["B"]),
        onnx.helper.make_node("Slice", ["BTC", "[1]", "[2]"], ["T"]),
        onnx.helper.make_node("Concat", ["B", "T", "[32]"], ["BT32"], axis=0),
        onnx.helper.make_node("Concat", ["B", "T", "[1]", "[32]"], ["BT1x32"],
                              axis=0),
        onnx.helper.make_node(
            "Concat", ["B", "T", "[32]", "[32]"], ["BT32x32"], axis=0)
    ]

    time_shift: onnx.NodeProto = onnx.helper.make_node(
            "time_shift", ["x", "x_last"],
            ["x_shift", "x_last_next"], domain=__domain)
    lerp: onnx.NodeProto = onnx.helper.make_node(
            "lerp", ["x", "x_shift", "x_k"], ["lerp"], domain=__domain)
    key_weight_t: onnx.NodeProto = onnx.helper.make_node(
            "Transpose", ["key.weight"], ["key_weight_t"])
    value_weight_t: onnx.NodeProto = onnx.helper.make_node(
            "Transpose", ["value.weight"], ["value_weight_t"])
    k: onnx.NodeProto = onnx.helper.make_node(
            "MatMul", ["lerp", "key_weight_t"], ["k"])
    relu: onnx.NodeProto = onnx.helper.make_node("Relu", ["k"], ["relu"])
    relu2: onnx.NodeProto = onnx.helper.make_node(
            "Mul", ["relu", "relu"], ["relu2"])
    x_s1_: onnx.NodeProto = onnx.helper.make_node(
            "MatMul", ["x", "s1"], ["x_s1_"])
    x_s1: onnx.NodeProto = onnx.helper.make_node(
            "Reshape", ["x_s1_", "BT1x32"], ["x_s1"])
    semb: onnx.NodeProto = onnx.helper.make_node(
            "Reshape", ["semb_", "BT32x32"], ["semb"])
    x_s1_semb_: onnx.NodeProto = onnx.helper.make_node(
            "MatMul", ["x_s1", "semb"], ["x_s1_semb_"])
    x_s1_semb: onnx.NodeProto = onnx.helper.make_node(
            "Reshape", ["x_s1_semb_", "BT32"], ["x_s1_semb"])
    x_s1_semb_s2: onnx.NodeProto = onnx.helper.make_node(
            "MatMul", ["x_s1_semb", "s2"], ["x_s1_semb_s2"])
    x_s1_semb_s2_s0: onnx.NodeProto = onnx.helper.make_node(
            "Add", ["x_s1_semb_s2", "s0"], ["x_s1_semb_s2_s0"])
    deepemb: onnx.NodeProto = onnx.helper.make_node(
            "Mul", ["relu2", "x_s1_semb_s2_s0"], ["deepemb"])
    # For normal RWKV-7
    _deepemb: onnx.NodeProto = onnx.helper.make_node(
            "Identity", ["relu2"], ["deepemb"])

    v: onnx.NodeProto = onnx.helper.make_node(
            "MatMul", ["deepemb", "value_weight_t"], ["v"])
    return [
            onnx.helper.make_function(
                __domain, "channel_mix",
                ["x", "x_last", "x_k", "key.weight", "value.weight"],
                ["v", "x_last_next"],
                [time_shift, lerp, key_weight_t,
                 value_weight_t, k, relu, relu2, _deepemb, v], __opset_imports
                ),
            onnx.helper.make_function(
                __domain, "channel_mix_a",
                ["x", "x_last", "x_k", "key.weight", "value.weight",
                 "s1", "semb_", "s2", "s0"],
                ["v", "x_last_next"],
                deepemb_constants + deepemb_shape + [
                    time_shift, lerp, key_weight_t, value_weight_t,
                    k, relu, relu2, x_s1_, x_s1, semb, x_s1_semb_, x_s1_semb,
                    x_s1_semb_s2, x_s1_semb_s2_s0, deepemb, v],
                __opset_imports)
            ]


def make_block() -> list[onnx.FunctionProto]:
    ln0: onnx.NodeProto = onnx.helper.make_node(
            "LayerNormalization",
            ["emb", "ln0.weight", "ln0.bias"], ["emb_ln0"])
    ln0_: onnx.NodeProto = onnx.helper.make_node(
            "Identity", ["emb"], ["emb_ln0"])

    ln1: onnx.NodeProto = onnx.helper.make_node(
            "LayerNormalization",
            ["emb_ln0", "ln1.weight", "ln1.bias"], ["emb_ln1"])
    tmix0: onnx.NodeProto = onnx.helper.make_node(
            "time_mix0",
            ["emb_ln1", "x_tmix_last", "wkv_state",
             "att.x_r", "att.x_w", "att.x_k", "att.x_v", "att.x_a", "att.x_g",
             "att.w1", "att.w2", "att.w0", "att.a1", "att.a2", "att.a0",
             "att.g1", "att.g2", "att.k_k", "att.k_a", "att.r_k",
             "att.receptance.weight", "att.key.weight", "att.value.weight",
             "att.output.weight", "att.ln_x.weight", "att.ln_x.bias"],
            ["emb_tmix", "v_first", "x_tmix_next", "wkv_next"],
            domain=__domain)
    tmix: onnx.NodeProto = onnx.helper.make_node(
            "time_mix",
            ["emb_ln1", "v_first", "x_tmix_last", "wkv_state",
             "att.x_r", "att.x_w", "att.x_k", "att.x_v", "att.x_a", "att.x_g",
             "att.w1", "att.w2", "att.w0", "att.a1", "att.a2", "att.a0",
             "att.v1", "att.v2", "att.v0", "att.g1", "att.g2",
             "att.k_k", "att.k_a", "att.r_k",
             "att.receptance.weight", "att.key.weight", "att.value.weight",
             "att.output.weight", "att.ln_x.weight", "att.ln_x.bias"],
            ["emb_tmix", "x_tmix_next", "wkv_next"], domain=__domain)
    tmix_x: onnx.NodeProto = onnx.helper.make_node(
            "Add", ["emb_ln0", "emb_tmix"], ["emb_tmix_x"])
    ln2: onnx.NodeProto = onnx.helper.make_node(
            "LayerNormalization",
            ["emb_tmix_x", "ln2.weight", "ln2.bias"], ["emb_tmix_x_ln2"])
    cmix: onnx.NodeProto = onnx.helper.make_node(
            "channel_mix",
            ["emb_tmix_x_ln2", "x_cmix_last",
             "ffn.x_k", "ffn.key.weight", "ffn.value.weight"],
            ["emb_cmix", "x_cmix_next"], domain=__domain)
    cmix_a: onnx.NodeProto = onnx.helper.make_node(
            "channel_mix_a",
            ["emb_tmix_x_ln2", "x_cmix_last",
             "ffn.x_k", "ffn.key.weight", "ffn.value.weight",
             "ffn.s1", "semb", "ffn.s2", "ffn.s0"],
            ["emb_cmix", "x_cmix_next"], domain=__domain)
    cmix_x: onnx.NodeProto = onnx.helper.make_node(
            "Add", ["emb_tmix_x", "emb_cmix"], ["emb_cmix_tmix_x"])

    return [
            onnx.helper.make_function(
                __domain, "block0",
                ["emb", "x_tmix_last", "wkv_state", "x_cmix_last",
                 "ln0.weight", "ln0.bias",
                 "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
                 "att.x_r", "att.x_w", "att.x_k", "att.x_v",
                 "att.x_a", "att.x_g",
                 "att.w1", "att.w2", "att.w0", "att.a1", "att.a2", "att.a0",
                 "att.g1", "att.g2", "att.k_k", "att.k_a", "att.r_k",
                 "att.receptance.weight", "att.key.weight", "att.value.weight",
                 "att.output.weight", "att.ln_x.weight", "att.ln_x.bias",
                 "ffn.x_k", "ffn.key.weight", "ffn.value.weight"],
                ["emb_cmix_tmix_x", "v_first",
                 "x_tmix_next", "wkv_next", "x_cmix_next"],
                [ln0, ln1, tmix0, tmix_x, ln2, cmix, cmix_x], __opset_imports),
            onnx.helper.make_function(
                __domain, "block",
                ["emb", "v_first", "x_tmix_last", "wkv_state", "x_cmix_last",
                 "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
                 "att.x_r", "att.x_w", "att.x_k", "att.x_v",
                 "att.x_a", "att.x_g",
                 "att.w1", "att.w2", "att.w0", "att.a1", "att.a2", "att.a0",
                 "att.v1", "att.v2", "att.v0", "att.g1", "att.g2",
                 "att.k_k", "att.k_a", "att.r_k",
                 "att.receptance.weight", "att.key.weight", "att.value.weight",
                 "att.output.weight", "att.ln_x.weight", "att.ln_x.bias",
                 "ffn.x_k", "ffn.key.weight", "ffn.value.weight"],
                ["emb_cmix_tmix_x", "x_tmix_next", "wkv_next", "x_cmix_next"],
                [ln0_, ln1, tmix, tmix_x, ln2, cmix, cmix_x], __opset_imports),
            onnx.helper.make_function(
                __domain, "block0_a",
                ["emb", "x_tmix_last", "wkv_state", "x_cmix_last",
                 "ln0.weight", "ln0.bias",
                 "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
                 "att.x_r", "att.x_w", "att.x_k", "att.x_v",
                 "att.x_a", "att.x_g",
                 "att.w1", "att.w2", "att.w0", "att.a1", "att.a2", "att.a0",
                 "att.g1", "att.g2", "att.k_k", "att.k_a", "att.r_k",
                 "att.receptance.weight", "att.key.weight", "att.value.weight",
                 "att.output.weight", "att.ln_x.weight", "att.ln_x.bias",
                 "ffn.x_k", "ffn.key.weight", "ffn.value.weight",
                 "ffn.s1", "semb", "ffn.s2", "ffn.s0"],
                ["emb_cmix_tmix_x", "v_first",
                 "x_tmix_next", "wkv_next", "x_cmix_next"],
                [ln0, ln1, tmix0, tmix_x, ln2, cmix_a, cmix_x],
                __opset_imports),
            onnx.helper.make_function(
                __domain, "block_a",
                ["emb", "v_first", "x_tmix_last", "wkv_state", "x_cmix_last",
                 "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
                 "att.x_r", "att.x_w", "att.x_k", "att.x_v",
                 "att.x_a", "att.x_g",
                 "att.w1", "att.w2", "att.w0", "att.a1", "att.a2", "att.a0",
                 "att.v1", "att.v2", "att.v0", "att.g1", "att.g2",
                 "att.k_k", "att.k_a", "att.r_k",
                 "att.receptance.weight", "att.key.weight", "att.value.weight",
                 "att.output.weight", "att.ln_x.weight", "att.ln_x.bias",
                 "ffn.x_k", "ffn.key.weight", "ffn.value.weight",
                 "ffn.s1", "semb", "ffn.s2", "ffn.s0"],
                ["emb_cmix_tmix_x", "x_tmix_next", "wkv_next", "x_cmix_next"],
                [ln0_, ln1, tmix, tmix_x, ln2, cmix_a, cmix_x],
                __opset_imports)]


def make_sampling(internal_dtype: int = onnx.TensorProto.FLOAT
                  ) -> onnx.FunctionProto:
    constants: list[onnx.NodeProto] = [
            onnx.helper.make_node("Constant", [], ["[-1]"], value_ints=[-1]),
            onnx.helper.make_node("Constant", [], ["[0]"], value_ints=[0]),
            onnx.helper.make_node("Constant", [], ["[1]"], value_ints=[1]),
            onnx.helper.make_node("Constant", [], ["[2]"], value_ints=[2]),
            onnx.helper.make_node("Constant", [], ["[3]"], value_ints=[3]),
            onnx.helper.make_node("Constant", [], ["-1"], value_int=-1),
            onnx.helper.make_node("Constant", [], ["0"], value_int=0),
            onnx.helper.make_node(
                "Constant", [], ["starts"], value_ints=[0, 0, 0]),
            onnx.helper.make_node("Cast", ["x_"], ["x"], to=internal_dtype),
            onnx.helper.make_node("Cast", ["temp_"], ["temp"],
                                  to=internal_dtype),
            onnx.helper.make_node("Unsqueeze", ["topk_", "0"], ["topk"]),
            onnx.helper.make_node("Cast", ["topp_"], ["topp"],
                                  to=internal_dtype)
    ]

    shapes: list[onnx.NodeProto] = [
            onnx.helper.make_node("Shape", ["x_"], ["BTV"]),
            onnx.helper.make_node("Slice", ["BTV", "[0]", "[1]"], ["B"]),
            onnx.helper.make_node("Slice", ["BTV", "[1]", "[2]"], ["T"]),
            onnx.helper.make_node("Slice", ["BTV", "[2]", "[3]"], ["V"]),
            onnx.helper.make_node("Mul", ["B", "T"], ["B*T"]),
            onnx.helper.make_node("Concat", ["B*T", "V"], ["B*TV"], axis=0),
            onnx.helper.make_node("Concat", ["B", "T"], ["BT"], axis=0),
            onnx.helper.make_node("Concat", ["BT", "[1]"], ["BT1"], axis=0),
            onnx.helper.make_node("Concat", ["BT", "topk"], ["BTK"], axis=0),
            onnx.helper.make_node("Concat", ["B*T", "topk"], ["B*TK"], axis=0)
    ]

    temperature: list[onnx.NodeProto] = [
            onnx.helper.make_node("Div", ["x", "temp"], ["x_temp"]),
            onnx.helper.make_node("Softmax", ["x_temp"], ["temp_softmax"])
    ]

    topk: list[onnx.NodeProto] = [
            onnx.helper.make_node("TopK", ["temp_softmax", "topk"],
                                  ["topk_temp_softmax_", "topk_idx"]),
            onnx.helper.make_node("Reshape",
                                  ["topk_idx", "B*TK"], ["topk_idx_bt"]),
            onnx.helper.make_node("ReduceSum", ["topk_temp_softmax_", "[-1]"],
                                  ["topk_temp_softmax_sum"]),
            onnx.helper.make_node(
                "Div", ["topk_temp_softmax_", "topk_temp_softmax_sum"],
                ["topk_scaled"])
    ]

    topp: list[onnx.NodeProto] = [
            onnx.helper.make_node(
                "Cast", ["0"], ["0_casted"], to=internal_dtype),
            onnx.helper.make_node(
                "CumSum", ["topk_scaled", "[-1]"], ["topk_scaled_cumsum"],
                exclusive=1),
            onnx.helper.make_node(
                "GreaterOrEqual", ["topk_scaled_cumsum", "topp"],
                ["topp_filter"]),
            onnx.helper.make_node("Where",
                                  ["topp_filter", "0_casted", "topk_scaled"],
                                  ["topp_topk_scaled"]),
            onnx.helper.make_node("ReduceSum", ["topp_topk_scaled", "[-1]"],
                                  ["topp_topk_scaled_sum"]),
            onnx.helper.make_node(
                "Div", ["topp_topk_scaled", "topp_topk_scaled_sum"],
                ["topp_scaled"]),
    ]

    sampling: list[onnx.NodeProto] = [
            onnx.helper.make_node(
                "Cast", ["topp_scaled"], ["topp_scaled_casted"],
                to=onnx.TensorProto.FLOAT),
            onnx.helper.make_node("Reshape", ["topp_scaled_casted", "B*TK"],
                                  ["topp_scaled_bt"]),
            onnx.helper.make_node("Log", ["topp_scaled_bt"], ["topp_scaled_bt_log"]),
            onnx.helper.make_node(
                "Multinomial", ["topp_scaled_bt_log"], ["sampled_idx_bt_"],
                dtype=onnx.TensorProto.INT64),
            onnx.helper.make_node(
                "GatherElements", ["topk_idx_bt", "sampled_idx_bt_"], ["idx_bt"], axis=1),
            onnx.helper.make_node("Reshape", ["idx_bt", "BT"], ["idx"])
    ]

    return onnx.helper.make_function(
            __domain, "sampling", ["x_", "temp_", "topk_", "topp_"], ["idx"],
            constants + shapes + temperature + topk + topp + sampling,
            __opset_imports)


def make_model_from_state_dict(args: argparse.Namespace
                               ) -> onnx.ModelProto | dict[onnx.ModelProto]:
    state_dict: dict[str, torch.Tensor] = torch.load(args.pt_file, "cpu")
    dtype: str = args.dtype

    torch_onnx_dtype: dict[torch.dtype, int] = {
            torch.float: onnx.TensorProto.FLOAT,
            torch.float16: onnx.TensorProto.FLOAT16,
            torch.bfloat16: onnx.TensorProto.BFLOAT16}
    dtype_table: dict[str, torch.dtype] = {
            "auto": state_dict["emb.weight"].dtype,
            "fp32": torch.float,
            "fp16": torch.float16,
            "bf16": torch.bfloat16}
    np_dtype_table: dict[torch.dtype, np.dtype] = {
            torch.float: np.float32,
            torch.float16: np.float16,
            torch.bfloat16: onnx._custom_element_types.bfloat16
            }
    wkv_dtype_table: dict[torch.dtype, int] = {
            torch.float: onnx.TensorProto.FLOAT,
            torch.float16: onnx.TensorProto.FLOAT,
            torch.bfloat16: onnx.TensorProto.BFLOAT16}

    # Get main dtype
    main_dtype: torch.dtype = dtype_table[dtype]
    onnx_main_dtype: int = torch_onnx_dtype[main_dtype]
    onnx_wkv_dtype: int = wkv_dtype_table[main_dtype]
    # Get embedding dimension
    dim: int = state_dict["emb.weight"].shape[1]
    # vocab size
    vocab_size: int = state_dict["emb.weight"].shape[0]
    # head size
    head_size: int = state_dict["blocks.0.att.r_k"].shape[1]
    n_head: int = dim // head_size

    # Get number of layers
    nlayers: int = 0
    while True:
        for k in state_dict.keys():
            if k.startswith(f"blocks.{nlayers}"):
                nlayers += 1
                continue
        break

    # Is using DeepEmbed
    is_deepemb: bool = "blocks.0.ffn.s_emb.weight" in state_dict.keys()

    normalize_function: onnx.FunctionProto = make_normalize()
    time_shift_function: onnx.FunctionProto = make_time_shift()
    linear_function: onnx.FunctionProto = make_linear()
    lerp_function: onnx.FunctionProto = make_lerp()
    loramlp_functions: list[onnx.FunctionProto] = make_loramlp()
    wkv7_function: onnx.FunctionProto = make_wkv7(wkv_dtype_table[main_dtype])
    time_mix_functions: list[onnx.FunctionProto] = make_time_mix(
            dim, head_size, dim)
    channel_mix_functions: onnx.FunctionProto = make_channel_mix(dim)
    block_functions: list[onnx.FunctionProto] = make_block()

    onnx.checker.check_function(normalize_function)
    onnx.checker.check_function(time_shift_function)
    onnx.checker.check_function(linear_function)
    onnx.checker.check_function(lerp_function)
    _ = [onnx.checker.check_function(loramlp_function)
         for loramlp_function in loramlp_functions]
    onnx.checker.check_function(wkv7_function)
    _ = [onnx.checker.check_function(time_mix_function)
         for time_mix_function in time_mix_functions]
    _ = [onnx.checker.check_function(channel_mix_function)
         for channel_mix_function in channel_mix_functions]
    _ = [onnx.checker.check_function(block_function)
         for block_function in block_functions]

    # Obtain TensorProtos of parameters
    tensor_proto_state_dict: dict[str, onnx.TensorProto] = {}
    parameters: dict[str, onnx.NodeProto] = {}
    for k in list(state_dict.keys()):
        tensor: np.ndarray = state_dict[k].detach(
                ).cpu().to(torch.float).numpy()
        tensor_proto_state_dict[k] = onnx.numpy_helper.from_array(
                tensor.astype(np_dtype_table[main_dtype]), f"{k}")

    # Input/Output value info
    x_value_info: onnx.ValueInfoProto = onnx.helper.make_tensor_value_info(
            "x", onnx.TensorProto.INT64, ["batch", "seq"])
    head_value_info: onnx.ValueInfoProto = onnx.helper.make_tensor_value_info(
            "head", onnx_main_dtype,
            ["batch", "seq", vocab_size])
    state_value_infos: list[onnx.ValueInfoProto] = []
    next_value_infos: list[onnx.ValueInfoProto] = []

    for i in range(nlayers):
        state_value_infos.append(onnx.helper.make_tensor_value_info(
            f"x_tmix_last_{i}", onnx_main_dtype, ["batch", 1, dim]))
        state_value_infos.append(onnx.helper.make_tensor_value_info(
            f"wkv_state_{i}", onnx_wkv_dtype,
            ["batch", n_head, head_size, head_size]))
        state_value_infos.append(onnx.helper.make_tensor_value_info(
            f"x_cmix_last_{i}", onnx_main_dtype, ["batch", 1, dim]))
        next_value_infos.append(onnx.helper.make_tensor_value_info(
            f"x_tmix_next_{i}", onnx_main_dtype, ["batch", 1, dim]))
        next_value_infos.append(onnx.helper.make_tensor_value_info(
            f"wkv_next_{i}", onnx_wkv_dtype,
            ["batch", n_head, head_size, head_size]))
        next_value_infos.append(onnx.helper.make_tensor_value_info(
            f"x_cmix_next_{i}", onnx_main_dtype, ["batch", 1, dim]))

    emb: onnx.NodeProto = onnx.helper.make_node(
            "Gather", ["emb.weight", "x"], ["emb"])

    semb: list[onnx.NodeProto] = []
    for i in range(nlayers):
        if not is_deepemb:
            break

        semb.append(onnx.helper.make_node(
            "Transpose", [f"blocks.{i}.ffn.s_emb_x.weight"],
            [f"blocks.{i}.ffn.s_emb_x.weight_t"]))
        semb.append(onnx.helper.make_node(
            "MatMul", ["emb.weight", f"blocks.{i}.ffn.s_emb_x.weight_t"],
            [f"blocks.{i}.ffn.s_emb_bias"]))
        semb.append(onnx.helper.make_node(
            "Add", [f"blocks.{i}.ffn.s_emb.weight",
                    f"blocks.{i}.ffn.s_emb_bias"],
            [f"blocks.{i}.ffn.s_emb.real_weight"]))
        semb.append(onnx.helper.make_node(
            "Gather", [f"blocks.{i}.ffn.s_emb.real_weight", "x"],
            [f"blocks.{i}.ffn.semb"]))

    if is_deepemb:
        block0: onnx.NodeProto = onnx.helper.make_node(
                "block0_a",
                ["emb", "x_tmix_last_0", "wkv_state_0", "x_cmix_last_0",
                 "blocks.0.ln0.weight", "blocks.0.ln0.bias",
                 "blocks.0.ln1.weight", "blocks.0.ln1.bias",
                 "blocks.0.ln2.weight", "blocks.0.ln2.bias",
                 "blocks.0.att.x_r", "blocks.0.att.x_w", "blocks.0.att.x_k",
                 "blocks.0.att.x_v", "blocks.0.att.x_a", "blocks.0.att.x_g",
                 "blocks.0.att.w1", "blocks.0.att.w2", "blocks.0.att.w0",
                 "blocks.0.att.a1", "blocks.0.att.a2", "blocks.0.att.a0",
                 "blocks.0.att.g1", "blocks.0.att.g2", "blocks.0.att.k_k",
                 "blocks.0.att.k_a", "blocks.0.att.r_k",
                 "blocks.0.att.receptance.weight", "blocks.0.att.key.weight",
                 "blocks.0.att.value.weight", "blocks.0.att.output.weight",
                 "blocks.0.att.ln_x.weight", "blocks.0.att.ln_x.bias",
                 "blocks.0.ffn.x_k",
                 "blocks.0.ffn.key.weight", "blocks.0.ffn.value.weight",
                 "blocks.0.ffn.s1", "blocks.0.ffn.semb",
                 "blocks.0.ffn.s2", "blocks.0.ffn.s0"],
                ["emb0", "v_first",
                 "x_tmix_next_0", "wkv_next_0", "x_cmix_next_0"],
                domain=__domain)
    else:
        block0: onnx.NodeProto = onnx.helper.make_node(
                "block0",
                ["emb", "x_tmix_last_0", "wkv_state_0", "x_cmix_last_0",
                 "blocks.0.ln0.weight", "blocks.0.ln0.bias",
                 "blocks.0.ln1.weight", "blocks.0.ln1.bias",
                 "blocks.0.ln2.weight", "blocks.0.ln2.bias",
                 "blocks.0.att.x_r", "blocks.0.att.x_w", "blocks.0.att.x_k",
                 "blocks.0.att.x_v", "blocks.0.att.x_a", "blocks.0.att.x_g",
                 "blocks.0.att.w1", "blocks.0.att.w2", "blocks.0.att.w0",
                 "blocks.0.att.a1", "blocks.0.att.a2", "blocks.0.att.a0",
                 "blocks.0.att.g1", "blocks.0.att.g2", "blocks.0.att.k_k",
                 "blocks.0.att.k_a", "blocks.0.att.r_k",
                 "blocks.0.att.receptance.weight", "blocks.0.att.key.weight",
                 "blocks.0.att.value.weight", "blocks.0.att.output.weight",
                 "blocks.0.att.ln_x.weight", "blocks.0.att.ln_x.bias",
                 "blocks.0.ffn.x_k",
                 "blocks.0.ffn.key.weight", "blocks.0.ffn.value.weight"],
                ["emb0", "v_first",
                 "x_tmix_next_0", "wkv_next_0", "x_cmix_next_0"],
                domain=__domain)
    blocks: list[onnx.NodeProto] = [block0]
    for i in range(1, nlayers):
        if is_deepemb:
            blocks.append(onnx.helper.make_node(
                "block_a",
                [f"emb{i - 1}", "v_first",
                 f"x_tmix_last_{i}", f"wkv_state_{i}", f"x_cmix_last_{i}",
                 f"blocks.{i}.ln1.weight", f"blocks.{i}.ln1.bias",
                 f"blocks.{i}.ln2.weight", f"blocks.{i}.ln2.bias",
                 f"blocks.{i}.att.x_r", f"blocks.{i}.att.x_w",
                 f"blocks.{i}.att.x_k", f"blocks.{i}.att.x_v",
                 f"blocks.{i}.att.x_a", f"blocks.{i}.att.x_g",
                 f"blocks.{i}.att.w1", f"blocks.{i}.att.w2", f"blocks.{i}.att.w0",
                 f"blocks.{i}.att.a1", f"blocks.{i}.att.a2", f"blocks.{i}.att.a0",
                 f"blocks.{i}.att.v1", f"blocks.{i}.att.v2", f"blocks.{i}.att.v0",
                 f"blocks.{i}.att.g1", f"blocks.{i}.att.g2", f"blocks.{i}.att.k_k",
                 f"blocks.{i}.att.k_a", f"blocks.{i}.att.r_k",
                 f"blocks.{i}.att.receptance.weight", f"blocks.{i}.att.key.weight",
                 f"blocks.{i}.att.value.weight", f"blocks.{i}.att.output.weight",
                 f"blocks.{i}.att.ln_x.weight", f"blocks.{i}.att.ln_x.bias",
                 f"blocks.{i}.ffn.x_k",
                 f"blocks.{i}.ffn.key.weight", f"blocks.{i}.ffn.value.weight",
                 f"blocks.{i}.ffn.s1", f"blocks.{i}.ffn.semb",
                 f"blocks.{i}.ffn.s2", f"blocks.{i}.ffn.s0"],
                [f"emb{i}", f"x_tmix_next_{i}", f"wkv_next_{i}",
                 f"x_cmix_next_{i}"], domain=__domain))
        else:
            blocks.append(onnx.helper.make_node(
                "block",
                [f"emb{i - 1}", "v_first",
                 f"x_tmix_last_{i}", f"wkv_state_{i}", f"x_cmix_last_{i}",
                 f"blocks.{i}.ln1.weight", f"blocks.{i}.ln1.bias",
                 f"blocks.{i}.ln2.weight", f"blocks.{i}.ln2.bias",
                 f"blocks.{i}.att.x_r", f"blocks.{i}.att.x_w",
                 f"blocks.{i}.att.x_k", f"blocks.{i}.att.x_v",
                 f"blocks.{i}.att.x_a", f"blocks.{i}.att.x_g",
                 f"blocks.{i}.att.w1", f"blocks.{i}.att.w2", f"blocks.{i}.att.w0",
                 f"blocks.{i}.att.a1", f"blocks.{i}.att.a2", f"blocks.{i}.att.a0",
                 f"blocks.{i}.att.v1", f"blocks.{i}.att.v2", f"blocks.{i}.att.v0",
                 f"blocks.{i}.att.g1", f"blocks.{i}.att.g2", f"blocks.{i}.att.k_k",
                 f"blocks.{i}.att.k_a", f"blocks.{i}.att.r_k",
                 f"blocks.{i}.att.receptance.weight", f"blocks.{i}.att.key.weight",
                 f"blocks.{i}.att.value.weight", f"blocks.{i}.att.output.weight",
                 f"blocks.{i}.att.ln_x.weight", f"blocks.{i}.att.ln_x.bias",
                 f"blocks.{i}.ffn.x_k",
                 f"blocks.{i}.ffn.key.weight", f"blocks.{i}.ffn.value.weight"],
                [f"emb{i}", f"x_tmix_next_{i}", f"wkv_next_{i}",
                 f"x_cmix_next_{i}"], domain=__domain))

    ln_out: onnx.NodeProto = onnx.helper.make_node(
            "LayerNormalization",
            [f"emb{nlayers - 1}", "ln_out.weight", "ln_out.bias"], ["ln_out"])
    head: onnx.NodeProto = onnx.helper.make_node(
            "linear", ["ln_out", "head.weight"], ["head"], domain=__domain)

    sampling_function: list[onnx.FunctionProto] = []
    if args.sampling:
        if args.topk == -1:
            args.topk = vocab_size

        assert 0.0 < args.temperature
        assert (0 < args.topk) and (args.topk <= vocab_size)
        assert (0.0 < args.topp) and (args.topp <= 1.0)

        y_value_info: onnx.ValueInfoProto = onnx.helper.make_tensor_value_info(
                "y", onnx.TensorProto.INT64, ["batch", "seq"])

        sampling: list[onnx.NodeProto] = [
                onnx.helper.make_node("Constant", [], ["temperature"],
                                      value_float=args.temperature),
                onnx.helper.make_node(
                    "Constant", [], ["topk"],
                    value_int=args.topk),
                onnx.helper.make_node("Constant", [], ["topp"],
                                      value_float=args.topp),
                onnx.helper.make_node(
                    "sampling", ["head", "temperature", "topk", "topp"], ["y"],
                    domain=__domain)
        ]
        sampling_function.append(make_sampling(torch_onnx_dtype[main_dtype]))
        rwkv_lm: onnx.GraphProto = onnx.helper.make_graph(
                list(parameters.values()) + [emb] + semb + blocks + [
                    ln_out, head] + sampling,
                "RWKV7-LM",
                [x_value_info] + state_value_infos,
                [y_value_info] + next_value_infos,
                initializer=list(tensor_proto_state_dict.values()))
    else:
        rwkv_lm: onnx.GraphProto = onnx.helper.make_graph(
                list(parameters.values()) + [emb] + semb + blocks + [ln_out, head],
                "RWKV7-LM",
                [x_value_info] + state_value_infos,
                [head_value_info] + next_value_infos,
                initializer=list(tensor_proto_state_dict.values()))

    rwkv_lm_model: onnx.ModelProto = onnx.helper.make_model(
            rwkv_lm, opset_imports=__opset_imports,
            functions=[normalize_function, time_shift_function, lerp_function,
                       linear_function] + loramlp_functions + [wkv7_function]
            + time_mix_functions +
            channel_mix_functions + block_functions + sampling_function
            )

    # onnx.checker.check_model(rwkv_lm_model, full_check=True)

    return rwkv_lm_model


def main():
    torch.set_default_device("cpu")

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="verbose output",
                        action="store_true")
    parser.add_argument("-t", "--dtype", help="data type", default="fp32",
                        choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("-s", "--sampling", help="include sampling function",
                        action="store_true")
    parser.add_argument("--temperature", help="temperature for sampling",
                        default=0.3, type=float)
    parser.add_argument(
            "--topk", help="TopK for sampling, -1 to disable",
            default=-1, type=int)
    parser.add_argument("--topp", help="TopP for sampling",
                        default=0.3, type=float)
    parser.add_argument("pt_file",
                        help="A PyTorch file which contains state dict.")
    parser.add_argument("onnx_file",
                        help="The ONNX file name which will be saved.")

    args: argparse.Namespace = parser.parse_args()
    model: onnx.ModelProto = make_model_from_state_dict(args)
    if args.verbose:
        print(model)
    onnx.save_model(
            model, args.onnx_file, save_as_external_data=True,
            location=f"{args.onnx_file}.data")
    onnx.checker.check_model(pathlib.Path(args.onnx_file), full_check=True)


if __name__ == "__main__":
    main()
