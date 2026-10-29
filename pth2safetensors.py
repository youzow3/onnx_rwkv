import argparse
import torch
import safetensors.torch


def main(args: argparse.Namespace):
    t: dict[str, torch.Tensor] = {}
    for k, v in torch.load(args.pth,
                           mmap=True,
                           map_location=torch.device("cpu")).items():
        if k == "head.weight":
            t[k] = v
        else:
            t[f"rwkv7.{k}"] = v

    safetensors.torch.save_file(t, args.st)


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("pth", help="Source PyTorch parameter file")
    parser.add_argument("st", help="Destination Safetensors parameter file")
    main(parser.parse_args())
