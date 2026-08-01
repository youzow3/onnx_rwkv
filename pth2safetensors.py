import argparse
import torch
import safetensors.torch


def main(args: argparse.Namespace):
    safetensors.torch.save_file(torch.load(args.pth, mmap=True), args.st)


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("pth", help="Source PyTorch parameter file")
    parser.add_argument("st", help="Destination Safetensors parameter file")
    main(parser.parse_args())
