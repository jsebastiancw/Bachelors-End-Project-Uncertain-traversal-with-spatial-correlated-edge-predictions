# Description: Main file to run the SEVAC simulation standalone
from imports import *
from functions import compare, VPG, VAC, off_policy_VPG, SEVAC
from draw_from_file import draw
from generate_net import generate_all


def main(prefer_cuda: bool = True):
    device = get_device(prefer_cuda=prefer_cuda)
    if device is not None and device.type == 'cuda':
        print(f"[SEVAC] Running with CUDA acceleration on GPU: {torch.cuda.get_device_name(0)}")
    elif device is not None:
        print("[SEVAC] CUDA GPU not detected; running on PyTorch CPU.")
    else:
        print("[SEVAC] PyTorch not installed; running on NumPy CPU fallback.")

    # Options: 'SiouxFalls', 'Anaheim', 'Friedrichshain', 'Winnipeg'
    network = 'Anaheim'
    iterations = 300
    alpha = 0.01

    # Run comparison of VPG, VAC, off-policy VPG, and SEVAC across OD pairs
    compare(network, iterations, alpha, device=device)

    # Other standalone entry points from original code:
    # file_path = 'Networks/Winnipeg/Winnipeg_network.csv'
    # target_path = 'Networks/Winnipeg/Winnipeg_network.csv'
    # draw('output_data/arrays_data_T=10.json')
    # generate_all(file_path, target_path, 0.1)


if __name__ == "__main__":
    main(prefer_cuda=True)
