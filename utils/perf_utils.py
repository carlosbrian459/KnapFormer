import torch


def get_peak_tflops_per_second():
    device_name = torch.cuda.get_device_name()

    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835
        elif "PCIe" in device_name:
            return 756
        else:  # for H100 SXM and other variants
            return 989
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989
    elif "B200" in device_name:
        # data from https://nvdam.widen.net/s/wwnsxrhm2w/blackwell-datasheet-3384703
        return 4500
    else:
        raise ValueError(f"Unknown device: {device_name}")
