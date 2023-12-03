import os
import argparse
import sys
import torch
from multiprocessing import cpu_count


def use_fp32_config():
    for config_file in [
        "32k.json",
        "40k.json",
        "48k.json",
        "48k_v2.json",
        "32k_v2.json",
    ]:
        with open(f"configs/{config_file}", "r") as f:
            strr = f.read().replace("true", "false")
        with open(f"configs/{config_file}", "w") as f:
            f.write(strr)


class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.has_gpu = torch.cuda.is_available()
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
            self.dml,
        ) = self.arg_parse()
        self.instead = ""
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def arg_parse() -> tuple:
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865, help="Listen port")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument("--colab", action="store_true", help="Launch in colab")
        parser.add_argument(
            "--noparallel", action="store_true", help="Disable parallel processing"
        )
        parser.add_argument(
            "--noautoopen",
            action="store_true",
            help="Do not open in browser automatically",
        )
        parser.add_argument(
            "--dml",
            action="store_true",
            help="torch_dml",
        )
        cmd_opts, unknown = parser.parse_known_args() # allows import to jupyter notebook
        print(f"unknown args: {unknown}")

        # cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.colab,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
            cmd_opts.dml,
        )

    # has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
    # check `getattr` and try it for compatibility
    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("Found GPU", self.gpu_name, ", force to fp32")
                self.is_half = False
                use_fp32_config()
            else:
                print("Found GPU", self.gpu_name)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                # + 0.4
            )
        elif self.has_mps():
            print("No supported Nvidia GPU found")
            self.device = self.instead = "mps"
            self.is_half = False
            use_fp32_config()
        else:
            print("No supported Nvidia GPU found")
            self.device = self.instead = "cpu"
            self.is_half = False
            use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 64
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32
        if self.dml:
            print("use DirectML instead")
            if(os.path.exists("runtime\Lib\site-packages\onnxruntime\capi\DirectML.dll")==False):
                try:
                    os.rename("runtime\Lib\site-packages\onnxruntime", "runtime\Lib\site-packages\onnxruntime-cuda")
                except:
                    pass
                try:
                    os.rename("runtime\Lib\site-packages\onnxruntime-dml", "runtime\Lib\site-packages\onnxruntime")
                except:
                    pass
            import torch_directml
            self.device = torch_directml.device(torch_directml.default_device())
            self.is_half = False
        else:
            if self.instead:
                print(f"use {self.instead} instead")
            if(os.path.exists("runtime\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll")==False):
                try:
                    os.rename("runtime\Lib\site-packages\onnxruntime", "runtime\Lib\site-packages\onnxruntime-dml")
                except:
                    pass
                try:
                    os.rename("runtime\Lib\site-packages\onnxruntime-cuda", "runtime\Lib\site-packages\onnxruntime")
                except:
                    pass
        return x_pad, x_query, x_center, x_max
