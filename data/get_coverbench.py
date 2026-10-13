"""Download CoverBench (gated, CC BY-ND 4.0) into data/coverbench/.

Accept the dataset terms at https://huggingface.co/datasets/google/coverbench
and log in (`huggingface-cli login`) first. Do not commit the file: its terms
ask that it not be uploaded where web crawlers can reach it.
"""
import os
import shutil

from huggingface_hub import hf_hub_download

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coverbench")
os.makedirs(out_dir, exist_ok=True)
path = hf_hub_download("google/coverbench", "eval/coverbench.json", repo_type="dataset")
shutil.copy(path, os.path.join(out_dir, "coverbench.json"))
print("CoverBench ready in data/coverbench/coverbench.json")
