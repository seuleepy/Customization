import os
os.environ["MKL_NUM_THREADS"] = "2" # export MKL_NUM_THREADS=2
os.environ["NUMEXPR_NUM_THREADS"] = "2" # export NUMEXPR_NUM_THREADS=2
os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=2
from cleanfid import fid
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path_1",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_path_2",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--result_path",
        type=str,
    )
    args = parser.parse_args()

    return args

args = parse_args()

score = fid.compute_kid(args.data_path_1, args.data_path_2)
print(score)

with open (f"{args.result_path}/kid_score.txt", 'w') as f:
    f.write(f"KID score: {score}")