import argparse
import os
from collections import Counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_dir", type=str, default="./data/dscript/data/pairs")
    parser.add_argument('--processed_dir', type=str, default="./data/dscript/processed")
    parser.add_argument('--top', type=int, default=8000)
    args = parser.parse_args()
    pair_fns = os.listdir(args.pair_dir)

    for pairfn in pair_fns:
        organism = pairfn.split("_")[0].strip()
        pairfp = os.path.join(args.pair_dir, pairfn)
        with open(pairfp, "r") as f:
            ppi_lines = f.readlines()
        tot = len(ppi_lines) * 2

        pro_list = []
        for ppiline in ppi_lines:
            fpro, spro, label = ppiline.split()
            pro_list.append(fpro)
            pro_list.append(spro)
        pro_counter = Counter(pro_list)

        print(f"===={pairfn}====")
        mostcommon = pro_counter.most_common(args.top)
        common_num = sum(c[1] for c in mostcommon)
        print(f"Top {args.top} proteins appereas {common_num} times, ratio {round(common_num/tot, 3)}")

