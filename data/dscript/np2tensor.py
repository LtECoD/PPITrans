import os
import torch
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", type=str, default='./data/dscript/processed/embs')
    parser.add_argument("--new_emb_dir", type=str, default='./data/dscript/processed/nembs')
    args = parser.parse_args()

    if not os.path.exists(args.new_emb_dir):
        os.mkdir(args.new_emb_dir)
    
    for subdir in os.listdir(args.emb_dir):
        print(f">>>> processing {subdir}")

        fullsubdir = os.path.join(args.emb_dir, subdir)
        new_fullsubdir = os.path.join(args.new_emb_dir, subdir)
        if not os.path.exists(new_fullsubdir):
            os.mkdir(new_fullsubdir)

        for idx, fn in enumerate(os.listdir(fullsubdir)):
            pro = os.path.splitext(fn)[0]

            fp = os.path.join(fullsubdir, fn)
            data = np.load(fp)
            data = torch.Tensor(data)
        
            new_fp = os.path.join(new_fullsubdir, pro+".ckpt")
            torch.save(data, new_fp)

            if (idx+1) % 1000 == 0:
                print(f"processed {idx+1} proteins")
        