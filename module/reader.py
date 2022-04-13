import os
import torch
import random
import numpy as np

from fairseq.data import FairseqDataset


class PPIDataset(FairseqDataset):
    def __init__(self, split, args):
        self.split = split
        self.max_len = args.max_len
        self.emb_dim = args.emb_dim

        # 读取序列文件
        with open(os.path.join(args.data_dir, "seqs", split+".fasta"), "r") as f:
            self.acid_seqs = dict(
                [line.strip().split("\t") for line in f.readlines()])   
        # 读取样本文件
        with open(os.path.join(args.data_dir, "pairs", split + ".tsv"), "r") as f:
            self.samples = [line.strip().split("\t") for line in f.readlines()]
        # 嵌入文件夹
        self.emb_sub_dir = os.path.join(args.data_dir, "embs", split)
        # 数据集中的unique蛋白质
        self.proteins = set([sample[0] for sample in self.samples]) | set([sample[1] for sample in self.samples])
        # 编码文件buffer
        self.embs_buffer = {}
        self.embs_buffer_size = 10000
       
        # 统计长度信息等
        self.count_statistics()

    def get_embeds(self, pro):
        if pro in self.embs_buffer:
            emb = self.embs_buffer[pro]
        else:
            emb = np.load(os.path.join(self.emb_sub_dir, pro+".npy"))
            if len(self.embs_buffer) >= self.embs_buffer_size:
                # 随机丢弃一个
                droped_pro = random.choice(list(self.embs_buffer.keys()))
                self.embs_buffer.pop(droped_pro)
            self.embs_buffer[pro] = emb
        return emb

    def count_statistics(self):
        num_of_unique_proteins = len(self.proteins)
        num_of_psamples = len([sample for sample in self.samples if sample[-1] == "1"])
        num_of_fsamples = len(self.samples) - num_of_psamples

        max_acid_seq_length = 0
        min_acid_seq_length = 1e5
        pro_of_max_len = pro_of_min_len = None
        avg_acid_seq_length = 0
        for pro in self.proteins:
            acid_seq = self.acid_seqs[pro]
            if len(acid_seq) > max_acid_seq_length:
                max_acid_seq_length = len(acid_seq)
                pro_of_max_len = pro
            if len(acid_seq) < min_acid_seq_length:
                min_acid_seq_length = len(acid_seq)
                pro_of_min_len = pro
            avg_acid_seq_length += len(acid_seq)
        avg_acid_seq_length = avg_acid_seq_length / len(self.proteins)

        print(f"========{self.split} Dataset Statistics========")
        print(f"Number of Positive Samples: {num_of_psamples}")
        print(f"Number of Negative Samples: {num_of_fsamples}")
        print(f"Number of Unique Proteins: {num_of_unique_proteins}")
        print(f"Proten {pro_of_max_len} Has Max Length: {max_acid_seq_length}")
        print(f"Proten {pro_of_min_len} Has Min Acids: {min_acid_seq_length}")
        print(f"Avg Acids Squence Length: {round(avg_acid_seq_length, 3)}")

    def __getitem__(self, index):
        fpro, spro, label = self.samples[index]
        fpro_seq = self.acid_seqs[fpro]
        spro_seq = self.acid_seqs[spro]

        fpro_emb = self.get_embeds(fpro)
        spro_emb = self.get_embeds(spro)

        assert len(fpro_seq) == len(fpro_emb)
        assert len(spro_seq) == len(spro_emb)
        #! padding到最大长度，后续为提升效率可改成padding到批最大长度
        fpro_emb = np.concatenate(
            (fpro_emb, np.zeros((self.max_len-len(fpro_emb), self.emb_dim))), axis=0)
        spro_emb = np.concatenate(
            (spro_emb, np.zeros((self.max_len-len(spro_emb), self.emb_dim))), axis=0)

        sample_dict = {
            "index": index,
            "fpro": fpro,
            "spro": spro,
            "femb": fpro_emb,
            "semb": spro_emb,
            "fseq": fpro_seq,
            "sseq": spro_seq,
            "label": label}
        return sample_dict
    
    def __len__(self):
        return len(self.samples)
    
    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        fpro, spro, _ = self.samples[index]
        return len(self.acid_seqs[fpro]) + len(self.acid_seqs[spro])

    def collater(self, samples):
        labels = torch.LongTensor([int(sample['label']) for sample in samples])
        fst_embs = [sample["femb"] for sample in samples]
        sec_embs = [sample["semb"] for sample in samples]
        fst_lens = [len(emb) for emb in fst_embs]
        sec_lens = [len(emb) for emb in sec_embs]

        fst_embs = np.array(fst_embs)
        sec_embs = np.array(sec_embs)

        model_inputs = {
            "fst_embs": torch.Tensor(fst_embs),
            "fst_lens": torch.LongTensor(fst_lens),
            "sec_embs": torch.Tensor(sec_embs),
            "sec_lens": torch.LongTensor(sec_lens)}

        indexs = torch.LongTensor([sample['index'] for sample in samples])
        fpros = [sample["fpro"] for sample in samples]
        spros = [sample["spro"] for sample in samples]
        fseqs = [sample["fseq"] for sample in samples]
        sseqs = [sample["sseq"] for sample in samples]
        data_info = {"indexs": indexs, "fpros": fpros, \
            "spros": spros, "fseqs": fseqs, "sseqs": sseqs}
        return {
            "inputs": model_inputs,
            "labels": labels,
            "infos": data_info}
    
    def shuffle(self):
        random.shuffle(self.samples)