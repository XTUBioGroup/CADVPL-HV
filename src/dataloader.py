# /root/autodl-tmp/DeepGNHV-master/src/dataloader.py
import os
import re
import math
import json
import argparse
import pickle
import torch
import numpy as np
import dgl
from tqdm import tqdm
from torch.utils.data import Dataset


# ==============================================================================
# 1. 基础工具函数
# ==============================================================================
def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


# ==============================================================================
# 2. PDB 解析（安全版，带 pLDDT）
# ==============================================================================
def read_atoms(file, chain="."):
    """
    返回:
      atoms  : [(x,y,z)]
      ajs    : [aa3]
      plddts : [bfactor]  (AlphaFold PDB B-factor 通常存 pLDDT)
    """
    pattern = re.compile(chain)
    atoms, ajs, plddts = [], [], []

    for line in file:
        line = line.rstrip("\n")
        if not line.startswith("ATOM"):
            continue

        atom_type = line[12:16].strip()
        chain_id = line[21:22]

        if atom_type != "CA" or not re.match(pattern, chain_id):
            continue

        try:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except ValueError:
            continue

        aa = line[17:20].strip()

        try:
            bfactor_str = line[60:66].strip()
            bfactor = float(bfactor_str) if bfactor_str else 0.0
        except ValueError:
            bfactor = 0.0

        atoms.append((x, y, z))
        ajs.append(aa)
        plddts.append(bfactor)

    return atoms, ajs, plddts


# ==============================================================================
# 3. 接触图
# ==============================================================================
def compute_contacts(atoms, threshold):
    contacts = []
    n = len(atoms)
    for i in range(n - 2):
        for j in range(i + 2, n):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i, j))
                contacts.append((j, i))
    return contacts


def knn(atoms, k=5, max_knn_nodes=3000):
    """
    O(N^2) 防护版 kNN
    - N 太大直接返回空，避免爆内存/极慢
    """
    N = len(atoms)
    if N == 0 or N > max_knn_nodes:
        return []

    dist_mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            dist_mat[i, j] = dist(atoms[i], atoms[j])

    index = np.argsort(dist_mat, axis=-1)

    contacts = []
    for i in range(N):
        cnt = 0
        for j in index[i]:
            if j != i and j != i - 1 and j != i + 1:
                contacts.append((i, int(j)))
                cnt += 1
                if cnt == k:
                    break
    return contacts


def pdb_to_cm(file, threshold):
    atoms, aa, plddts = read_atoms(file)
    r_contacts = compute_contacts(atoms, threshold)
    k_contacts = knn(atoms)
    return r_contacts, k_contacts, aa, plddts


# ==============================================================================
# 4. PPI ID 筛选
# ==============================================================================
def collect_protein_ids_from_ppi(ppi_paths, col_idx):
    if isinstance(ppi_paths, str):
        ppi_paths = [ppi_paths]

    use_ids = set()
    for ppi_path in ppi_paths:
        if not os.path.exists(ppi_path):
            continue
        with open(ppi_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > col_idx:
                    use_ids.add(parts[col_idx])
    return use_ids


# ==============================================================================
# 5. 主处理逻辑：生成 features + edges + id2idx
# ==============================================================================
def data_processing(
        root_dir,
        prefix,
        out_dir="processed_data",
        ppi_path=None,
        ppi_col=None,
        distance=10.0,
        clip_plddt=True,
):
    print(f"[{prefix}] 扫描目录: {root_dir}")
    if not os.path.exists(root_dir):
        print(f"[Error] 目录不存在: {root_dir}")
        return

    protein_dirs = [p for p in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, p))]
    print(f"[{prefix}] 原始蛋白数: {len(protein_dirs)}")

    if ppi_path is not None and ppi_col is not None:
        use_ids = collect_protein_ids_from_ppi(ppi_path, ppi_col)
        before = len(protein_dirs)
        protein_dirs = [p for p in protein_dirs if p in use_ids]
        print(f"[{prefix}] PPI 筛选后: {len(protein_dirs)}/{before}")

    os.makedirs(out_dir, exist_ok=True)
    feat_dir = os.path.join(out_dir, f"{prefix}_features")
    os.makedirs(feat_dir, exist_ok=True)

    r_edge_list, k_edge_list, seq_edge_list, kept_ids = [], [], [], []

    for pid in tqdm(protein_dirs, desc=f"[{prefix}] Processing"):
        pdb_path = os.path.join(root_dir, pid, f"{pid}.pdb")
        prot_path = os.path.join(root_dir, pid, f"{pid}.protT5_tokens")
        esm2_path = os.path.join(root_dir, pid, f"{pid}.esm2_tokens")

        if not os.path.exists(pdb_path) or not os.path.exists(prot_path) or not os.path.exists(esm2_path):
            continue

        with open(pdb_path, "r") as f:
            r_c, k_c, aa, plddts = pdb_to_cm(f, distance)

        try:
            x = torch.load(prot_path, map_location="cpu").float()      # ProtT5
            esm2 = torch.load(esm2_path, map_location="cpu").float()   # ESM-2
        except Exception as e:
            print(f"[Warning] {prefix}/{pid} 特征读取失败: {e}")
            continue

        # 兼容有些保存格式可能是 [1, L, D]
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        if esm2.dim() == 3 and esm2.size(0) == 1:
            esm2 = esm2.squeeze(0)

        # 强一致性校验
        if len(aa) != len(plddts):
            print(f"[Skip] {prefix}/{pid}: len(aa) != len(plddts)")
            continue
        if x.shape[0] != len(aa):
            print(f"[Skip] {prefix}/{pid}: protT5长度不匹配, {x.shape[0]} vs {len(aa)}")
            continue
        if esm2.shape[0] != len(aa):
            print(f"[Skip] {prefix}/{pid}: esm2长度不匹配, {esm2.shape[0]} vs {len(aa)}")
            continue

        L = int(x.shape[0])
        seq_edges = [(i, i + 1) for i in range(L - 1)] + [(i + 1, i) for i in range(L - 1)]

        plddt = torch.tensor(plddts, dtype=torch.float32).view(-1, 1) / 100.0
        if clip_plddt:
            plddt = torch.clamp(plddt, 0.0, 1.0)

        torch.save(
            {
                "x": x,          # ProtT5 residue features
                "esm2": esm2,    # ESM-2 residue features
                "plddt": plddt,
            },
            os.path.join(feat_dir, f"{pid}.pt")
        )

        r_edge_list.append(r_c)
        k_edge_list.append(k_c)
        seq_edge_list.append(seq_edges)
        kept_ids.append(pid)

    with open(os.path.join(out_dir, f"{prefix}.protein.rball.edges.pkl"), "wb") as f:
        pickle.dump(r_edge_list, f)
    with open(os.path.join(out_dir, f"{prefix}.protein.knn.edges.pkl"), "wb") as f:
        pickle.dump(k_edge_list, f)
    with open(os.path.join(out_dir, f"{prefix}.protein.seq.edges.pkl"), "wb") as f:
        pickle.dump(seq_edge_list, f)

    with open(os.path.join(out_dir, f"{prefix}.protein.id2idx.json"), "w") as f:
        json.dump({pid: i for i, pid in enumerate(kept_ids)}, f, indent=2)

    print(f"[{prefix}] 完成，最终蛋白数: {len(kept_ids)}")
    print(f"[{prefix}] 输出目录: {out_dir}")


# ==============================================================================
# 6. 单体 Dataset：支持 rec_train.py 和 PPI 共享
# ==============================================================================
class ProteinDatasetDGL(Dataset):
    def __init__(self, processed_dir, prefix, to_device=False, device=None):
        """
        processed_dir: processed_data 根目录
        prefix: human / virus
        to_device: 是否在 __getitem__ 时把图搬到 device
        """
        super().__init__()
        self.processed_dir = processed_dir
        self.prefix = prefix

        self.to_device = bool(to_device)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        id2idx_path = os.path.join(processed_dir, f"{prefix}.protein.id2idx.json")
        if not os.path.exists(id2idx_path):
            raise FileNotFoundError(f"找不到索引文件: {id2idx_path}。请先运行预处理。")

        with open(id2idx_path, "r") as f:
            self.id2idx = json.load(f)

        self.idx2id = [None] * len(self.id2idx)
        for pid, idx in self.id2idx.items():
            self.idx2id[int(idx)] = pid

        self.cache_path = os.path.join(processed_dir, f"{prefix}_dgl_graphs.bin")
        if os.path.exists(self.cache_path):
            print(f"[{prefix}] 直接加载缓存图: {self.cache_path}")
            self.graphs, _ = dgl.load_graphs(self.cache_path)
        else:
            print(f"[{prefix}] 未找到缓存图，开始构图并缓存: {self.cache_path}")
            self.graphs = self._build_and_cache()

    def _build_and_cache(self):
        feat_dir = os.path.join(self.processed_dir, f"{self.prefix}_features")

        with open(os.path.join(self.processed_dir, f"{self.prefix}.protein.seq.edges.pkl"), "rb") as f:
            seq_all = pickle.load(f)
        with open(os.path.join(self.processed_dir, f"{self.prefix}.protein.knn.edges.pkl"), "rb") as f:
            knn_all = pickle.load(f)
        with open(os.path.join(self.processed_dir, f"{self.prefix}.protein.rball.edges.pkl"), "rb") as f:
            rball_all = pickle.load(f)

        graphs = []
        for i, pid in enumerate(self.idx2id):
            feat_path = os.path.join(feat_dir, f"{pid}.pt")
            if not os.path.exists(feat_path):
                raise FileNotFoundError(f"缺失特征文件: {feat_path}")

            data = torch.load(feat_path, map_location="cpu")
            x = data["x"]
            esm2 = data["esm2"]
            plddt = data.get("plddt", torch.ones((x.shape[0], 1), dtype=torch.float32))

            if x.dim() == 3 and x.size(0) == 1:
                x = x.squeeze(0)
            if esm2.dim() == 3 and esm2.size(0) == 1:
                esm2 = esm2.squeeze(0)

            if esm2.shape[0] != x.shape[0]:
                raise ValueError(f"{pid} 的 esm2 与 protT5 长度不一致: {esm2.shape[0]} vs {x.shape[0]}")

            g = dgl.heterograph(
                {
                    ("amino_acid", "SEQ", "amino_acid"): seq_all[i],
                    ("amino_acid", "STR_KNN", "amino_acid"): knn_all[i],
                    ("amino_acid", "STR_DIS", "amino_acid"): rball_all[i],
                },
                num_nodes_dict={"amino_acid": int(x.shape[0])},
            )

            g.nodes["amino_acid"].data["x"] = x
            g.nodes["amino_acid"].data["esm2"] = esm2
            g.nodes["amino_acid"].data["plddt"] = plddt
            graphs.append(g)

        dgl.save_graphs(self.cache_path, graphs)
        print(f"[{self.prefix}] 图缓存已保存到: {self.cache_path}")
        return graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        if self.to_device:
            return g.to(self.device)
        return g


# ==============================================================================
# 7. Protein Collate (For Pretraining / RecNet)
# ==============================================================================
def protein_collate(samples):
    """
    用于单体预训练的 collate_fn
    """
    return dgl.batch(samples)


# ==============================================================================
# 8. PPI Pair Dataset (For PPI Classification)
# ==============================================================================
class PPIPairDataset(Dataset):
    """
    读取成对的相互作用文件 (HumanID, VirusID, Label)
    并通过索引从 Human/Virus Dataset 中直接获取图对象。
    """

    def __init__(self, file_paths, human_ds, virus_ds):
        """
        file_paths: list of strings, 训练/验证/测试文件的路径
        human_ds: ProteinDatasetDGL 实例 (包含所有 Human 图)
        virus_ds: ProteinDatasetDGL 实例 (包含所有 Virus 图)
        """
        super().__init__()
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        self.human_ds = human_ds
        self.virus_ds = virus_ds
        self.pairs = []

        print(f"Loading PPI pairs from {len(file_paths)} files...")

        for fp in file_paths:
            if not os.path.exists(fp):
                print(f"[Warning] PPI file not found: {fp}")
                continue

            with open(fp, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue

                    hid, vid, label = parts[0], parts[1], float(parts[2])

                    if hid in human_ds.id2idx and vid in virus_ds.id2idx:
                        h_idx = human_ds.id2idx[hid]
                        v_idx = virus_ds.id2idx[vid]
                        self.pairs.append((h_idx, v_idx, label))

        print(f"Loaded {len(self.pairs)} valid pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        h_idx, v_idx, label = self.pairs[idx]
        g_h = self.human_ds[h_idx]
        g_v = self.virus_ds[v_idx]
        return g_h, g_v, label


# ==============================================================================
# 9. PPI Pair Collate (For PPI Training)
# ==============================================================================
def ppi_pair_collate(batch):
    """
    用于 PPI 训练的 collate_fn
    batch: list of (g_h, g_v, label)
    return:
        batched_g_h: DGLHeteroGraph
        batched_g_v: DGLHeteroGraph
        labels: torch.FloatTensor, shape [B, 1]
    """
    g_h_list, g_v_list, labels = zip(*batch)

    batched_g_h = dgl.batch(g_h_list)
    batched_g_v = dgl.batch(g_v_list)

    labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    return batched_g_h, batched_g_v, labels


# ==============================================================================
# 10. CLI 入口 (预处理用)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--ppi", default=None)
    parser.add_argument("--ppi_col", type=int, default=None)
    parser.add_argument("--out_dir", default="processed_data")
    parser.add_argument("--distance", type=float, default=10.0)
    args = parser.parse_args()

    ppi_paths = args.ppi.split(",") if args.ppi else None

    data_processing(
        root_dir=args.dir,
        prefix=args.prefix,
        out_dir=args.out_dir,
        ppi_path=ppi_paths,
        ppi_col=args.ppi_col,
        distance=args.distance,
    )