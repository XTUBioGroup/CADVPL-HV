import os
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

try:
    from dataloader import PPIPairDataset, ppi_pair_collate, ProteinDatasetDGL
    from models import PPI_Classifier_SchemeA
    from utils import set_seed, check_writable
except ImportError as e:
    raise ImportError(f"导入模块失败: {e}。请确保 dataloader.py, models.py, utils.py 在当前目录。")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# 辅助函数
# ============================================================

def build_weighted_bce_loss(pos_weight_value, device):
    pos_weight = torch.tensor([float(pos_weight_value)], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def info_nce_loss(features, temperature=0.07):
    """
    SimCLR 风格的 InfoNCE 损失
    features: [2*B, D]  前 B 个是视图1，后 B 个是视图2
    """
    B = features.shape[0] // 2
    device = features.device

    # 归一化
    features = F.normalize(features, dim=-1)

    # 相似度矩阵
    sim_matrix = torch.matmul(features, features.T) / temperature  # [2B, 2B]

    # 构造标签：对于每个样本 i，正样本是 (i + B) % (2B)
    labels = torch.arange(B, device=device)
    labels = torch.cat([labels + B, labels], dim=0)  # [2B]

    # 去掉对角线自身
    mask = torch.eye(2*B, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

    # 计算交叉熵损失（正样本视为类别标签）
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def get_fusion_alphas(model):
    """
    兼容 DataParallel / DDP 包装
    返回:
      alpha_h, alpha_v (float or None)
    """
    m = model.module if hasattr(model, "module") else model

    alpha_h = None
    alpha_v = None

    if hasattr(m, "proto_fusion_h") and hasattr(m.proto_fusion_h, "alpha"):
        alpha_h = float(m.proto_fusion_h.alpha.detach().cpu().item())

    if hasattr(m, "proto_fusion_v") and hasattr(m.proto_fusion_v, "alpha"):
        alpha_v = float(m.proto_fusion_v.alpha.detach().cpu().item())

    return alpha_h, alpha_v


def run_evaluate(model, data_loader, device, criterion=None, desc="Evaluate"):
    model.eval()
    all_probs = []
    all_labels = []
    loss_sum = 0.0

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    pbar = tqdm(data_loader, desc=desc, unit="batch", leave=False)

    with torch.no_grad():
        for g_h, g_v, labels in pbar:
            g_h = g_h.to(device)
            g_v = g_v.to(device)
            labels = labels.to(device).float().view(-1, 1)

            with autocast():
                logits = model(g_h, g_v)
                loss = criterion(logits, labels)

            loss_sum += loss.item()
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.float().cpu().numpy().flatten())
            all_labels.extend(labels.float().cpu().numpy().flatten())

    avg_loss = loss_sum / max(1, len(data_loader))

    try:
        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
        auprc = 0.0

    preds = np.array(all_probs) > 0.5
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "loss": avg_loss,
        "auc": auc,
        "auprc": auprc,
        "acc": acc,
        "f1": f1
    }


# ============================================================
# 主训练函数
# ============================================================

def train_ppi(param, args):
    # ====================================================
    # A. 初始化与数据加载
    # ====================================================
    print("=" * 60)
    print("[Init] 正在加载全量蛋白质图缓存...")
    human_ds_global = ProteinDatasetDGL(
        processed_dir=args.processed_dir,
        prefix=args.human_prefix,
        to_device=False
    )
    virus_ds_global = ProteinDatasetDGL(
        processed_dir=args.processed_dir,
        prefix=args.virus_prefix,
        to_device=False
    )
    print("=" * 60)

    print("[PPI] 构建训练集...")
    train_dataset = PPIPairDataset(args.train_files, human_ds_global, virus_ds_global)

    print("[PPI] 构建验证集...")
    val_dataset = PPIPairDataset(args.val_files, human_ds_global, virus_ds_global)

    test_dataset = None
    if args.test_files:
        print("[PPI] 构建测试集...")
        test_dataset = PPIPairDataset(args.test_files, human_ds_global, virus_ds_global)

    # ====================================================
    # B. DataLoader 设置
    # ====================================================
    batch_size = args.batch_size
    accum_steps = args.accumulation_steps
    effective_batch = batch_size * accum_steps

    print("[Config] 启用混合精度训练 (AMP)")
    print(f"[Config] 物理 Batch Size: {batch_size}")
    if accum_steps > 1:
        print(f"[Config] 梯度累积步数: {accum_steps} -> 等效 Batch: {effective_batch}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ppi_pair_collate,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ppi_pair_collate,
        num_workers=4,
        pin_memory=True
    )

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ppi_pair_collate,
            num_workers=4,
            pin_memory=True
        )

    # ====================================================
    # C. 模型初始化
    # ====================================================
    human_weights = None
    virus_weights = None

    if args.pretrained_human_ckpt and os.path.exists(args.pretrained_human_ckpt):
        print(f"[Init] 检测到 Human 预训练权重: {args.pretrained_human_ckpt}")
        human_weights = torch.load(args.pretrained_human_ckpt, map_location='cpu')

    if args.pretrained_virus_ckpt and os.path.exists(args.pretrained_virus_ckpt):
        print(f"[Init] 检测到 Virus 预训练权重: {args.pretrained_virus_ckpt}")
        virus_weights = torch.load(args.pretrained_virus_ckpt, map_location='cpu')

    model = PPI_Classifier_SchemeA(
        param,
        human_state_dict=human_weights,
        virus_state_dict=virus_weights,
        node_type="amino_acid"
    ).to(device)

    print(f"[Model] 初始化完成。Device: {device}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 可训练参数量: {total_params / 1e6:.2f} M")

    alpha_h, alpha_v = get_fusion_alphas(model)
    if alpha_h is not None and alpha_v is not None:
        print(f"[Model] Prototype Fusion alpha init -> human: {alpha_h:.4f}, virus: {alpha_v:.4f}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=param["learning_rate"],
        weight_decay=1e-4
    )

    # ====================================================
    # D. 损失函数与调度器
    # ====================================================
    # 分类损失：可选择 Focal Loss 或 Weighted BCE
    use_focal_loss = args.use_focal_loss
    if use_focal_loss:
        focal_alpha = float(param.get("focal_alpha", 0.3))
        focal_gamma = float(param.get("focal_gamma", 2.0))
        criterion_cls = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print(f"[Loss] Using Focal Loss with alpha={focal_alpha}, gamma={focal_gamma}")
    else:
        pos_weight_value = float(param.get("pos_weight", 1.0))
        criterion_cls = build_weighted_bce_loss(pos_weight_value, device)
        print(f"[Loss] Using Weighted BCE, pos_weight = {pos_weight_value:.4f}")

    # 对比损失超参数
    contrastive_weight = float(param.get("contrastive_weight", 0.1))
    contrastive_temperature = float(param.get("contrastive_temperature", 0.07))
    print(f"[Contrastive] weight = {contrastive_weight}, temperature = {contrastive_temperature}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    scaler = GradScaler()

    # ====================================================
    # E. 早停设置
    # ====================================================
    best_val_auprc = float("-inf")
    best_epoch_for_auprc = 0
    best_epoch = 0

    early_stop_patience = int(args.early_stop_patience)
    early_stop_min_delta = float(args.early_stop_min_delta)
    early_stop_warmup = int(args.early_stop_warmup)
    early_stop_metric = str(args.early_stop_metric).lower()
    early_stop_mode = str(args.early_stop_mode).lower()

    if early_stop_metric not in {"auprc", "auc", "f1", "loss"}:
        raise ValueError(f"--early_stop_metric must be one of: auprc/auc/f1/loss, got {early_stop_metric}")
    if early_stop_mode not in {"max", "min"}:
        raise ValueError(f"--early_stop_mode must be 'max' or 'min', got {early_stop_mode}")

    print(f"[EarlyStop] Enabled: patience={early_stop_patience}, metric={early_stop_metric}, mode={early_stop_mode}")
    no_improve_count = 0

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    alpha_tag = f"alpha{float(param.get('alpha_init', 0.1)):.2f}"
    contrast_tag = f"contrast{contrastive_weight:.2f}"
    output_dir = f"../results/{param['dataset']}/{timestamp}_SchemeA_protofusion_{alpha_tag}_{contrast_tag}"
    check_writable(output_dir)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(param, f, indent=2)

    best_model_path = os.path.join(output_dir, "best_ppi_model.ckpt")

    def is_improved(curr, best, mode, min_delta):
        if mode == "max":
            return curr > best + min_delta
        else:
            return curr < best - min_delta

    def get_metric(metrics_dict, key):
        return float(metrics_dict[key])

    best_score = float("-inf") if early_stop_mode == "max" else float("inf")

    print(f"[PPI] Start training for {param['epochs']} epochs...")

    # ====================================================
    # F. 训练循环
    # ====================================================
    for epoch in range(1, param["epochs"] + 1):
        model.train()
        total_train_loss = 0.0
        total_cls_loss = 0.0
        total_contrast_loss = 0.0

        optimizer.zero_grad(set_to_none=True)
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", unit="batch")

        for step, (g_h, g_v, labels) in enumerate(train_pbar):
            g_h = g_h.to(device)
            g_v = g_v.to(device)
            labels = labels.to(device).float().view(-1, 1)

            with autocast():
                # 返回分类 logits 和对比学习所需的单模态表示
                logits, contrast_dict = model(g_h, g_v, return_contrastive=True)

                # 分类损失
                loss_cls = criterion_cls(logits, labels)

                # 对比损失（分别对人蛋白和病毒蛋白计算）
                feats_h = torch.cat([contrast_dict["z_h_prot"], contrast_dict["z_h_esm"]], dim=0)  # [2B, D]
                loss_contrast_h = info_nce_loss(feats_h, temperature=contrastive_temperature)

                feats_v = torch.cat([contrast_dict["z_v_prot"], contrast_dict["z_v_esm"]], dim=0)
                loss_contrast_v = info_nce_loss(feats_v, temperature=contrastive_temperature)

                loss_contrast = (loss_contrast_h + loss_contrast_v) / 2

                # 总损失
                loss = loss_cls + contrastive_weight * loss_contrast

                if accum_steps > 1:
                    loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # 记录损失（用于显示）
            display_loss = loss.item() * accum_steps if accum_steps > 1 else loss.item()
            total_train_loss += display_loss
            total_cls_loss += loss_cls.item()
            total_contrast_loss += loss_contrast.item()

            train_pbar.set_postfix({
                "loss": f"{display_loss:.4f}",
                "cls": f"{loss_cls.item():.4f}",
                "ctr": f"{loss_contrast.item():.4f}"
            })

        # 处理最后未累积的梯度
        if (step + 1) % accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = total_train_loss / max(1, len(train_loader))
        avg_cls = total_cls_loss / max(1, len(train_loader))
        avg_contrast = total_contrast_loss / max(1, len(train_loader))

        # -------------------------
        # Validation
        # -------------------------
        val_metrics = run_evaluate(
            model,
            val_loader,
            device,
            criterion=criterion_cls,   # 验证时只用分类损失
            desc=f"Epoch {epoch:02d} [Val]"
        )

        scheduler.step(val_metrics["auprc"])

        alpha_h, alpha_v = get_fusion_alphas(model)

        alpha_msg = ""
        if alpha_h is not None and alpha_v is not None:
            alpha_msg = f" | alpha_h: {alpha_h:.4f} | alpha_v: {alpha_v:.4f}"

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_loss:.4f} (Cls: {avg_cls:.4f}, Ctr: {avg_contrast:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Val AUPRC: {val_metrics['auprc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
            f"{alpha_msg}"
        )

        # -------------------------
        # Save Best
        # -------------------------
        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_epoch_for_auprc = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best AUPRC ({best_val_auprc:.4f}) at epoch {best_epoch_for_auprc}! Model Saved.")

        # -------------------------
        # Early Stopping
        # -------------------------
        curr_score = get_metric(val_metrics, early_stop_metric)

        if epoch <= early_stop_warmup:
            if is_improved(curr_score, best_score, early_stop_mode, 0.0):
                best_score = curr_score
                best_epoch = epoch
            continue

        if is_improved(curr_score, best_score, early_stop_mode, early_stop_min_delta):
            best_score = curr_score
            best_epoch = epoch
            no_improve_count = 0
            print(f"[EarlyStop] Improved {early_stop_metric}: {best_score:.6f} at epoch {best_epoch}")
        else:
            no_improve_count += 1
            print(
                f"[EarlyStop] No improvement ({no_improve_count}/{early_stop_patience}) "
                f"on {early_stop_metric} (best={best_score:.6f} at epoch {best_epoch})"
            )

            if no_improve_count >= early_stop_patience:
                print(
                    f"[EarlyStop] Stop training. "
                    f"Best {early_stop_metric}={best_score:.6f} at epoch {best_epoch}."
                )
                break

    # ====================================================
    # G. Final Test
    # ====================================================
    if test_loader:
        print("\n" + "=" * 60)
        print("[PPI] Final Testing...")
        print("=" * 60)

        if os.path.exists(best_model_path):
            print(f"[Test] Loading best model from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        else:
            print("[Warning] Best model checkpoint not found, testing with last epoch weights.")

        alpha_h, alpha_v = get_fusion_alphas(model)
        if alpha_h is not None and alpha_v is not None:
            print(f"[Test] Loaded model fusion alpha -> human: {alpha_h:.4f}, virus: {alpha_v:.4f}")

        test_metrics = run_evaluate(
            model,
            test_loader,
            device,
            criterion=criterion_cls,
            desc="[Final Test]"
        )

        if alpha_h is not None:
            test_metrics["alpha_h"] = alpha_h
        if alpha_v is not None:
            test_metrics["alpha_v"] = alpha_v
        test_metrics["best_val_auprc"] = best_val_auprc
        test_metrics["best_val_auprc_epoch"] = best_epoch_for_auprc

        print(f"\n[Final Result] Best Val AUPRC (save criterion): {best_val_auprc:.4f} @ epoch {best_epoch_for_auprc}")
        print(f"[Final Result] Test Metrics: {json.dumps(test_metrics, indent=2)}")

        with open(os.path.join(output_dir, "test_results.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
    else:
        print("\n[PPI] Training Finished. No Test set provided.")


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 路径参数
    parser.add_argument("--processed_dir", type=str, required=True, help="Path to graph cache dir")
    parser.add_argument("--train_files", type=str, nargs='+', required=True, help="List of train pair files")
    parser.add_argument("--val_files", type=str, nargs='+', required=True, help="List of val pair files")
    parser.add_argument("--test_files", type=str, nargs='+', default=None, help="List of test pair files")

    parser.add_argument("--human_prefix", type=str, default="human")
    parser.add_argument("--virus_prefix", type=str, default="virus")

    # 预训练权重路径
    parser.add_argument("--pretrained_human_ckpt", type=str, default=None)
    parser.add_argument("--pretrained_virus_ckpt", type=str, default=None)

    parser.add_argument("--dataset", type=str, default="PPI_VF_Split")

    # 模型参数
    parser.add_argument("--input_dim", type=int, default=1024)
    parser.add_argument("--esm2_dim", type=int, default=1280)
    parser.add_argument("--protein_dim", type=int, default=128)

    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="residual",
        choices=["prot_only", "esm_only", "concat", "residual"]
    )

    parser.add_argument("--resid_hidden_dim", type=int, default=128)
    parser.add_argument("--resid_num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout_ratio", type=float, default=0.1)

    # Prototype 参数
    parser.add_argument("--num_prototypes", type=int, default=128)
    parser.add_argument("--proto_temperature", type=float, default=1.0)

    # 新增: prototype residual fusion 初始权重
    parser.add_argument("--alpha_init", type=float, default=0.1,
                        help="Initial alpha for prototype-level residual fusion")

    # 训练超参
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)

    # Weighted BCE
    parser.add_argument("--pos_weight", type=float, default=1.0,
                        help="Positive class weight for BCEWithLogitsLoss")

    # 早停参数
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--early_stop_warmup", type=int, default=5)
    parser.add_argument("--early_stop_metric", type=str, default="auprc")
    parser.add_argument("--early_stop_mode", type=str, default="max")

    # 新增：对比学习参数
    parser.add_argument("--contrastive_weight", type=float, default=0.1,
                        help="Weight for contrastive loss (lambda)")
    parser.add_argument("--contrastive_temperature", type=float, default=0.07,
                        help="Temperature for InfoNCE loss")
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="Use Focal Loss instead of BCE")
    parser.add_argument("--focal_alpha", type=float, default=0.3,
                        help="Alpha parameter for Focal Loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma parameter for Focal Loss")

    args = parser.parse_args()
    param = args.__dict__.copy()

    # 兼容 models.py 中使用的参数名
    param["resid_n_layer"] = param["resid_num_layers"]
    param["in_size"] = param["input_dim"]

    set_seed(param["seed"])
    train_ppi(param, args)