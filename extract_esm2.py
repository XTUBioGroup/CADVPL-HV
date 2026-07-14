from transformers import AutoTokenizer, EsmModel
import torch
import re
import os
import argparse
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Start', add_help=False)
    parser.add_argument('--monomoer_dir', type=str, help='Path for protein monomer directory.')
    parser.add_argument('--local_model_dir', type=str, default=os.environ.get('ESM2_LOCAL_DIR'),
                        help='Local directory of ESM-2. If unset, will try online repo.')
    parser.add_argument('--model_name', type=str, default='facebook/esm2_t33_650M_UR50D',
                        help='HF model name for ESM-2.')
    args = parser.parse_args()

    monomoer_dir = Path(args.monomoer_dir)

    model_dir = args.local_model_dir
    if model_dir and Path(model_dir).exists():
        print(f"[INFO] Loading ESM-2 from local dir: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = EsmModel.from_pretrained(model_dir, local_files_only=True).eval()
    else:
        print(f"[INFO] Local model dir not provided or not found. Falling back to online repo: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = EsmModel.from_pretrained(args.model_name).eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for item in os.listdir(monomoer_dir):
        fasta_path = monomoer_dir / item / f'{item}.fasta'
        token_representation_path = monomoer_dir / item / f'{item}.esm2_tokens'

        if fasta_path.exists() and not token_representation_path.exists():
            print(f'Treating {item}')

            with open(fasta_path, 'r') as h:
                fasta_sequence = h.readlines()[1].strip('\n')

            fasta_sequence = re.sub(r'[UZOB]', 'X', fasta_sequence)

            ids = tokenizer.batch_encode_plus(
                [fasta_sequence],
                add_special_tokens=True,
                padding='longest'
            )

            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)

            with torch.no_grad():
                embedding_rpr = model(input_ids=input_ids, attention_mask=attention_mask)

            protein_esm2_embedding = embedding_rpr.last_hidden_state[0, 1:-1].cpu()

            if protein_esm2_embedding.shape[0] != len(fasta_sequence):
                raise ValueError(
                    f'Length mismatch for {item}: '
                    f'seq_len={len(fasta_sequence)}, '
                    f'emb_len={protein_esm2_embedding.shape[0]}'
                )

            torch.save(protein_esm2_embedding, token_representation_path)
            print(f'Successfully treated {item}')