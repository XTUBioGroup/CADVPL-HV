from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import os
import argparse
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Start', add_help=False)
    parser.add_argument('--monomoer_dir', type=str, help='Path for protein monomer directory.')
    # 新增: 支持本地模型目录 (参数优先, 否则读取环境变量 PROT_T5_LOCAL_DIR)
    parser.add_argument('--local_model_dir', type=str, default=os.environ.get('PROT_T5_LOCAL_DIR'),
                        help='Local directory of ProtT5 (e.g., snapshot_download output). If unset, will try online repo.')
    args = parser.parse_args()
    monomoer_dir = Path(args.monomoer_dir)

    model_dir = args.local_model_dir
    if model_dir and Path(model_dir).exists():
        print(f"[INFO] Loading ProtT5 from local dir: {model_dir}")
        tokenizer = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False, local_files_only=True)
        model     = T5EncoderModel.from_pretrained(model_dir, local_files_only=True).eval()
    else:
        print("[INFO] Local model dir not provided or not found. Falling back to online repo: Rostlab/prot_t5_xl_uniref50")
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
        model     = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").eval()
    # "Rostlab/prot_t5_xl_uniref50" can be replaced with the file at "/home/username/.cache/huggingface/hub/models--Rostlab--prot_t5_xl_uniref50/snapshots/xxxxxxxxxxx"
    # You can also implement ProtT5 by following the guidance at https://github.com/agemagician/ProtTrans

    for item in os.listdir(monomoer_dir):
        fasta_path                = monomoer_dir/item/f'{item}.fasta'
        token_representation_path = monomoer_dir/item/f'{item}.protT5_tokens'
        if fasta_path.exists() and not token_representation_path.exists():
            print(f'Treating {item}')
            with open(fasta_path,'r') as h:
                fasta_sequence = h.readlines()[1].strip('\n')
            sequence_list = [fasta_sequence]
            sequence_processed = [' '.join(list(re.sub(r'[UZOB]', 'X', sequence))) for sequence in sequence_list]
            ids = tokenizer.batch_encode_plus(sequence_processed, add_special_tokens=True, padding='longest')
            input_ids = torch.tensor(ids['input_ids'])
            attention_mask = torch.tensor(ids['attention_mask'])
            with torch.no_grad():
                embedding_rpr = model(input_ids=input_ids,attention_mask=attention_mask)
            protein_protT5_embedding = embedding_rpr.last_hidden_state[0,:-1].cpu()
            torch.save(protein_protT5_embedding,token_representation_path)
            print(f'Successfully treated {item}')

