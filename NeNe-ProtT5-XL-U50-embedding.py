#!/usr/bin/env python
# coding: utf-8
"""

Usage: ProtT5-XL-U50-embedding.py <input fasta file>'

"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
import sys
import re
import argparse
from tqdm import tqdm
from datetime import datetime
from Bio import SeqIO
import torch
from transformers import T5EncoderModel, T5Tokenizer
import gc


def natural_key(string_):
    '''
    Define sort key that is integer-aware
    '''
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def find_csv(userpath):

    def is_csv(filename):
        fna = filename.lower()
        return fna.endswith(".csv")

    csv_filenames = sorted([os.path.join(userpath, path) for path in os.listdir(userpath) if is_csv(path)], key=natural_key)
    return csv_filenames

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print(
            "\n Time taken: %i hours %i minutes and %s seconds."
            % (thour, tmin, round(tsec, 2))
        )


parser = argparse.ArgumentParser(
    description="\n Calculate average ProtT5-XL-U50 1024 vector representation for a set of FASTA sequences.\n",
    epilog="\n \n",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("sequences", help="path to the FASTA file")
parser.add_argument("-nogpu", dest="usegpu", help="do not use GPU [default: use if available]", action="store_false", default=True, required=False)

args = parser.parse_args()

if os.access(args.sequences, os.R_OK):

    FastaFile = open(args.sequences, "r")
    nseq = 0
    for line in FastaFile:
        if line.startswith(">"):
            nseq += 1
    FastaFile.close()

    start_time = timer(None)
    stripped = os.path.splitext(args.sequences)[0]
    if not os.path.exists(stripped):
        os.makedirs(stripped)
    # Load the vocabulary and ProtT5-XL-UniRef50 Model
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50", do_lower_case=False
    )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    gc.collect()

    # Load the model into the GPU if avilabile and switch to inference mode
    if args.usegpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.half().to(device)
        model = model.half().eval()
    else:
        device = torch.device('cpu')
        model = model.to(device)
        model = model.eval()

    fasta_sequences = SeqIO.parse(open(args.sequences), "fasta")
    with tqdm(total=nseq, desc="  Embedding", ncols=120, mininterval=3) as pbar:
        for fasta in fasta_sequences:
            sequences = []
            name, seq = fasta.id.split(" ")[0], str(fasta.seq)
            # delete * signs prodigal puts at the end of sequence
            seq = re.sub("\*", "", seq)
            if len(seq) > 4000:
                seq = seq[:4000]
            slen = len(seq)
#            sequences.append(" ".join(str(fasta.seq)))
            sequences.append(" ".join(seq))
            name = re.sub("\|", "_", name)
            name = re.sub("\/", "_", name)
            # map rarely occured amino acids (U,Z,O,J,B) to (X)
            sequences = [re.sub(r"[UZOJB]", "X", sequence) for sequence in sequences]
            # Tokenize, encode sequences and load it into the GPU if possibile
            ids = tokenizer.batch_encode_plus(
                sequences, add_special_tokens=True, padding=True
            )
            input_ids = torch.tensor(ids["input_ids"]).to(device)
            attention_mask = torch.tensor(ids["attention_mask"]).to(device)
            # Extracting sequence features and load it into the CPU if needed
            with torch.no_grad():
                embedding = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = embedding.last_hidden_state.cpu().numpy()
            # Remove padding (\<pad>) and special tokens (\</s>) that is added by ProtT5-XL-UniRef50 model
            features = []
            for seq_num in range(len(embedding)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = embedding[seq_num][: seq_len - 1]
                features.append(seq_emd)
            # df = pd.DataFrame(data=np.array(features).flatten().reshape(slen,1024), columns=features)
            df = pd.DataFrame(np.array(features).astype(np.float).reshape(slen, 1024).mean(axis=0))
            df = df.transpose(copy=True)
            df.to_csv(stripped + '/' + name + ".csv", index=False, float_format="%.9f")
            pbar.update()

    timer(start_time)
    print('\n Embedding completed, combining individual protein files ...')

    csv_name = stripped + '_pLM.csv'
    csv_list = find_csv(stripped)

    start_time = timer(None)
    ctr = 0
    with tqdm(total=len(csv_list), desc="  Averaging", ncols=120, mininterval=2) as pbar:
        for csvs in csv_list:
            file_root = os.path.splitext(os.path.basename(csvs))[0]
            z = pd.read_csv(csvs)
            z.insert(0, 'Protein_ID', file_root)
            if ctr == 0:
                CSVFile = z
            else:
                CSVFile = pd.concat([CSVFile, z], axis=0, ignore_index=True)
            ctr = ctr + 1
            pbar.update()

    df = pd.DataFrame(CSVFile.mean(axis=0))
    df = df.transpose()
    df.insert(0, 'Assembly_ID', stripped)
    df.to_csv(csv_name, index=False, float_format='%.9f')
    os.system('rm -rf %s' % stripped)
    timer(start_time)
    # Force Python garbage collection
    gc.collect()
    # Empty the PyTorch CUDA cache
    torch.cuda.empty_cache()

else:
    parser.error(
        '\n !!! Input file "%s" does not exist in this directory !!!\n' % args.sequences
    )
