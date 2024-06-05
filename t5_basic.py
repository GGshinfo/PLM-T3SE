#t5 half-encorder
from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using {}".format(device))


def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device)  # move model to GPU
    model = model.eval()  # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    return model, tokenizer


def get_embeddings(model, tokenizer, seqs, per_residue, per_protein, sec_struct,
                   max_residues=4000, max_seq_len=3000, max_batch=1):
    res_names = []

    if sec_struct:
        pass

    results = {"residue_embs": dict(),
               "protein_embs": dict(),
               "sec_structs": dict()
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            print(pdb_ids)
            res_names.append(pdb_ids)
            print(seq_lens)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            # if sec_struct:  # in case you want to predict secondary structure from embeddings
            #     d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)

            for batch_idx, identifier in enumerate(pdb_ids):  # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                if per_residue:  # store per-residue embeddings (Lx1024)
                    results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
                if per_protein:  # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

    # passed_time = time.time() - start
    # avg_time = passed_time / len(results["residue_embs"]) if per_residue else passed_time / len(results["protein_embs"])
    # print('\n############# EMBEDDING STATS #############')
    # print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    # print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    # print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
    #     passed_time / 60, avg_time))
    # print('\n############# END #############')
    return results, res_names

def get_seq(path,filename):
    from Bio.SeqIO import parse
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq

    path = path + filename
    file = open(path)
    records = parse(file, "fasta")  # 解析序列文件的内容，并将该内容作为SeqRecord对象的列表返回。
    # 使用python for循环遍历记录，并打印序列记录(SqlRecord)的属性，例如id，名称，描述，序列数据等
    seqs = []
    names = []
    cnt = 0
    greater_512 = 0
    for record in records:
        names.append(str(record.name) + str(cnt))
        tmp = str(record.seq)
        len(tmp)
        seqs.append(tmp)
        cnt = cnt + 1
    seq_dict = dict(zip(names, seqs))
    print(greater_512)
    print(len(seq_dict))
    return  seq_dict

if __name__=='__main__':
    model, tokenizer = get_T5_model()
    import gc
    gc.collect()
    per_residue = 0
    per_protein = 1
    sec_struct = 0

    #这里更改数据的路径
    path='data_t3se/train_1491/'
    filename='train.txt'
    seq_dict=get_seq(path,filename)
    results, res_names = get_embeddings(model, tokenizer, seq_dict,
                                        per_residue, per_protein, sec_struct)
    import numpy as np
    res_names = np.array(res_names)
    print(res_names[0])
    print(res_names)
    print(results["protein_embs"])
    print(len(results["protein_embs"]))

    import pickle
    # 保存文件
    with open(filename+'_t5.pkl', "wb") as tf:
        pickle.dump(results["protein_embs"], tf)
    tf.close()
    with open(filename+'_t5.pkl', "rb") as tf:
        feature_dict = pickle.load(tf)  # 11262
    feature_N = np.array([item for item in feature_dict.values()])  # 11262,1024
    print(feature_N.shape)
    np.savez(filename+'_t5.npz', feature_N)

