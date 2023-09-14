import numpy as np
from string import punctuation
import unidecode
import nltk.stem.snowball as sb
from nltk.corpus import stopwords
import pandas as pd
import json


# read file needed
def read_file(filename):
    dataset_val = pd.read_csv(filename, sep='\t')
    # Only take the data we need
    pid_passage_val = dataset_val[['pid', 'passage']].drop_duplicates(subset='pid')
    qid_query_val = dataset_val[['qid', 'queries']].drop_duplicates(subset='qid')

    return dataset_val, pid_passage_val, qid_query_val


def process_dataset(dataset):
    passages = []
    sb_stemmer = sb.SnowballStemmer('english')
    stop_word = stopwords.words('english')
    lis = dataset
    # Extract terms (1-gram) and remove punctuation
    for lin in lis:
        lin = lin.strip('\n').translate(str.maketrans('', '', punctuation)).lower()
        lin = unidecode.unidecode(lin)
        lin = lin.split(' ')
        # Convert to stem
        words = []
        for word in lin:
            word = sb_stemmer.stem(word)
            if not word in stop_word:
                words.append(word)
        passages.append(words)
    return passages


def invert_index(dataset):
    cols = list(dataset)
    data_set = dataset.drop_duplicates(subset=cols[0])
    pid = data_set[cols[0]].tolist()
    passages = data_set[cols[1]].tolist()
    # using the function process_dataset()
    passages_pro = process_dataset(passages)
    invert_dict = {}
    for p, lin in zip(pid, passages_pro):
        for word in lin:
            words = invert_dict.keys()
            count = lin.count(word)
            if word in words:
                invert_dict[word].update({p: count})
            else:
                invert_dict.setdefault(word, {p: count})

    return invert_dict


def cal_tf(invert_dict):
    psg_tf = {}
    for token, pid_num in invert_dict.items():
        # n_t is number of passages where t appears
        for pid, num in pid_num.items():
            tf_val = invert_dict[token][pid]
            if pid in psg_tf.keys():
                psg_tf[pid].update({token: tf_val})
            else:
                psg_tf.setdefault(pid, {token: tf_val})
    return psg_tf


def tfidf_qry(data_queries):
    invert_dict_qry = invert_index(data_queries)
    qry_tf = cal_tf(invert_dict_qry)
    return qry_tf


def cal_ap_ndcg(df, top_n=100):
    df = df.iloc[:top_n]
    num_rel, pre_sum, dcg, idcg = 0, 0, 0, 0
    index = df[df.relevancy == 1.0].index.tolist()
    df_ideal = df.sort_values(by='relevancy', ascending=False)
    rel_list1 = df["relevancy"].tolist()
    rel_list2 = df_ideal["relevancy"].tolist()
    df_rel = df.iloc[index]
    if len(df_rel) != 0:
        for i in range(len(df)):
            if rel_list1[i] == 1:
                num_rel += 1
                pre_sum += num_rel / (i + 1)
        ap = pre_sum / num_rel
    else:
        ap = 0
    for i in range(len(df)):
        rel1, rel2 = rel_list1[i], rel_list2[i]
        dcg += (2 ** rel1 - 1) / np.log2(i + 2)
        idcg += (2 ** rel2 - 1) / np.log2(i + 2)
    if idcg != 0:
        ndcg = dcg / idcg
    else:
        ndcg = 0
    return ap, ndcg


def bm25(invert_dict, pid_passage_val, psg_tf, data_qid, data_all, qry_tf):
    k1, k2, b, N = 1.2, 100, 0.75, len(pid_passage_val)
    n = len(data_qid)
    ap_top100, ndcg_top100 = np.zeros(n), np.zeros(n)
    ap_top10, ndcg_top10 = np.zeros(n), np.zeros(n)
    dl, bm25_dict = {}, {}
    bm25_df = pd.DataFrame()
    avdl = 0.0
    # get the length and average length of passage
    for p, ocurr in psg_tf.items():
        d_i = sum(ocurr.values())
        dl.setdefault(p, d_i)
        avdl += d_i
    avdl = avdl / N
    # perform bm25 to each of the pairs
    for i in range(n):
        qid = data_qid[i]
        bm25_dict[qid] = {}
        token_qry = qry_tf[qid]
        index = data_all[data_all.qid == qid].index.tolist()
        pid_lis = data_all.iloc[index, 1].tolist()
        for pid in pid_lis:
            dl_i = dl[str(pid)]
            K = k1 * ((1 - b) + ((b * dl_i) / avdl))
            token_psg = psg_tf[str(pid)]
            words = set(token_qry.keys()).intersection(set(token_psg.keys()))
            score = 0.0
            for token in words:
                f = psg_tf[str(pid)][token]
                qf = qry_tf[qid][token]
                n = len(invert_dict[token])
                score += np.log((N - n + 0.5) / (n + 0.5)) * (((k1 + 1) * f) / (K + f)) * (((k2 + 1) * qf) / (k2 + qf))
            bm25_dict[qid][pid] = score
        # keep the top 100 scores
        bm25_dict[qid] = sorted(bm25_dict[qid].items(), key=lambda x: x[1], reverse=True)[:100]
        # convert dict to data frame
        df_sub = pd.DataFrame(bm25_dict[qid], columns=["pid", "score"])
        df_sub.insert(0, 'qid', qid)
        df_rel = data_all.iloc[index, [1, 4]]
        df = pd.merge(df_sub, df_rel, how='left', on='pid')
        ap_top100[i], ndcg_top100[i] = cal_ap_ndcg(df)
        ap_top10[i], ndcg_top10[i] = cal_ap_ndcg(df, 10)
        bm25_df = pd.concat([bm25_df, df])
    map_top100, mNDCG_top100 = np.mean(ap_top100), np.mean(ndcg_top100)
    map_top10, mNDCG_top10 = np.mean(ap_top10), np.mean(ndcg_top10)
    return bm25_df, map_top100, mNDCG_top100, map_top10, mNDCG_top10


if __name__ == '__main__':
    # Run the function above
    dataset_val, pid_passage_val, qid_query_val = read_file("validation_data.tsv")
    invert_dict = invert_index(pid_passage_val)
    # convert the dict to txt file
    with open('invert_dict.txt', 'w', encoding='utf-8') as f:
        f.write(json.dumps(invert_dict))
    file_inverted = open("invert_dict.txt", 'r')
    invert_dict = json.loads(file_inverted.read())
    file_inverted.close()
    psg_tf = cal_tf(invert_dict)
    qry_tf = tfidf_qry(qid_query_val)
    data_qid = qid_query_val['qid'].tolist()
    bm25_df, map_top100, mNDCG_top100, map_top10, mNDCG_top10 = \
        bm25(invert_dict, pid_passage_val, psg_tf, data_qid, dataset_val, qry_tf)
    print("When processing top 100 scores: ", "Average precision is ", map_top100, "; NDCG is ", mNDCG_top100)
    print("When processing top 10 scores: ", "Average precision is ", map_top10, "; NDCG is ", mNDCG_top10)
    # OUTPUT
    # When processing top 100 scores:  Average precision is  0.2356262174108393 ; NDCG is  0.35477226749267743
    # When processing top 10 scores:  Average precision is  0.22385342348321444 ; NDCG is  0.28602853094266595
