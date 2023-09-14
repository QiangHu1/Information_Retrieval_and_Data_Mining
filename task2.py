import task1
import pandas as pd
import numpy as np
import random
from gensim.models import word2vec
from matplotlib import pyplot as plt


# load file needed and selects relevant and irrelevant samples with ratio 1:10
def sample(filename):
    dataset = pd.read_csv(filename, sep='\t')
    num_samples = 10
    index_pos = dataset[dataset.relevancy == 1.0].index.tolist()
    index_neg = np.arange(len(dataset))
    index_neg = np.delete(index_neg, index_pos)
    train_pos = dataset.iloc[index_pos]
    train_neg = dataset.iloc[index_neg]
    list_qid_pid = train_pos[['qid', 'pid']]
    train_data = train_pos
    for i in range(len(list_qid_pid)):
        qid, pid = list_qid_pid.iloc[i]
        ind_neg = train_neg[train_neg.qid == qid].index.tolist()
        ind_neg = [i for i in ind_neg if i < len(train_neg)]
        if len(ind_neg) >= num_samples:
            neg_samples = random.sample(ind_neg, num_samples)
        else:
            neg_samples = ind_neg
        train_data = pd.concat([train_data, train_neg.iloc[neg_samples]], ignore_index=True)
    train_data = train_data.sample(frac=1.0).reset_index(drop=True)
    return train_data


# preprocessing the queries and passages in the dataset
def processing(dataset):
    passages = dataset['passage'].tolist()
    queries = dataset['queries'].tolist()
    # using the same steps in task 1
    passages_pro = task1.process_dataset(passages)
    queries_pro = task1.process_dataset(queries)
    return passages_pro, queries_pro


#  generates Word2Vec model
def gen_embedding(passages_pro, queries_pro):
    model_pass = word2vec.Word2Vec(passages_pro, sg=1, min_count=1)
    model_query = word2vec.Word2Vec(queries_pro, sg=1, min_count=1)
    return model_pass, model_query


# calculate word embeddings for passages and queries
def cal_embedding(dataset, passages_pro, queries_pro, model_pass, model_query):
    list_vec_p, list_vec_q = [], []
    for i in range(len(dataset)):
        list_pass, list_qry = passages_pro[i], queries_pro[i]
        m, n = len(list_pass), len(list_qry)
        vec_pa, vec_qu = 0, 0
        for j in list_pass:
            vec_pa += model_pass.wv[j]
        for k in list_qry:
            vec_qu += model_query.wv[k]
        if m > 0:
            list_vec_p.append(vec_pa / m)
        else:
            list_vec_p.append(0)
        if n > 0:
            list_vec_q.append(vec_qu / n)
        else:
            list_vec_q.append(0)
    dataset['passage'], dataset['queries'] = list_vec_p, list_vec_q
    return dataset


# build the Logistic Regression model
class LogisticRegression:
    def __init__(self, x_train, y_train, lr, epoch=1500, tol=0.01):
        self.xTr = x_train
        self.yTr = y_train
        self.lr = lr
        self.epoch = epoch
        self.cost = []
        self.tol = tol
        self.b = 0
        self.w = np.zeros(x_train.shape[1])

    # perform the sigmoid function to x sets
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # perdict the scores using the modified wight
    def predict(self, x):
        p = self.sigmoid(x @ self.w + self.b)
        return p

    # store losses for each epoch
    def cur_loss(self):
        x, y = self.xTr, self.yTr
        sig = self.predict(x)
        cost = -(y.T @ np.log(sig) + (1 - y).T @ np.log(1 - sig)) / len(y)
        self.cost.append(cost)

    # update the weight and interception for each training
    def train(self):
        x, y = self.xTr, self.yTr
        for i in range(self.epoch):
            sig = self.predict(x)
            l = sig - y
            m = len(y)
            w_i = (x.T @ l).T / m
            b_i = np.sum(l) / m
            self.w -= self.lr * w_i
            self.b -= self.b * b_i
            self.cur_loss()


# load data and split them
def split_data(dataset):
    m = len(dataset)
    xTr, yTr = np.zeros(shape=(m, 200)), np.zeros(m)
    for i, row in dataset.iterrows():
        vec_pas, vec_qry = row['passage'], row['queries']
        x_i = np.hstack((vec_pas, vec_pas))
        xTr[i], yTr[i] = x_i, row['relevancy']
    return xTr, yTr


# plot the figure to analyze the effect of the learning rate
def eval_lr(lr_list, x, y):
    plt.figure()
    plt.title("Loss versus epoch with different learn rates")
    plt.xlabel("Loss")
    plt.ylabel("epoch")
    for i in lr_list:
        model_i = LogisticRegression(x, y, i)
        model_i.train()
        loss = model_i.cost
        plt.plot(loss, label=f"lr = {i}")
    plt.legend()
    plt.savefig("diff_lr.png")


# using the similar way to calculate the AP and NDCG
def cal_ap_ndcg(df, top_n=100):
    df = df.iloc[:top_n]
    df_ideal = df.sort_values(by='relevancy', ascending=False)
    rel_list = df["relevancy"].tolist()
    rel_list_i = df_ideal["relevancy"].tolist()
    num_rel, pre_sum, dcg, idcg = 0, 0, 0, 0
    for i in range(len(df)):
        rel = rel_list[i]
        dcg += (2 ** rel - 1) / np.log2(i + 2)
        if rel == 1:
            num_rel += 1
            pre_sum += num_rel / (i + 1)
    if num_rel != 0:
        ap = pre_sum / num_rel
    else:
        ap = 0
    for i in range(len(df_ideal)):
        rel_i = rel_list_i[i]
        idcg += (2 ** rel_i - 1) / np.log2(i + 2)
    if idcg != 0:
        ndcg = dcg / idcg
    else:
        ndcg = 0
    return ap, ndcg


# evaluate the model using validation dataset
def perform_val(filename, model):
    val_data = pd.read_csv(filename, sep='\t')
    passages_val, queries_val = processing(val_data)
    model_pass_val, model_query_val = gen_embedding(passages_val, queries_val)
    val_data = cal_embedding(val_data, passages_val, queries_val, model_pass_val, model_query_val)
    val_data.to_pickle("val_data.pickle")
    val_data = pd.read_pickle("val_data.pickle")
    x_val, y_val = split_data(val_data)
    y_prd = model.predict(x_val)
    val_data.insert(val_data.shape[1], "score", y_prd)
    val_data = val_data[['qid', 'pid', "relevancy", "score"]]
    qid_val = val_data['qid'].drop_duplicates().tolist()
    n = len(qid_val)
    ap_top100, ndcg_top100 = np.zeros(n), np.zeros(n)
    for i in range(len(qid_val)):
        qid = qid_val[i]
        sub_df = val_data[val_data.qid == qid].sort_values('score', ascending=False)[0:100]
        ap_top100[i], ndcg_top100[i] = cal_ap_ndcg(sub_df)
    return np.mean(ap_top100), np.mean(ndcg_top100)


# process the testfile generated from candidate_passages_top1000.tsv
def process_testfile(filename, model):
    val_data = pd.read_csv(filename, sep='\t', names=['qid', 'pid', 'queries', 'passage'])
    passages_val, queries_val = processing(val_data)
    model_pass_val, model_query_val = gen_embedding(passages_val, queries_val)
    val_data = cal_embedding(val_data, passages_val, queries_val, model_pass_val, model_query_val)
    val_data.to_pickle("test_data.pickle")
    val_data = pd.read_pickle("test_data.pickle")
    val_data.insert(val_data.shape[1], "relevancy", 0)
    x_val, y_val = split_data(val_data)
    y_prd = model.predict(x_val)
    val_data.insert(val_data.shape[1], "score", y_prd)
    return val_data, y_prd, y_val


# output data for top 100 passages for each query in test-queries.tsv
def output_LR(test_qry, val_data):
    qid_list = test_qry["qid"].tolist()
    df_LR = pd.DataFrame()
    for i in range(len(qid_list)):
        qid = qid_list[i]
        sub_df = val_data[val_data.qid == qid].sort_values('score', ascending=False)[0:100]
        rank = np.arange(len(sub_df)) + 1
        sub_df.insert(1, 'rank', rank)
        df_LR = pd.concat([df_LR, sub_df], sort=True)
    df_LR_new = pd.DataFrame(columns=['qid', 'pid', 'rank', 'score'])
    df_LR_new = pd.concat([df_LR_new, df_LR], join="inner", sort=True)
    df_LR_new.insert(1, 'Aname', "A2")
    df_LR_new.insert(5, 'algoname', "LR")
    df_LR_new = df_LR_new[['qid', 'Aname', 'pid', 'rank', 'score', 'algoname']]
    df_LR_new.to_csv("LR.txt", sep='\t', index=False, header=False)


if __name__ == '__main__':
    train_data = sample("train_data.tsv")
    passages_pro, queries_pro = processing(train_data)
    model_pass, model_query = gen_embedding(passages_pro, queries_pro)
    train_data = cal_embedding(train_data, passages_pro, queries_pro, model_pass, model_query)
    train_data.to_pickle("test2.pickle")
    train_data = pd.read_pickle("test2.pickle")
    x, y = split_data(train_data)
    model_lr = LogisticRegression(x, y, 0.01)
    model_lr.train()
    map_top100, mNDCG_top100 = perform_val("validation_data.tsv", model_lr)
    print("When processing top 100 scores: ", "Average precision is ", map_top100, "; NDCG is ", mNDCG_top100)
    # OUTPUT
    # When processing top 100 scores:  Average precision is  0.010386474132945543 ; NDCG is  0.030965383273730065

    test_data, y_prd, y_val = process_testfile("candidate_passages_top1000.tsv", model_lr)
    test_qry = pd.read_csv("test-queries.tsv", sep='\t', names=["qid", "queries"])
    output_LR(test_qry, test_data)
    lr_list = [0.01, 0.001, 0.0005, 0.0001]
    eval_lr(lr_list, x, y)
