import pandas as pd
import task2
import numpy as np
import xgboost as xgb


# load data and split them
def load_data(file_tra, file_val):
    train_data = pd.read_pickle(file_tra)
    val_data = pd.read_pickle(file_val)
    xTr, yTr = task2.split_data(train_data)
    x_val, y_val = task2.split_data(val_data)
    return val_data, xTr, yTr, x_val, y_val


# generate models with different pairs of parameters
def gen_model(xTr, yTr, x_val, y_val):
    lr_list = [0.01, 0.005, 0.001, 0.0005]
    dep_list = np.arange(6) + 3
    para_list = np.zeros((24, 2))
    count = 0
    model_list, y_pred_list = [], []
    for i in lr_list:
        for j in dep_list:
            para_list[count, 0] = i
            para_list[count, 1] = j
            count += 1
    for i in range(len(para_list)):
        params = {
            'max_depth': int(para_list[i, 1]),
            'eta': para_list[i, 0],
            'objective': 'rank:pairwise',
            'eval_metric': ["ndcg", "map"]
        }
        d_tra = xgb.DMatrix(xTr, label=yTr)
        d_val = xgb.DMatrix(x_val, label=y_val)
        model_LM = xgb.train(params, d_tra)
        # predict scores
        y_pred = model_LM.predict(d_val)
        y_pred_list.append(y_pred)
        model_list.append(model_LM)
    return model_list, para_list, y_pred_list


# evaluate the model using validation dataset
def eval_model(val_data_i, y_prd):
    val_data_i.insert(val_data_i.shape[1], "score", y_prd)
    val_data_i = val_data_i[['qid', 'pid', "relevancy", "score"]]
    qid_val = val_data_i['qid'].drop_duplicates().tolist()
    n = len(qid_val)
    ap_top100, ndcg_top100 = np.zeros(n), np.zeros(n)
    for i in range(len(qid_val)):
        qid = qid_val[i]
        sub_df = val_data_i[val_data_i.qid == qid].sort_values('score', ascending=False)[0:100]
        ap_top100[i], ndcg_top100[i] = task2.cal_ap_ndcg(sub_df)
    return np.mean(ap_top100), np.mean(ndcg_top100)


# compare the performance of each model to get the best model
def best_model(model_list, para_list, y_pred_list, val_data):
    map_list = np.zeros(24)
    mNDCG_list = np.zeros(24)
    for i in range(len(para_list)):
        val_data_i = val_data.copy(deep=True)
        map_top100, mNDCG_top100 = eval_model(val_data_i, y_pred_list[i])
        map_list[i] = map_top100
        mNDCG_list[i] = mNDCG_top100
    index = np.where(mNDCG_list == np.max(mNDCG_list))[0][0]
    return model_list[index], para_list[index], y_pred_list[index], map_list[index], mNDCG_list[index]


# process the testfile generated from candidate_passages_top1000.tsv
def process_testfile(filename, model):
    # val_data = pd.read_csv(filename, sep='\t', names=['qid', 'pid', 'queries', 'passage'])
    # passages_val, queries_val = task2.processing(val_data)
    # model_pass_val, model_query_val = task2.gen_embedding(passages_val, queries_val)
    # val_data = task2.cal_embedding(val_data, passages_val, queries_val, model_pass_val, model_query_val)
    # val_data.to_pickle("test_data.pickle")
    data = pd.read_pickle("test_data.pickle")
    data.insert(data.shape[1], "relevancy", 0)
    x_val, y_val = task2.split_data(data)
    # predict scores
    dtest = xgb.DMatrix(x_val)
    y_prd = model.predict(dtest)
    data.insert(data.shape[1], "score", y_prd)
    return data, y_prd, y_val


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
    df_LR_new.insert(5, 'algoname', "LM")
    df_LR_new = df_LR_new[['qid', 'Aname', 'pid', 'rank', 'score', 'algoname']]
    df_LR_new.to_csv("LM.txt", sep='\t', index=False, header=False)


def output_LM(xTr, yTr):
    params = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'rank:pairwise',
        'eval_metric': ["ndcg", "map"]
    }
    d_tra = xgb.DMatrix(xTr, label=yTr)
    model_LM = xgb.train(params, d_tra)
    test_data, y_prd, y_val = process_testfile("candidate_passages_top1000.tsv", model_LM)
    test_qry = pd.read_csv("test-queries.tsv", sep='\t', names=["qid", "queries"])
    output_LR(test_qry, test_data)


if __name__ == '__main__':
    val_data, xTr, yTr, x_val, y_val = load_data("test2.pickle", "val_data.pickle")
    model_list, para_list, y_pred_list = gen_model(xTr, yTr, x_val, y_val)
    best_mod, best_para, y_pred_bs, map_bs, mndcg_bs = best_model(model_list, para_list, y_pred_list, val_data)
    print("When processing top 100 scores: ", "Best lr is ", best_para[0], "; Best depth is ", best_para[1])
    print("When processing top 100 scores: ", "Average precision is ", map_bs, "; NDCG is ", mndcg_bs)
    # OUTPUT
    # When processing top 100 scores:  Best lr is  0.01 ; Best depth is  3.0
    # When processing top 100 scores:  Average precision is  0.013676473259280933 ; NDCG is  0.03613177677975519
    output_LM(xTr, yTr)
