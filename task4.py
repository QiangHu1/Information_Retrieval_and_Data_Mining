import task2
import task3
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import regularizers
import pandas as pd


# Prepare data and split them
def load_data(file_tra, file_val):
    train_data = pd.read_pickle(file_tra)
    val_data = pd.read_pickle(file_val)
    xTr, yTr = task2.split_data(train_data)
    x_val, y_val = task2.split_data(val_data)
    return val_data, xTr, yTr, x_val, y_val


# generate model
def gen_model(x_train, y_train, x_valid):
    model_NN = Sequential()
    model_NN.add(
        LSTM(20, input_shape=(200, 1), activation='tanh', kernel_regularizer=regularizers.l2(0.01), dropout=0.2,
             recurrent_dropout=0.2))
    model_NN.add(Dense(1))
    model_NN.compile(loss='mean_squared_error', optimizer='rmsprop')
    model_NN.fit(x_train, y_train, epochs=5, batch_size=500)
    y_pred = model_NN.predict(x_valid)
    return model_NN, y_pred


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
    y_prd = model.predict(x_val)
    data.insert(data.shape[1], "score", y_prd)
    return data, y_prd, y_val


# output data for top 100 passages for each query in test-queries.tsv
def output_NN(model):
    val_data, y_prd, y_val = process_testfile("candidate_passages_top1000.tsv", model)
    test_qry = pd.read_csv("test-queries.tsv", sep='\t', names=["qid", "queries"])
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
    df_LR_new.insert(5, 'algoname', "NN")
    df_LR_new = df_LR_new[['qid', 'Aname', 'pid', 'rank', 'score', 'algoname']]
    df_LR_new.to_csv("NN.txt", sep='\t', index=False, header=False)


if __name__ == '__main__':
    val_data, xTr, yTr, x_val, y_val = load_data("test2.pickle", "val_data.pickle")
    model_NN, y_prd = gen_model(xTr, yTr, x_val)
    map_top100, mNDCG_top100 = task3.eval_model(val_data, y_prd)
    print("When processing top 100 scores: ", "Average precision is ", map_top100, "; NDCG is ", mNDCG_top100)
    # OUTPUT
    # When processing top 100 scores:  Average precision is  0.011773336342715606 ; NDCG is  0.031244285554811124
    output_NN(model_NN)

