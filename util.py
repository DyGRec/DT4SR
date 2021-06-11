import sys
import copy
import random
import numpy as np
import multiprocessing
import time
import os
import heapq
from collections import defaultdict


import metrics
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import copy

Ks = [1, 2, 3, 4, 5, 10, 20, 40, 50, 60, 70, 80, 90,100]
cores = multiprocessing.cpu_count() // 2

def load_file_and_sort(filename):
    data = defaultdict(list)
    max_uind = 0
    max_iind = 0
    with open(filename, 'r') as f:
        for line in f:
            one_interaction = line.rstrip().split("\t")
            uind = int(one_interaction[0]) + 1
            iind = int(one_interaction[1]) + 1
            max_uind = max(max_uind, uind)
            max_iind = max(max_iind, iind)
            t = float(one_interaction[2])
            data[uind].append((iind, t))

    sorted_data = {}
    for u, i_list in data.items():
        sorted_interactions = sorted(i_list, key=lambda x:x[1])
        seq = [interaction[0] for interaction in sorted_interactions]
        sorted_data[u] = seq

    return sorted_data, max_uind, max_iind

def data_load(data_name):
    train_file = f"../../seq_itemsim/data/{data_name}/train.txt"
    valid_file = f"../../seq_itemsim/data/{data_name}/valid.txt"
    test_file = f"../../seq_itemsim/data/{data_name}/test.txt"
    user_train, usernum, itemnum = load_file_and_sort(train_file)
    user_valid, _, _ = load_file_and_sort(valid_file)
    user_test, _, _ = load_file_and_sort(test_file)

    num_valid = sum([len(i_list) for _, i_list in user_valid.items()])

    num_test = sum([len(i_list) for _, i_list in user_test.items()])
    print("num: ", num_valid, num_test)

    return [user_train, user_valid, user_test, usernum, itemnum]

def data_loadMoHRdata(data_name):
    dataset = np.load('./data/'+data_name+'Partitioned.npy', allow_pickle=True)

    [user_training, user_validation, user_testing, Item, usernum, itemnum] = dataset

    user_valid = defaultdict(list)
    user_test = defaultdict(list)
    user_train = defaultdict(list)
    for u, ituple in user_validation.items():
        if len(ituple) > 0:
            user_valid[u+1] = [ituple[1]+1]
    for u, ituple in user_testing.items():
        if len(ituple) > 0:
            user_test[u+1] = [ituple[1]+1]
    for u, ilist in user_training.items():
        user_train[u+1] = [i+1 for i in ilist]

    num_valid = sum([len(i_list) for _, i_list in user_valid.items()])

    num_test = sum([len(i_list) for _, i_list in user_test.items()])
    print("num: ", num_valid, num_test)

    return [user_train, user_valid, user_test, usernum, itemnum+1]

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))
    mrr = metrics.mrr(r)

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc, 'mrr': mrr}

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    pos_ranks = defaultdict(list)
    for ind in range(len(item_score)):
        pred_item = item_score[ind][0]
        if pred_item in user_pos_test:
            pos_ranks[pred_item].append(ind+1)

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc, pos_ranks


def ranklist_by_sorted(user_pos_test, item_score, Ks):

    K_max = max(Ks)
    K_max_item_score = heapq.nsmallest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc, pos_ranks = get_auc(item_score, user_pos_test)
    return r, auc, K_max_item_score, pos_ranks


def eval_one_interaction(x):
    result = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }
    test_user = x[0]
    test_item = x[1]
    score_dict = x[2]
    #score_dict = {item: item_score for item, item_score in scores}
    num_test_item_candidates = x[3]
    user_pos_test = [test_item]

    r, auc, K_max_pred_items, pos_ranks = ranklist_by_sorted(user_pos_test, score_dict, Ks)
    #if len(score_dict) < num_test_item_candidates:
    #    r = rank_corrected(np.array(r), len(score_dict), num_test_item_candidates)
    re = get_performance(user_pos_test, r, auc, Ks)
    result['precision'] += re['precision']
    result['recall'] += re['recall']
    result['ndcg'] += re['ndcg']
    result['hit_ratio'] += re['hit_ratio']
    result['auc'] += re['auc']
    result['mrr'] += re['mrr']

    return result, test_user, pos_ranks

def eval_one_setitems(x):
    result = {
            "recall": 0,
            "ndcg": 0
    }
    ranks = x[0]
    k_ind = x[1]
    test_num_items = x[2]
    freq_ind = x[3]

    result['recall'] = metrics.itemperf_recall(ranks, Ks[k_ind])
    result['ndcg'] = metrics.itemperf_ndcg(ranks, Ks[k_ind], test_num_items)

    return result, k_ind, freq_ind



def rank_corrected(r, m, n):
    pos_ranks = np.argwhere(r==1)[:,0]
    corrected_r = np.zeros_like(r)
    for each_sample_rank in list(pos_ranks):
        corrected_rank = int(np.floor(((n-1)*each_sample_rank)/m))
        if corrected_rank >= len(corrected_r) - 1:
            continue
        corrected_r[corrected_rank] = 1
    assert np.sum(corrected_r) <= 1
    return corrected_r


def evaluate(model, dataset, args, sess, testorvalid):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    item_freq = defaultdict(int)
    for u, ilist in train.items():
        for itemid in ilist:
            item_freq[itemid] += 1
    freq_quantiles = np.array([1, 3, 7, 20, 50])
    items_in_freqintervals = [[] for _ in range(len(freq_quantiles)+1)]
    for item, freq_i in item_freq.items():
        interval_ind = -1
        for quant_ind, quant_freq in enumerate(freq_quantiles):
            if freq_i <= quant_freq:
                interval_ind = quant_ind
                break
        if interval_ind == -1:
            interval_ind = len(items_in_freqintervals) - 1
        items_in_freqintervals[interval_ind].append(item)



    results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }

    short_seq_results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }
    num_short_seqs = 0

    long_seq_results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }
    num_long_seqs = 0

    short7_seq_results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }
    num_short7_seqs = 0

    short37_seq_results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }
    num_short37_seqs = 0

    medium3_seq_results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }
    num_medium3_seqs = 0

    medium7_seq_results = {
            "precision": np.zeros(len(Ks)),
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "hit_ratio": np.zeros(len(Ks)),
            "auc": 0.,
            "mrr": 0.,
    }
    num_medium7_seqs = 0

    if testorvalid == "test":
        eval_data = test
    else:
        eval_data = valid
    num_valid_interactions = 0
    pool = multiprocessing.Pool(cores)

    all_predictions_results = defaultdict(list)

    batch_u = []
    batch_u_seq = []
    batch_item_idx = []
    batch_test_item = []


    u_ind = 0
    eval_num_users = 0
    for u, i_list in eval_data.items():
        u_ind += 1
        if len(train[u]) < 1 or len(eval_data[u]) < 1:
            print("skipping ", u)
            continue
        eval_num_users += 1


        rated = set(train[u])
        if testorvalid == "test":
            valid_set = set(valid.get(u, []))
            rated = rated | valid_set

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if testorvalid == "test":
            if u in valid:
                for i in reversed(valid[u]):
                    if idx == -1: break
                    seq[idx] = i
                    idx -= 1
        for i in reversed(train[u]):
            if idx == -1: break
            seq[idx] = i
            idx -= 1
        item_idx = [i_list[0]]
        if args.evalnegsample == -1:
            item_idx += list(set([i for i in range(itemnum)]) - rated - set([i_list[0]]))
        else:
            item_candiates = list(set([i for i in range(itemnum)]) - rated - set([i_list[0]]))
            if args.evalnegsample >= len(item_candiates):
                item_idx += item_candiates
            else:
                item_idx += list(np.random.choice(item_candiates, size=args.evalnegsample, replace=False))

        #batch_u_seq.append(seq)
        #batch_u.append(u)
        #batch_test_item.append(i_list[0])

        predictions = model.predict(sess, [u], [seq], item_idx)
        item_scores_dict = {}
        for ind in range(predictions.shape[0]):
            item_scores_dict[item_idx[ind]] = predictions[ind]

        all_predictions_results[u] = [item_scores_dict, i_list[0]]

        #batch_u_seq = []
        #batch_item_idx = []
        #batch_u = []
        #batch_test_item = []


    assert len(batch_u) == 0
    assert eval_num_users == len(all_predictions_results)
    print('eval num users: ', eval_num_users)


    pred_scores_list = []
    test_item_list = []
    test_user_list = []
    test_num_test_candidates = []

    all_predictions_results_output = []

    for test_u, pred_scores_gt in all_predictions_results.items():
        test_user_idx, test_item_idx = test_u, pred_scores_gt[1]
        # unk_predictions = [item_score[1] for item_score in pred_scores]
        #
        # scaler = MinMaxScaler()
        # scale_pred = list(np.transpose(scaler.fit_transform(np.transpose(np.array([unk_predictions]))))[0])

        pred_scores_list.append(pred_scores_gt[0])
        test_item_list.append(test_item_idx)
        test_user_list.append(test_user_idx)
        test_num_test_candidates.append(itemnum)

    
    batch_data = zip(test_user_list, test_item_list, pred_scores_list, test_num_test_candidates)
    batch_result = pool.map(eval_one_interaction, batch_data)

    test_user_set = set()
    all_pos_items_ranks = defaultdict(list)

    for oneresult in batch_result:
        re, result_user, pos_items_ranks = oneresult
        results["precision"] += re["precision"]
        results["recall"] += re["recall"]
        results["ndcg"] += re["ndcg"]
        results["hit_ratio"] += re["hit_ratio"]
        results["auc"] += re["auc"]
        results["mrr"] += re["mrr"]
        test_user_set.add(result_user)

        for i, rank_list in pos_items_ranks.items():
                all_pos_items_ranks[i].extend(rank_list)

        if len(train[result_user]) <= 3:
            short_seq_results["precision"] += re["precision"]
            short_seq_results["recall"] += re["recall"]
            short_seq_results["ndcg"] += re["ndcg"]
            short_seq_results["hit_ratio"] += re["hit_ratio"]
            short_seq_results["auc"] += re["auc"]
            short_seq_results["mrr"] += re["mrr"]
            num_short_seqs += 1

        if len(train[result_user]) <= 7:
            short7_seq_results["precision"] += re["precision"]
            short7_seq_results["recall"] += re["recall"]
            short7_seq_results["ndcg"] += re["ndcg"]
            short7_seq_results["hit_ratio"] += re["hit_ratio"]
            short7_seq_results["auc"] += re["auc"]
            short7_seq_results["mrr"] += re["mrr"]
            num_short7_seqs += 1

        if len(train[result_user]) > 3 and len(train[result_user]) <= 7:
            short37_seq_results["precision"] += re["precision"]
            short37_seq_results["recall"] += re["recall"]
            short37_seq_results["ndcg"] += re["ndcg"]
            short37_seq_results["hit_ratio"] += re["hit_ratio"]
            short37_seq_results["auc"] += re["auc"]
            short37_seq_results["mrr"] += re["mrr"]
            num_short37_seqs += 1

        if len(train[result_user]) > 3 and len(train[result_user]) < 20:
            medium3_seq_results["precision"] += re["precision"]
            medium3_seq_results["recall"] += re["recall"]
            medium3_seq_results["ndcg"] += re["ndcg"]
            medium3_seq_results["hit_ratio"] += re["hit_ratio"]
            medium3_seq_results["auc"] += re["auc"]
            medium3_seq_results["mrr"] += re["mrr"]
            num_medium3_seqs += 1

        if len(train[result_user]) > 7 and len(train[result_user]) < 20:
            medium7_seq_results["precision"] += re["precision"]
            medium7_seq_results["recall"] += re["recall"]
            medium7_seq_results["ndcg"] += re["ndcg"]
            medium7_seq_results["hit_ratio"] += re["hit_ratio"]
            medium7_seq_results["auc"] += re["auc"]
            medium7_seq_results["mrr"] += re["mrr"]
            num_medium7_seqs += 1

        if len(train[result_user]) >= 20:
            long_seq_results["precision"] += re["precision"]
            long_seq_results["recall"] += re["recall"]
            long_seq_results["ndcg"] += re["ndcg"]
            long_seq_results["hit_ratio"] += re["hit_ratio"]
            long_seq_results["auc"] += re["auc"]
            long_seq_results["mrr"] += re["mrr"]
            num_long_seqs += 1

    results["precision"] /= len(test_user_set)
    results["recall"] /= len(test_user_set)
    results["ndcg"] /= len(test_user_set)
    results["hit_ratio"] /= len(test_user_set)
    results["auc"] /= len(test_user_set)
    results["mrr"] /= len(test_user_set)
    print(f"testing #of users: {len(test_user_set)}")
    assert eval_num_users == len(test_user_set)


    if num_short_seqs > 0:
        short_seq_results["precision"] /= num_short_seqs
        short_seq_results["recall"] /= num_short_seqs
        short_seq_results["ndcg"] /= num_short_seqs
        short_seq_results["hit_ratio"] /= num_short_seqs
        short_seq_results["auc"] /= num_short_seqs
        short_seq_results["mrr"] /= num_short_seqs
    print(f"testing #of short seq users with less than 3 training points: {num_short_seqs}")

    if num_short7_seqs > 0:
        short7_seq_results["precision"] /= num_short7_seqs
        short7_seq_results["recall"] /= num_short7_seqs
        short7_seq_results["ndcg"] /= num_short7_seqs
        short7_seq_results["hit_ratio"] /= num_short7_seqs
        short7_seq_results["auc"] /= num_short7_seqs
        short7_seq_results["mrr"] /= num_short7_seqs
    print(f"testing #of short seq users with less than 7 training points: {num_short7_seqs}")

    if num_short37_seqs > 0:
        short37_seq_results["precision"] /= num_short37_seqs
        short37_seq_results["recall"] /= num_short37_seqs
        short37_seq_results["ndcg"] /= num_short37_seqs
        short37_seq_results["hit_ratio"] /= num_short37_seqs
        short37_seq_results["auc"] /= num_short37_seqs
        short37_seq_results["mrr"] /= num_short37_seqs
    print(f"testing #of short seq users with 3 - 7 training points: {num_short37_seqs}")

    if num_medium3_seqs > 0:
        medium3_seq_results["precision"] /= num_medium3_seqs
        medium3_seq_results["recall"] /= num_medium3_seqs
        medium3_seq_results["ndcg"] /= num_medium3_seqs
        medium3_seq_results["hit_ratio"] /= num_medium3_seqs
        medium3_seq_results["auc"] /= num_medium3_seqs
        medium3_seq_results["mrr"] /= num_medium3_seqs
    print(f"testing #of short seq users with medium3 training points: {num_medium3_seqs}")

    if num_medium7_seqs > 0:
        medium7_seq_results["precision"] /= num_medium7_seqs
        medium7_seq_results["recall"] /= num_medium7_seqs
        medium7_seq_results["ndcg"] /= num_medium7_seqs
        medium7_seq_results["hit_ratio"] /= num_medium7_seqs
        medium7_seq_results["auc"] /= num_medium7_seqs
        medium7_seq_results["mrr"] /= num_medium7_seqs
    print(f"testing #of short seq users with medium7 training points: {num_medium7_seqs}")

    if num_long_seqs > 0:
        long_seq_results["precision"] /= num_long_seqs
        long_seq_results["recall"] /= num_long_seqs
        long_seq_results["ndcg"] /= num_long_seqs
        long_seq_results["hit_ratio"] /= num_long_seqs
        long_seq_results["auc"] /= num_long_seqs
        long_seq_results["mrr"] /= num_long_seqs

    print(f"testing #of short seq users with >= 20 training points: {num_long_seqs}")




    test_num_items_in_intervals = []
    interval_results = [{'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))} for _ in range(len(items_in_freqintervals))]

    all_freq_all_ranks = []
    all_ks = []
    all_numtestitems = []
    all_freq_ind = []
    for freq_ind, item_list in enumerate(items_in_freqintervals):
        num_item_pos_interactions = 0
        all_ranks = []
        interval_items = []
        for item in item_list:
            pos_ranks_oneitem = all_pos_items_ranks.get(item, [])
            if len(pos_ranks_oneitem) > 0:
                interval_items.append(item)
            all_ranks.extend(pos_ranks_oneitem)
        for k_ind in range(len(Ks)):
            all_ks.append(k_ind)
            all_freq_all_ranks.append(all_ranks)
            all_numtestitems.append(args.evalnegsample+1)
            all_freq_ind.append(freq_ind)
        test_num_items_in_intervals.append(interval_items)

    item_eval_freq_data = zip(all_freq_all_ranks, all_ks, all_numtestitems, all_freq_ind)
    batch_item_result = pool.map(eval_one_setitems, item_eval_freq_data)

    for oneresult in batch_item_result:
        result_dict = oneresult[0]
        k_ind = oneresult[1]
        freq_ind = oneresult[2]
        interval_results[freq_ind]['recall'][k_ind] = result_dict['recall']
        interval_results[freq_ind]['ndcg'][k_ind] = result_dict['ndcg']



    item_freq = freq_quantiles
    for i in range(len(item_freq)+1):
        if i == 0:
            print('For items in freq between 0 - %d with %d items: ' % (item_freq[i], len(test_num_items_in_intervals[i])))
        elif i == len(item_freq):
            print('For items in freq between %d - max with %d items: ' % (item_freq[i-1], len(test_num_items_in_intervals[i])))
        else:
            print('For items in freq between %d - %d with %d items: ' % (item_freq[i-1], item_freq[i], len(test_num_items_in_intervals[i])))
        for k_ind in range(len(Ks)):
            k = Ks[k_ind]
            print('Recall@%d:%.6f, NDCG@%d:%.6f'%(k, interval_results[i]['recall'][k_ind], k, interval_results[i]['ndcg'][k_ind]))


    return results, short_seq_results, short7_seq_results, short37_seq_results, medium3_seq_results, medium7_seq_results, long_seq_results, all_predictions_results_output

