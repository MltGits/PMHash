import numpy as np
import pickle
import sys

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def calc_map(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map


def calc_topMap(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    print(num_query)
    topkmap = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(np.int32)
        # print(tsum)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkmap


def pr_curve(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    dim = np.shape(rB)
    bit = dim[1]
    all_ = dim[0]
    precision = np.zeros(bit + 1)
    recall = np.zeros(bit + 1)
    num_query = queryL.shape[0]
    num_database = retrievalL.shape[0]
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        all_sum = np.sum(gnd).astype(np.float32)
        # print(all_sum)
        if all_sum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        # print(hamm.shape)
        ind = np.argsort(hamm)
        # print(ind.shape)
        gnd = gnd[ind]
        hamm = hamm[ind]
        hamm = hamm.tolist()
        # print(len(hamm), num_database - 1)
        max_ = hamm[num_database - 1]
        max_ = int(max_)
        t = 0
        for i in range(1, max_):
            if i in hamm:
                idd = hamm.index(i)
                if idd != 0:
                    sum1 = np.sum(gnd[:idd])
                    precision[t] += sum1 / idd
                    recall[t] += sum1 / all_sum
                else:
                    precision[t] += 0
                    recall[t] += 0
                t += 1
        # precision[t] += all_sum / num_database
        # recall[t] += 1
        for i in range(t,  bit + 1):
            precision[i] += all_sum / num_database
            recall[i] += 1
    true_recall = recall / num_query
    precision = precision / num_query
    print(true_recall)
    print(precision)
    return true_recall, precision

def CalcNDCG_N(N, qB, rB, queryL, retrievalL):
    num_q = qB.shape[0]
    a_NDCG = 0.0
    NDCG = 0.0
    for i in range(num_q):
        DCG = 0.0
        max_DCG = 0.0
        sim = (np.dot(queryL[i, :], retrievalL.transpose())).astype(np.float32)
        # qL = np.sum(queryL[i,:]).astype(np.float32)
        # rL = np.sum(retrievalL, axis=1).astype(np.float32)
        # L = np.power(qL * rL, 0.5).astype(np.float32)
        # sim = sim / L
        hamm = calc_hammingDist(qB[i, :], rB)
        ind = np.argsort(hamm)
        sim_sort = np.argsort(sim)
        for k in range(N):
            gain = 2 ** sim[ind[k]] - 1
            gain_max = 2 ** sim[sim_sort[- k - 1]] - 1
            log = np.log2(k + 2)
            DCG += gain / log
            max_DCG += gain_max / log
        NDCG += DCG / max_DCG
    a_NDCG = NDCG / num_q
    return a_NDCG

def precision_topn(qB, rB, queryL, retrievalL, topk=1000):
    n = topk // 100
    precision = np.zeros(n)
    num_query = queryL.shape[0]
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        for i in range(1, n + 1):
            a = gnd[:i * 100]
            #print(len(a))
            precision[i - 1] += float(a.sum()) / (i * 100.)
    a_precision = precision / num_query
    return a_precision

def precision_with_radius(qB, rB, queryL, retrievalL, radius=2):
    num_correct = 0.0
    num_total = 0.0
    precisions = 0.0
    num_query = queryL.shape[0]
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)

        for i in ind:
            if hamm[i] <= radius:
                num_total += 1
                if gnd[i] == 1.0:
                    num_correct += 1
        if num_total == 0:
            continue
        precisions += num_correct / num_total
    return precisions / num_query


def visualization_retrieval(qB, rB, queryL, retrievalL, topk=10, filenames=[],image_path=""):
    images_path = []
    with open(image_path,mode="w") as f:
        num_query = queryL.shape[0]
        for iter in range(num_query):
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
            hamm = calc_hammingDist(qB[iter, :], rB)
            ind = np.argsort(hamm)[:topk]
            filename = np.array(filenames)[ind.tolist()].tolist()
            images = " ".join(filename)
            f.write(images)
            f.write(" " +str(gnd[ind].sum()))
            f.write("\n")
            images_path.append(filename)

    return images_path

def visualization_code(test_hash):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    test = np.zeros((1000, 32))
    test[0:100] = test_hash[0:100, :]
    test[100:200] = test_hash[1000:1100, :]
    test[200:300] = test_hash[2000:2100, :]
    test[300:400] = test_hash[3000:3100, :]
    test[400:500] = test_hash[4000:4100, :]
    test[500:600] = test_hash[5000:5100, :]
    test[600:700] = test_hash[6000:6100, :]
    test[700:800] = test_hash[7000:7100, :]
    test[800:900] = test_hash[8000:8100, :]
    test[900:1000] = test_hash[9000:9100, :]

    labels = np.array(range(1000))
    labels[0:100] = 0
    labels[100:200] = 1
    labels[200:300] = 2
    labels[300:400] = 3
    labels[400:500] = 4
    labels[500:600] = 5
    labels[600:700] = 6
    labels[700:800] = 7
    labels[800:900] = 8
    labels[900:1000] = 9

    # 进行可视化
    # 对哈希码进行分类
    tsne = TSNE(n_components=2, random_state=42)
    hash_codes_tsne = tsne.fit_transform(test)
    #
    # # 为每个聚类分配独特的颜色
    colors = ListedColormap(['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'pink'])

    # 绘制可视化结果
    plt.scatter(hash_codes_tsne[:, 0], hash_codes_tsne[:, 1], c=labels, cmap=colors)
    plt.title("PMLHash")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
    plt.show()
















































































































































































































































































