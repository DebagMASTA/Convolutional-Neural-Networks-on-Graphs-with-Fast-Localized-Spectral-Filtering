import csv
import sys, os, time
import argparse, yaml, shutil, math
import scipy.sparse.linalg
sys.path.insert(0, '..')

import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

filename = "\\kakuninn_accuracy.csv"

matrix_filename = "\\kakuninn_matrix.csv"

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', '-g', type=int, default=0,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                    help='base directory path of program files')
# parser.add_argument('--config_path', type=str, default='configs/base.yml',
#                     help='path to config file')
parser.add_argument('--out', '-o', default='adj',
                    help='Directory to output the result')

# parser.add_argument('--model', '-m',
#                     default="D:\\PycharmProjects\\3DCNN_chainer\\3DCNN\\results\\0104_group2\\training\\CNN3D_7800.npz",
#                     help='Load model data(snapshot)')

parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                    help='Root directory path of input image')
parser.add_argument('--test_list', default='configs/validation_list.txt',
                    help='Path to test image list file')
args = parser.parse_args()

if not os.path.exists(os.path.join(args.base, args.out)):
    os.makedirs(os.path.join(args.base, args.out))

################################
#org_train:元データ
#value_train：データの値
#dat_train_3D：CNNへの入力
#ans_train：正解ラベル
##################################

def make_Laplacian(kinbou=26):
    len = 15964
    x_min = 34
    y_min = 25
    x_size = 60
    y_size = 77
    z_size = 59
    size = x_size * y_size * z_size

    # folder='C:\\Users\\yambe\\Documents\\Experiment\\all\\'
    path = os.path.dirname(os.path.abspath(__file__)) + '\\Laplacian\\0114\\'

    """学習データ読み込み"""
    # num_train=220                              #トレーニングデータ数
    # dat_train=np.empty(len)                     # xはデータ、ｙは正解ラベル
    # ans_train=np.empty(4)
    # folder='C:\\Users\\yambe\\Documents\\Experiment\\all_WSFM\\'              #データが入ってるフォルダ
    #
    #
    # f=open('C:\\Users\\yambe\\Documents\\Experiment\\3DCNN\\CV\\3fold\\group3\\train_name.txt')              #症例テキストのパス
    # lines_train=f.readlines()                 #テキストファイル１行ずつ読み取り
    # f.close()
    #
    #
    # print('now loading training dataset...')
    # for case in lines_train:                      #caseは各症例のデータ名、各症例のループ
    #     case=case.replace('\n','')

    with open('C:\\Users\\yambe\\Documents\\Study\\Experiment\\all_data\\A-1.dat', 'r') as org:
        """学習データについて"""
        org = list(csv.reader(org))  # データの末尾に無駄な１行あり
        org = np.reshape(org, (len + 1, 4))  # 15965*4のnumpy.arrayに変形
        org = np.delete(org, len, 0)  # いらない末尾を削除
        value = [n[3] for n in org]  # ４列目を、１次元配列value_trainとする
        value = np.array(value, dtype=float)  # float型にする

    """3次元化"""

    loc = np.delete(org, 3, 1)  # locは15964*3の濃度値がある座標を示す
    loc = np.array(loc, dtype=int)
    dat_3D = np.full((size), 0.0)  # [学習データ数][60*77*59]
    ind_3D = np.full((size), 0)
    adj = np.full((len, len), 0)
    """1次元配列の、濃度値を入れるインデックスの取得"""

    for n in range(len):
        ind = (loc[n][0] - x_min) + (loc[n][1] - y_min) * x_size + loc[n][2] * x_size * y_size
        dat_3D[ind] = value[n]
        ind_3D[ind] = n

    dat_3D = np.reshape(dat_3D.astype('float32'), (z_size, y_size, x_size))
    ind_3D = np.reshape(ind_3D.astype('int'), (z_size, y_size, x_size))

    # print('dat_3D\n',dat_3D[30][0])
    # print('dat_3D.shape\n',dat_3D.shape)
    # print('ind_3D\n',ind_3D[0][30])
    # print('ind_3D.shape\n',ind_3D.shape)
    if kinbou == 26:
        for z in range(z_size):
            for y in range(y_size):
                for x in range(x_size):
                    if dat_3D[z][y][x] > 0:
                        if x - 1 >= 0:  # 1
                            if dat_3D[z][y][x - 1] > 0:
                                adj[ind_3D[z][y][x]][ind_3D[z][y][x - 1]] = 1
                                adj[ind_3D[z][y][x - 1]][ind_3D[z][y][x]] = 1

                            if y - 1 >= 0:  # 2
                                if dat_3D[z][y - 1][x - 1] > 0:
                                    adj[ind_3D[z][y][x]][ind_3D[z][y - 1][x - 1]] = 1
                                    adj[ind_3D[z][y - 1][x - 1]][ind_3D[z][y][x]] = 1
                                if z - 1 >= 0:  # 3
                                    if dat_3D[z - 1][y - 1][x - 1] > 0:
                                        adj[ind_3D[z][y][x]][ind_3D[z - 1][y - 1][x - 1]] = 1
                                        adj[ind_3D[z - 1][y - 1][x - 1]][ind_3D[z][y][x]] = 1
                                if z + 1 < z_size:  # 4
                                    if dat_3D[z + 1][y - 1][x - 1] > 0:
                                        adj[ind_3D[z][y][x]][ind_3D[z + 1][y - 1][x - 1]] = 1
                                        adj[ind_3D[z + 1][y - 1][x - 1]][ind_3D[z][y][x]] = 1

                            if y + 1 < y_size:  # 5
                                if dat_3D[z][y + 1][x - 1] > 0:
                                    adj[ind_3D[z][y][x]][ind_3D[z][y + 1][x - 1]] = 1
                                    adj[ind_3D[z][y + 1][x - 1]][ind_3D[z][y][x]] = 1
                                if z - 1 >= 0:  # 6
                                    if dat_3D[z - 1][y + 1][x - 1] > 0:
                                        adj[ind_3D[z][y][x]][ind_3D[z - 1][y + 1][x - 1]] = 1
                                        adj[ind_3D[z - 1][y + 1][x - 1]][ind_3D[z][y][x]] = 1
                                if z + 1 < z_size:  # 7
                                    if dat_3D[z + 1][y + 1][x - 1] > 0:
                                        adj[ind_3D[z][y][x]][ind_3D[z + 1][y + 1][x - 1]] = 1
                                        adj[ind_3D[z + 1][y + 1][x - 1]][ind_3D[z][y][x]] = 1

                            if z - 1 >= 0:  # 8
                                if dat_3D[z - 1][y][x - 1] > 0:
                                    adj[ind_3D[z][y][x]][ind_3D[z - 1][y][x - 1]] = 1
                                    adj[ind_3D[z - 1][y][x - 1]][ind_3D[z][y][x]] = 1
                            if z + 1 < z_size:  # 9
                                if dat_3D[z + 1][y][x - 1] > 0:
                                    adj[ind_3D[z][y][x]][ind_3D[z + 1][y][x - 1]] = 1
                                    adj[ind_3D[z + 1][y][x - 1]][ind_3D[z][y][x]] = 1

                        if x + 1 < x_size:  # 10
                            if dat_3D[z][y][x + 1] > 0:
                                adj[ind_3D[z][y][x]][ind_3D[z][y][x + 1]] = 1
                                adj[ind_3D[z][y][x + 1]][ind_3D[z][y][x]] = 1

                            if y - 1 >= 0:  # 11
                                if dat_3D[z][y - 1][x + 1] > 0:
                                    adj[ind_3D[z][y][x]][ind_3D[z][y - 1][x + 1]] = 1
                                    adj[ind_3D[z][y - 1][x + 1]][ind_3D[z][y][x]] = 1
                                if z - 1 >= 0:  # 12
                                    if dat_3D[z - 1][y - 1][x + 1] > 0:
                                        adj[ind_3D[z][y][x]][ind_3D[z - 1][y - 1][x + 1]] = 1
                                        adj[ind_3D[z - 1][y - 1][x + 1]][ind_3D[z][y][x]] = 1
                                if z + 1 < z_size:  # 13
                                    if dat_3D[z + 1][y - 1][x + 1] > 0:
                                        adj[ind_3D[z][y][x]][ind_3D[z + 1][y - 1][x + 1]] = 1
                                        adj[ind_3D[z + 1][y - 1][x + 1]][ind_3D[z][y][x]] = 1

                            if y + 1 < y_size:  # 14
                                if dat_3D[z][y + 1][x + 1] > 0:
                                    adj[ind_3D[z][y][x]][ind_3D[z][y + 1][x + 1]] = 1
                                    adj[ind_3D[z][y + 1][x + 1]][ind_3D[z][y][x]] = 1
                                if z - 1 >= 0:  # 15
                                    if dat_3D[z - 1][y + 1][x + 1] > 0:
                                        adj[ind_3D[z][y][x]][ind_3D[z - 1][y + 1][x + 1]] = 1
                                        adj[ind_3D[z - 1][y + 1][x + 1]][ind_3D[z][y][x]] = 1
                                if z + 1 < z_size:  # 16
                                    if dat_3D[z + 1][y + 1][x + 1] > 0:
                                        adj[ind_3D[z][y][x]][ind_3D[z + 1][y + 1][x + 1]] = 1
                                        adj[ind_3D[z + 1][y + 1][x + 1]][ind_3D[z][y][x]] = 1

                            if z - 1 >= 0:  # 17
                                if dat_3D[z - 1][y][x + 1] > 0:
                                    adj[ind_3D[z][y][x]][ind_3D[z - 1][y][x + 1]] = 1
                                    adj[ind_3D[z - 1][y][x + 1]][ind_3D[z][y][x]] = 1
                            if z + 1 < z_size:  # 18
                                if dat_3D[z + 1][y][x + 1] > 0:
                                    adj[ind_3D[z][y][x]][ind_3D[z + 1][y][x + 1]] = 1
                                    adj[ind_3D[z + 1][y][x + 1]][ind_3D[z][y][x]] = 1

                        if y - 1 >= 0:  # 19
                            if dat_3D[z][y - 1][x] > 0:
                                adj[ind_3D[z][y][x]][ind_3D[z][y - 1][x]] = 1
                                adj[ind_3D[z][y - 1][x]][ind_3D[z][y][x]] = 1
                            if z - 1 >= 0:  # 20
                                if dat_3D[z - 1][y - 1][x] > 0:
                                    adj[ind_3D[z][y][x]][ind_3D[z - 1][y - 1][x]] = 1
                                    adj[ind_3D[z - 1][y - 1][x]][ind_3D[z][y][x]] = 1
                            if z + 1 < z_size:  # 21
                                if dat_3D[z + 1][y - 1][x] > 0:
                                    adj[ind_3D[z][y][x]][ind_3D[z + 1][y - 1][x]] = 1
                                    adj[ind_3D[z + 1][y - 1][x]][ind_3D[z][y][x]] = 1
                        if y + 1 < y_size:  # 22
                            if dat_3D[z][y + 1][x] > 0:
                                adj[ind_3D[z][y][x]][ind_3D[z][y + 1][x]] = 1
                                adj[ind_3D[z][y + 1][x]][ind_3D[z][y][x]] = 1
                            if z - 1 >= 0:  # 23
                                if dat_3D[z - 1][y + 1][x] > 0:
                                    adj[ind_3D[z][y][x]][ind_3D[z - 1][y + 1][x]] = 1
                                    adj[ind_3D[z - 1][y + 1][x]][ind_3D[z][y][x]] = 1
                            if z + 1 < z_size:  # 24
                                if dat_3D[z + 1][y + 1][x] > 0:
                                    adj[ind_3D[z][y][x]][ind_3D[z + 1][y + 1][x]] = 1
                                    adj[ind_3D[z + 1][y + 1][x]][ind_3D[z][y][x]] = 1
                        if z - 1 >= 0:  # 25
                            if dat_3D[z - 1][y][x] > 0:
                                adj[ind_3D[z][y][x]][ind_3D[z - 1][y][x]] = 1
                                adj[ind_3D[z - 1][y][x]][ind_3D[z][y][x]] = 1
                        if z + 1 < z_size:  # 26
                            if dat_3D[z + 1][y][x] > 0:
                                adj[ind_3D[z][y][x]][ind_3D[z + 1][y][x]] = 1
                                adj[ind_3D[z + 1][y][x]][ind_3D[z][y][x]] = 1

        # df = pd.DataFrame(adj)
        # df.to_csv(os.path.join(args.base, args.out) + matrix_filename)

    A = scipy.sparse.csr_matrix(adj.astype(np.float32))
    # graphs, perm = coarsening.coarsen(A, levels=4, self_connections=False)
    # print('perm', perm)
    # L = [graph.laplacian(A, normalized=True) for A in graphs]
    # graph.plot_spectrum(L)
    # scipy.sparse.save_npz(path+'laplacian.npz',L,compressed = True)
    return(A)



