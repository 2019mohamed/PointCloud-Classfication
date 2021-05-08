# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:25:55 2021

@author: M
"""

import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, Amplitude
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
import itertools
import networkx as nx
import openml
from gtda.plotting import plot_point_cloud
from sklearn.neural_network import MLPClassifier
from karateclub import FeatherGraph
from tqdm import tqdm
from mapper import Mappe
shapenetpart_seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
shapenetpart_seg_start_index = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class Dataset(data.Dataset):
    def __init__(self, root, dataset_name='modelnet40', class_choice=None,
            num_points=2048, split='train', load_name=True, load_file=True,
            segmentation=False, random_rotate=False, random_jitter=False, 
            random_translate=False):

        assert dataset_name.lower() in ['shapenetcorev2', 'shapenetpart', 
            'modelnet10', 'modelnet40', 'shapenetpartpart']
        assert num_points <= 2048        

        if dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']
        else:
            assert split.lower() in ['train', 'test', 'all']

        if dataset_name not in ['shapenetcorev2', 'shapenetpart'] and segmentation == True:
            raise AssertionError

        self.root = os.path.join(root, dataset_name + '_' + '*hdf5_2048')
        self.dataset_name = dataset_name
        self.class_choice = class_choice
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_file = load_file
        self.segmentation = segmentation
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        
        self.path_h5py_all = []
        self.path_name_all = []
        self.path_file_all = []

        if self.split in ['train','trainval','all']:   
            self.get_path('train')
        if self.dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            if self.split in ['val','trainval','all']: 
                self.get_path('val')
        if self.split in ['test', 'all']:   
            self.get_path('test')

        self.path_h5py_all.sort()
        data, label, seg = self.load_h5py(self.path_h5py_all)

        if self.load_name or self.class_choice != None:
            self.path_name_all.sort()
            self.name = self.load_json(self.path_name_all)    # load label name

        if self.load_file:
            self.path_file_all.sort()
            self.file = self.load_json(self.path_file_all)    # load file name
        
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0) 
        if self.segmentation:
            self.seg = np.concatenate(seg, axis=0) 

        if self.class_choice != None:
            indices = (self.name == class_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            if self.segmentation:
                self.seg = self.seg[indices]
                self.seg_num_all = shapenetpart_seg_num[id_choice]
                self.seg_start_index = shapenetpart_seg_start_index[id_choice]
            if self.load_file:
                self.file = self.file[indices]
        elif self.segmentation:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def get_path(self, type):
        path_h5py = os.path.join(self.root, '*%s*.h5'%type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, '%s*_id2name.json'%type)
            self.path_name_all += glob(path_json)
        if self.load_file:
            path_json = os.path.join(self.root, '%s*_id2file.json'%type)
            self.path_file_all += glob(path_json)
        return 

    def load_h5py(self, path):
        all_data = []
        all_label = []
        all_seg = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            if self.segmentation:
                seg = f['seg'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            if self.segmentation:
                all_seg.append(seg)
        return all_data, all_label, all_seg

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j =  open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        label = self.label[item]
        if self.load_name:
            name = self.name[item]  # get label name
        if self.load_file:
            file = self.file[item]  # get file name

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)
        
        if self.segmentation:
            seg = self.seg[item]
            seg = torch.from_numpy(seg)
            return point_set, label, seg, name, file
        else:
            return point_set, label, name, file

    def __len__(self):
        return self.data.shape[0]

def delaunay_triangulate(P: np.ndarray):
    """
    Perform delaunay triangulation on point set P.
    :param P: point set
    :return: adjacency matrix A
    """
    n = P.shape[0]
    if n < 3:
        A = fully_connect(P)
    else:
        try:
            d = Delaunay(P)
            #assert d.coplanar.size == 0, 'Delaunay triangulation omits points.'
            A = np.zeros((n, n))
            for simplex in d.simplices:
                for pair in itertools.permutations(simplex, 2):
                    A[pair] = 1
        except QhullError as err:
            print('Delaunay triangulation error detected. Return fully-connected graph.')
            print('Traceback:')
            print(err)
            A = fully_connect(P)
    return A


def fully_connect(P: np.ndarray, thre=None):
    """
    Fully connect a graph.
    :param P: point set
    :param thre: edges that are longer than this threshold will be removed
    :return: adjacency matrix A
    """
    n = P.shape[0]
    A = np.ones((n, n)) - np.eye(n)
    if thre is not None:
        xyz = P[:, :3]
        dist = -2 * xyz @ xyz.T
        dist += np.sum(xyz ** 2, axis=-1)[:, None]
        dist += np.sum(xyz ** 2, axis=-1)[None, :]
        PP_dist_flag = dist > (thre**2)
        # P_rep = np.expand_dims(P[:, :3], axis=1).repeat(n, axis=1)
        # PP_dist_flag = np.sqrt(np.sum(np.square(P_rep - P[None,:, :3]), axis=2)) > thre
        A[PP_dist_flag] = 0
        # for i in range(n):
        #     for j in range(i):
        #         if np.linalg.norm(P[i] - P[j]) > thre:
        #             A[i, j] = 0
        #             A[j, i] = 0
    return A

if __name__ == '__main__':
    
    root = 'C:\\Users\\M'

    # choose dataset name from 'shapenetcorev2', 'shapenetpart', 'modelnet40' and 'modelnet10'
    dataset_name = 'modelnet10'

    # choose split type from 'train', 'test', 'all', 'trainval' and 'val'
    # only shapenetcorev2 and shapenetpart dataset support 'trainval' and 'val'
    #split = 'train'
    train = Dataset(root=root, dataset_name=dataset_name, num_points=2048//2, split='train')
    test = Dataset(root=root, dataset_name=dataset_name, num_points=2048//2, split='test')
    
    N_train , N_test = train.__len__() , test.__len__()
    print('train ',N_train,' test ',N_test)
    '''
    item = np.random.randint(0,3990)
    print(item)
    ps, lb, n, f = d[item]
    print(lb)
    ps = ps.detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(ps[:,0] , ps[:,1] , ps[: , 2])
    plt.show()
    '''
    train_data , test_data , train_label , test_label = list() , list() , list(), list()
    def emb (g):
        
        emd = []
        embedder = FeatherGraph()
        g = nx.Graph(g)
        if nx.is_connected(g):
            relabel = { n : i for i,n in enumerate(list(g))}
            g = nx.relabel_nodes(g , relabel)
            embedder.fit([g])
            emd = embedder.get_embedding()[0]
        else:
            CC = nx.connected_components(g)
            embCC = []
            for c in list(CC):
                relabel = { n : i for i,n in enumerate(c)}
                sub = nx.subgraph(g,nbunch = c)
                sub = nx.relabel_nodes(sub , relabel)                
                embedder.fit([sub])
                embCC.append(embedder.get_embedding()[0])
                
            embCC = np.array(embCC)
            embCC = np.concatenate(tuple(embCC) , axis = 0)
            embCC = np.random.choice(embCC, size=500, replace=False)
            emd = embCC
        return emd
    
    for i in tqdm(range(N_train)):
        data = train[i][0].detach().numpy()
        #A = delaunay_triangulate(train[i][0].detach().numpy())
        #g = nx.from_numpy_matrix(A)
        g = Mappe(data , 20)
        train_data.append(emb(g))
        train_label.append(train[i][1].detach().numpy()[0])
            
    for i in tqdm(range(N_test)):
        data = test[i][0].detach().numpy()
        #A = delaunay_triangulate(test[i][0].detach().numpy())
        #g = nx.from_numpy_matrix(A)
        g = Mappe(data , 20)
        test_data.append(emb(g))
        test_label.append(test[i][1].detach().numpy()[0])
        
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    print(test_data.shape,' ',test_label.shape)
        
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(train_data, train_label)
    print('Fitting...')
    print('Our ACC of RTrees is:...')   
    

    print(model.score(test_data, test_label))
    
    clf = MLPClassifier(random_state=42, max_iter=3000)
    clf.fit(train_data, train_label)
    print('Fitting...')
    print('Our ACC of MLP is:...')
    print(clf.score(test_data, test_label))        
        
        
    '''   
    print('Data finished...!')
    VR = VietorisRipsPersistence(homology_dimensions = (0,1,2,3))
    
    diagrams_train = VR.fit_transform(train_data)
    print('diagrams_train finished ...')
    diagrams_test = VR.fit_transform(test_data)
    print('diagrams_test finished...')

    PE = PersistenceEntropy()
    NP = Amplitude()
    
    F_train = PE.fit_transform(diagrams_train)
    N_train = NP.fit_transform(diagrams_train)
    F_train = np.concatenate((F_train,N_train) , axis = 1)
    
    print('F_train finished...')
    F_test = PE.fit_transform(diagrams_test)
    N_test = NP.fit_transform(diagrams_test)
    F_test = np.concatenate((F_test , N_test) , axis = 1)
    
    print('F_test finished...')
    
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(F_train, train_label)
    print('Fitting...')
    print('Our ACC of RTrees is:...')   
    

    print(model.score(F_test, test_label))
    
    clf = MLPClassifier(random_state=1, max_iter=1000)
    clf.fit(F_train, train_label)
    print('Fitting...')
    print('Our ACC of MLP is:...')
    print(clf.score(F_test, test_label))
    
    '''

    

    
   
    
