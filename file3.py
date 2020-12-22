import os
import json
import numpy as np
import warnings
import torch
import argparse
from datasets import *
from models import *
from train import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score,roc_auc_score,accuracy_score
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs to train.')
parser.add_argument('--dataset', default='Yeast', choices=['random_condition', 'Yeast', 'Wine'])
parser.add_argument('--num_gen', default=200, help='Number of generation')
parser.add_argument('--test_ratio', default=0.4)

args = parser.parse_args()

X, Y = load_dataset(args.dataset)
test_ratio = args.test_ratio
epochs = args.epochs

if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y=splitDataSet(X,Y,test_ratio)
    gen_X,gen_Y=load_generated_dataset(dataset=args.dataset)
    model=LogisticRegression()
    model.fit(train_X,train_Y.reshape(-1,1))
    predict_Y=model.predict(test_X)
    accuracy=accuracy_score(test_Y,predict_Y)
    recall=recall_score(test_Y,predict_Y,average='macro')
    precision=precision_score(test_Y,predict_Y,average='macro')
    print('dataset:{}, accuracy:{}, recall:{}, precision:{}'.format(args.dataset,accuracy,recall,precision))
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(gen_X, gen_Y.reshape(-1,1))
    predict_Y = model.predict(test_X)
    accuracy = accuracy_score(test_Y, predict_Y)
    recall = recall_score(test_Y, predict_Y,average='macro')
    precision = precision_score(test_Y, predict_Y,average='macro')
    print('dataset:{}, accuracy:{}, recall:{}, precision:{}'.format(args.dataset,accuracy,recall,precision))



