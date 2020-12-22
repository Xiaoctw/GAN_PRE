#在这里测试一下GMM
from sklearn.mixture import GaussianMixture
from  datasets import *
import warnings
import attr
warnings.filterwarnings('ignore')
def fun(val):
    print(val)


if __name__ == '__main__':
    X,Y=RandomDataset(400,100)
    print(X.shape)
    print(Y.shape)
    gmm=GaussianMixture(n_components=3,covariance_type='full')
    print(GMM(X[:,0]))
    getattr('fun')(4)
    # gmm.fit(X)
    # #获取均值和方差
    # print(gmm.covariances_)
    # print(gmm.means_)
    # plot_data(X,gmm.predict(X))
    # print(gmm.predict_proba(X)[:5])
   # predict=gmm.predict(X)
  #  print(predict)
   # print(Y)