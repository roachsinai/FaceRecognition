import numpy as np 
from matplotlib import pyplot  as plt
from mpl_toolkits.mplot3d import Axes3D
import time
class FisherFaces(object):
    def __init__(self):
        pass

    def test(self):
        x1=np.array([[4,1],[2,4],[2,3],[3,6],[4,4]])
        x2=np.array([[9,10],[6,8],[9,5],[8,7],[10,8]])
        x1_mean=x1.mean(axis=0)
        x2_mean=x2.mean(axis=0)
        S_w1=(x1-x1_mean).T.dot((x1-x1_mean))/5
        S_w2=(x2-x2_mean).T.dot((x2-x2_mean))/5
        S_w=S_w1+S_w2
        S_b=(x1-x2).T.dot((x1-x2))/5
        M=np.linalg.inv(S_w).dot(S_b)
        eigen_value,eigen_vector=np.linalg.eig(M)
        w=eigen_vector[:,0]
        #w=np.array([0.6,0.8])
        numerator=w.T.dot(S_b).dot(w)
        denominator=w.T.dot(S_w).dot(w)
        res=numerator/denominator
        pass

def main():
    model=FisherFaces()
    model.test()
    pass

if __name__ == '__main__':
    main()