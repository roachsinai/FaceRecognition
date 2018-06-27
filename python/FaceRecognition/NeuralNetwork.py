import numpy as np 
from matplotlib import pyplot  as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import time
import cv2
import math

NUM_CLASSES = -1
input_channel_num=1
class NeuralNetwork(object):
    def __init__(self):
        self.face_U = None

    def load_data(self):
        num=1000
        path =  "F:/Datasets/LFW/raw_faces_data_" + str(num) + ".txt"
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        self.face_U = fs.getNode("face_U")
        self.face_U = self.face_U.mat().T

        self.people_names_numeral_mat = fs.getNode("people_names_numeral_mat")
        self.people_names_numeral_mat = self.people_names_numeral_mat.mat().squeeze()
        #self.people_names_numeral_mat=np.array(self.people_names_numeral_mat).reshape(-1,1)
        self.people_names_numeral_mat=self.people_names_numeral_mat.astype(np.int64)
        pass

    def train(self):
        self.load_data()
        image_h = 250
        image_w = 250

        regularizer = 0
        face_U = self.face_U
        people_names_numeral_mat=self.people_names_numeral_mat
        NUM_CLASSES=np.max(people_names_numeral_mat)
        N = face_U.shape[0]
        ratio = 0.8
        train_num=int(N * ratio)
        test_num=N-train_num
        data = {}
        data['x_train'] = face_U[0:train_num,:]
        data['y_train']=people_names_numeral_mat[0:train_num]
        data['x_test'] = face_U[train_num:,:]
        data['y_test']= people_names_numeral_mat[train_num:]
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32,[None,image_h,image_w,input_channel_num])
        y = tf.placeholder(tf.int64,[None,])

        from NeuralNetworks.SqueezeNet import SqueezeNet
        model_SN = SqueezeNet()
        y_pred = model_SN.model(x,NUM_CLASSES,input_channel_num=input_channel_num)
        y_pred = tf.reshape(y_pred,[-1,NUM_CLASSES])

        mean_loss = self.loss(y_pred,y)

        step = self.optimizer(mean_loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        N = data['x_train'].shape[0]
        x_train = data['x_train'].reshape(N,image_h,image_w,input_channel_num)
        y_train = data['y_train']

        N = data['x_test'].shape[0]
        x_test = data['x_test'].reshape(N,image_h,image_w,input_channel_num)
        y_test = data['y_test']

        self.run_model(sess,x,y,mean_loss,step,
                  y_pred,x_train,y_train,lr_decay=0.98,
                  epochs=200,batch_size=100,
                  print_every=10,plot_losses=False,dropout=0.5)

    def run_model(self,sess,x,y,mean_loss,train_step,y_pred,x_in,y_in,lr_decay=0.95,epochs=1,batch_size=100,print_every=100,plot_losses=False,dropout=0.5):
        N = x_in.shape[0]
        correct_prediction = tf.equal(tf.argmax(y_pred,1),y)
        accuracy = tf.reduce _mean(tf.cast(correct_prediction,tf.float32))
        train_indices = np.arange(x_in.shape[0])
        np.random.shuffle(train_indices)

        variables = [mean_loss,correct_prediction,accuracy]

        training_now = train_step is not None
        if training_now:
            print('Train')

            variables.append(train_step)
            iter_cnt = 0
            for e in range(epochs):
                correct = 0
                losses = []
                iter_num_per_epoch = int(math.ceil(N / batch_size))
                for i in range(iter_num_per_epoch):
                    start_idx = (i * batch_size) % N
                    over_num = start_idx + batch_size - train_indices.shape[0]
                    train_indices_merge = []
                    if over_num > 0:
                        train_indices_merge.append(train_indices[start_idx:start_idx + batch_size])
                        train_indices_merge.append(train_indices[0:over_num])
                        idx = np.concatenate(train_indices_merge,axis=0)
                    else:
                        idx = train_indices[start_idx:start_idx + batch_size]

                    feed_dict = {x:x_in[idx],
                               y:y_in[idx],
                               }
                    actual_batch_size = y_in[idx].shape[0]
                    loss,corr,accu,_ = sess.run(variables,feed_dict=feed_dict)
                    losses.append(loss * actual_batch_size)
                    correct+=np.sum(corr)
                    if training_now and (iter_cnt % print_every) == 0:
                        print('Iteration {0}: with minibatch training loss = {1:.3} and accuracy of {2:.2}'.format(iter_cnt,loss,accu))
                    iter_cnt+=1
                total_accuracy = correct / N
                total_loss = np.sum(losses) / N
                print('Epoch {2}, Overall loss = {0:.3} and accuracy of {1}'.format(total_loss,total_accuracy,e + 1))
                if plot_losses:
                    plt.plot(losses)
                    plt.grid(True)
                    plt.title('Epoch {} Loss'.format(e + 1))
                    plt.xlabel('minibatch number')
                    plt.ylabel('minibatch loss')
                    plt.show()
        else:
            print('Validation or Test')
            iter_num_per_epoch = int(math.ceil(N / batch_size))
            losses = []
            correct = 0
            for i in range(iter_num_per_epoch):
                start_idx = (i * batch_size) % N
                over_num = start_idx + batch_size - train_indices.shape[0]
                if over_num > 0:
                    train_indices_merge.append(train_indices[start_idx:start_idx + batch_size])
                    train_indices_merge.append(train_indices[0:over_num])
                    idx = tf.concat(train_indices_merge,axis=0)
                else:
                    idx = train_indices[start_idx:start_idx + batch_size]
                    feed_dict = {x:x_in[idx],
                               y:y_in[idx],
                               dropout_prob:dropout,
                               }
                loss,corr,accuracy = sess.run(variables,feed_dict=feed_dict)
                actual_batch_size = y_in[idx].shape[0]
                losses.append(loss * actual_batch_size)
                correct+=np.sum(corr)
                losses.append(loss * N)
                print('Iteration {0}: with minibatch training loss = {1:.3} and accuracy of {2:.2}'.format(i,loss,accuracy))
            total_accuracy = np.sum(correct) / N
            total_loss = np.sum(losses) / N
            print('Overall loss = {0:.3} and accuracy of {1}'.format(total_loss,total_accuracy))
        return total_loss,total_accuracy

    def loss(self,y_pred,y,reg=0):
        mean_loss = \
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=y_pred))
        return mean_loss
    
    def optimizer(self,mean_loss):
        global_step = tf.Variable(0,trainable=False)
        starter_learning_rate = 1e-3
        leanrning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,100,0.98,staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=leanrning_rate)
        train_step = optimizer.minimize(mean_loss,global_step=global_step)
        return train_step

def main():
    model = NeuralNetwork()
    model.train()

if __name__ == '__main__':
    main()