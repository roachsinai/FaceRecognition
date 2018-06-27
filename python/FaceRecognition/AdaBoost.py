import numpy as np 
class AdaBoost(object):
    def __init__(self,training_set):
        self.training_set = training_set
        self.N=len(self.training_set)
        self.weights = np.ones(self.N) / self.N
        self.RULES = []
        self.ALPHA = []

    def set_rule(self,func,test=False):
        for t in self.training_set:
            print(func(t[0]))
            #print(t[1])
        errors = np.array([t[1] != func(t[0]) for t in self.training_set])
        e = (errors * self.weights).sum()
        if test: return e
        alpha = 0.5 * np.log((1 - e) / e)
        print(e)
        print(alpha)
        w=np.zeros(self.N)
        for i in range(self.N):
            if errors[i] == 1:
                w[i] = self.weights[i] * np.exp(alpha)
            else:
                w[i] = self.weights[i] * np.exp(-alpha)
        self.weights = w / w.sum()
        self.RULES.append(func)
        self.ALPHA.append(alpha)

    def evaluate(self):
        NR = len(self.RULES)
        for (x,l) in self.training_set:
            hx = [self.ALPHA[i] * self.RULES[i](x) for i in range(NR)]
            print(x)
            print(np.sign(l) == np.sign(sum(hx)))

    def neural_net_solver(self):
        #y=a*x[0]+b*x[1]
        np.random.seed(666)
        coeffs=np.random.randn(2,32)*1e-1
        coeffs1=np.random.randn(32,2)*1e-1
        scores=[]
        y_true=[]
        D=len(self.training_set[0][0])
        x=np.zeros((self.N,D))
        for i in range(self.N):
            t=self.training_set[i]
            for j in range(len(t[0])):
                x[i,j]=t[0][j]
            y_true.append(t[1])
        label=[y==1 for y in y_true]
        label=np.array(label)
        label=label.astype(int)

        iter_num=10000
        learning_rate=1e-3
        learning_rate_decay=0.99
        reg=1e-4

        for i in range(iter_num):
            h=np.matmul(x,coeffs)
            #h=np.maximum(0,h)
            h=1/(1+np.exp(-h))
            scores=np.matmul(h,coeffs1)

            shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
            softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores)+1e-8, axis = 1).reshape(-1,1)
            loss = -np.sum(np.log(softmax_output[range(self.N), label]))
            loss /= self.N 
            #loss+=0.5*reg*(np.sum(coeffs*coeffs)+np.sum(coeffs1*coeffs1))
            #loss+=0.5*reg*(np.sum(coeffs*coeffs))
            dscores=softmax_output.copy()
            dscores[range(self.N),label]+=-1


            #cor_scores=scores[range(self.N),label].reshape(-1,1)
            #delta=0.1
            #margins=np.maximum(0,scores-cor_scores+delta)
            #margins[range(self.N),label]=0
            #loss=np.sum(margins)
            #loss/=self.N
            ##loss+=0.5*reg*(np.sum(coeffs1*coeffs1)+np.sum(coeffs1*coeffs1))
            #dscores=np.zeros((self.N,D))
            #dscores[margins>0]=1
            #dscores[range(self.N), list(label)] = 0
            #dscores[range(self.N), list(label)] = -np.sum(dscores,1)

            dcoeffs1=np.matmul(h.T,dscores)
            #dcoeffs/=self.N
            dcoeffs1/=self.N
            #dcoeffs1+=reg*coeffs1
            dh=np.matmul(dscores,coeffs1.T)
            #dh[dh<=0]=0
            dh=1/(1+np.exp(-h)) *(1-1/(1+np.exp(-h)))*dh
            dcoeffs=np.matmul(x.T,dh)
            #dcoeffs+=reg*dcoeffs

            if i%100==0 and i >=100:
                learning_rate*=learning_rate_decay
            coeffs-=learning_rate*dcoeffs
            coeffs1-=learning_rate*dcoeffs1
            
            h=np.matmul(x,coeffs)
            #h=np.maximum(0,h)
            h=1/(1+np.exp(-h))
            scores=np.matmul(h,coeffs1)
            y=np.argmax(scores,1)
            cor_num=np.sum(label==y)
            accuracy=cor_num/self.N
            print("loss: ",loss," accuracy: ",accuracy)

    @staticmethod
    def test():
        example = []
        example.append(((1,2),1))
        example.append(((1,4),1))
        example.append(((2.5,5.5),1))
        example.append(((3.5,6.5),1))
        example.append(((4,5.4),1))
        example.append(((2,1),-1))
        example.append(((2,4),-1))
        example.append(((3.5,3.5),-1))
        example.append(((5,2),-1))
        example.append(((5,5.5),-1))

        m=AdaBoost(example)

        m.set_rule(lambda x: 2*(x[0]<1.5)-1)
        m.set_rule(lambda x: 2*(x[0]<4.5)-1)
        m.set_rule(lambda x: 2*(x[1]>5)-1)

    @staticmethod
    def test1():
        example = []
        example.append(((1,2),1))
        example.append(((1,4),1))
        example.append(((2.5,5.5),1))
        example.append(((3.5,6.5),1))
        example.append(((4,5.4),1))
        example.append(((2,1),-1))
        example.append(((2,4),-1))
        example.append(((3.5,3.5),-1))
        example.append(((5,2),-1))
        example.append(((5,5.5),-1))

        m=AdaBoost(example)
        m.neural_net_solver();

def main():
    AdaBoost.test()
    #AdaBoost.test1()
    pass 

if __name__ == '__main__':
    main()