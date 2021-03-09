

class perceptron():
    
    def __init__(self,eta,beta,epoch):
        self.eta=eta
        self.beta=beta
        self.epoch=epoch
        
    def activation(self,inputs,weights):
        X=np.array(inputs)
        W=np.array(weights)
        return sum(W*X)
    
    def sigmoid(self,a):
        return 1/(1+np.e**(-self.beta*a))
    
    def error(self,s,y):
        return ((s-y)**2)/2
    
    def output(self,Weight):
        for i in zip(self.df['dim1'],self.df['dim2'],self.df['class']):
            X=np.array([1,i[0],i[1]])
            sig=self.sigmoid(self.beta,self.activation(X,Weight))
        return sig
    
    def train(self,df,CLASS,W_initial):
        self.df=df
        self.CLASS=CLASS
        self.W_initial=W_initial
        avg_error=0
        
        for j in range(self.epoch):
            
            for i in zip(df['dim1'],df['dim2'],df['class']):
                X=np.array([1,i[0],i[1]])
                sig=self.sigmoid(self.activation(X,W_initial))
                
                # if sig>0.5:
                #     yn_pred=1
                # else:
                #     yn_pred=0
                    
                    
                c1=1 if i[2]==CLASS else 0
                avg_error+=self.error(sig,c1)
                W_initial+=self.eta*(c1-sig)*self.beta*sig*(1-sig)*X   #updating weight
            
            avg_error=avg_error/(2*len(Train))  #updating avg_error
        return W_initial, avg_error
        
    def test(self,df,Weight,Class):
        self.df=df
        predicted=[]
        
        for i in zip(df['dim1'],df['dim2'],df['class']):
            Input=np.array([1,i[0],i[1]])
            sig=self.sigmoid(self.activation(Input,Weight))
        
            if sig>0.5:
                yn_pred=Class
            else:
                yn_pred=0  
            predicted.append(yn_pred)
            
        return predicted
