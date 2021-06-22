import random
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import gzip
import cv2
from numpy.core.fromnumeric import sort

def load_data():
    with gzip.open('mnist/train-images-idx3-ubyte.gz','rb') as f:
        train_X=idx2numpy.convert_from_file(f)

    with gzip.open('mnist/t10k-images-idx3-ubyte.gz','rb') as f:
        test_X=idx2numpy.convert_from_file(f)

    with gzip.open('mnist/train-labels-idx1-ubyte.gz','rb') as f:
        train_y=idx2numpy.convert_from_file(f)

    with gzip.open('mnist/t10k-labels-idx1-ubyte.gz','rb') as f:
        test_y=idx2numpy.convert_from_file(f)

    val_X=train_X[50000:]
    val_y=train_y[50000:]
    train_X=train_X[:50000]
    train_y=train_y[:50000]

    train_X=train_X.astype('float32')/255
    test_X=test_X.astype('float32')/255
    val_X=val_X.astype('float32')/255

    training_inputs=[np.reshape(x,(784,1)) for x in train_X]
    training_results=[vectorized_result(y) for y in train_y]
    training_data=zip(training_inputs,training_results)

    validation_inputs=[np.reshape(x,(784,1)) for x in val_X]
    validation_data=zip(validation_inputs,val_y)

    test_inputs=[np.reshape(x,(784,1)) for x in test_X]
    test_data=zip(test_inputs,test_y)

    return training_data,validation_data,test_data

def load_test_data(test_X,test_y):
    test_X=test_X.astype('float32')/255
    test_inputs=[np.reshape(x,(784,1)) for x in test_X]
    test_data=zip(test_inputs,test_y)

    return test_data

def key_sort(val):
    return val[1]

def preprocess_data():
    image=cv2.imread('img_9.jpg')
    #convert image to grayscale
    grey=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
    #set pixel value < 75-> 0,else 0.Then invert 0->1,1->0 
    _,thresh=cv2.threshold(grey.copy(),75,255,cv2.THRESH_BINARY_INV)
    #find the boundaries of digits,return list of boundaries
    # CV2.CHAIN_APPROX_SIMPLE used for only returning the boundaries of the formed shape
    #CV2.RETR_EXTERNAL for extracting only external contours
    contours, _=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    preprocessed_digits=[]

    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        # Creating a rectangle around the digit in the original image
        cv2.rectangle(image,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
        
        # Cropping out the digit from the image corresponding to current contour
        digit=thresh[y:y+h,x:x+w]
        
        # Resizing that digit to (18, 18)
        resized_digit=cv2.resize(digit,(18,18))
        
        # Padding the digit with 5 pixels of black color (zeros) on each side to finally produce the image of (28, 28)
        padded_digit=np.pad(resized_digit,((5,5),(5,5)),"constant",constant_values=0)

        if(w>10 and h>10):
          preprocessed_digits.append((padded_digit,x))

    preprocessed_digits.sort(key=key_sort)
    ans=[x[0] for x in preprocessed_digits]
    # for x in ans:
    #     plt.imshow(x,cmap="gray")
    #     plt.show()
    return np.array(ans)

def vectorized_result(j):
  e=np.zeros((10,1))
  e[j]=1.0
  return e

def sigmoid(z):
  return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z)*(1-sigmoid(z))

class CrossEntropyCost(object):
  @staticmethod
  def fn(a,y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
  @staticmethod
  def delta(z,a,y):
    return (a-y)

class QuadraticCost(object):
  @staticmethod
  def fn(a,y):
    return 0.5*np.linalg.norm(a-y)**2
  @staticmethod
  def delta(z,a,y):
    return (a-y)*sigmoid_prime(z)

class Network(object):
  def __init__(self,sizes,cost=CrossEntropyCost):
    self.num_layers=len(sizes)
    self.sizes=sizes
    self.default_weight_initializer()
    self.cost=cost

  def default_weight_initializer(self):
    self.biases=[np.random.randn(y,1) for y in self.sizes[1:]]
    self.weights=[np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

  def large_weight_initializer(self):
    self.biases=[np.random.randn(y,1) for y in self.sizes[1:]]
    self.weights=[np.random.randn(y,x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
 
  def feedforward(self,a):
    for b,w in zip(self.biases,self.weights):
      a=sigmoid(np.dot(w,a)+b)
    return a

  def SGD(self,training_data,epochs,mini_batch_size,eta,lmbda=0.0,
          evaluation_data=None,monitor_evaluation_cost=False,
          monitor_evaluation_accuracy=False,monitor_training_cost=False,
          monitor_training_accuracy=False,early_stopping_n=0):

    best_accuracy=1
    training_data=list(training_data)
    n=len(training_data)

    if evaluation_data:
      evaluation_data=list(evaluation_data)
      n_data=len(evaluation_data)

    best_accuracy=0
    no_accuracy_change=0

    evaluation_cost,evaluation_accuracy=[],[]
    training_cost,training_accuracy=[],[]

    for j in range(epochs):
      random.shuffle(training_data)
      mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch,eta,lmbda,len(training_data))
      print("Epoch %s training complete"%j)

      if monitor_training_cost:
        cost=self.total_cost(training_data,lmbda)
        training_cost.append(cost)
        print("Cost on training data: {}".format(cost))

      if monitor_training_accuracy:
        accuracy=self.accuracy(training_data,convert=True)
        training_accuracy.append(accuracy)
        print("Accuracy on training data: {}/{}".format(accuracy,n))

      if monitor_evaluation_cost:
        cost=self.total_cost(evaluation_data,lmbda,convert=True)
        evaluation_cost.append(cost)
        print("Cost on evaluation data: {}".format(cost))

      if monitor_evaluation_accuracy:
        accuracy=self.accuracy(evaluation_data)
        evaluation_accuracy.append(accuracy)
        print("Accuracy on evaluation data: {}/{}".format(accuracy, n_data))
      
      if early_stopping_n>0:
        if accuracy>best_accuracy:
          best_accuracy=accuracy
          no_accuracy_change=0
        else:
          no_accuracy_change+=1
        
        if no_accuracy_change==early_stopping_n:
          self.make_plots(evaluation_cost,evaluation_accuracy,training_cost,training_accuracy,j+1,n,n_data)
          return evaluation_cost,evaluation_accuracy,training_cost,training_accuracy

    self.make_plots(evaluation_cost,evaluation_accuracy,training_cost,training_accuracy,epochs,n,n_data)
    return evaluation_cost,evaluation_accuracy,training_cost,training_accuracy

  def update_mini_batch(self,mini_batch,eta,lmbda,n):
    nabla_b=[np.zeros(b.shape) for b in self.biases]
    nabla_w=[np.zeros(w.shape) for w in self.weights]
 
    for x,y in mini_batch:
      delta_nabla_b,delta_nabla_w=self.backdrop(x,y)
      nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
      nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]    

    self.weights=[(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
    self.biases=[b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]
  
  def backdrop(self,x,y):
    nabla_b=[np.zeros(b.shape) for b in self.biases]
    nabla_w=[np.zeros(w.shape) for w in self.weights]

    activation=x
    activations=[x]
    zs=[]

    for b,w in zip(self.biases,self.weights):
      z=np.dot(w,activation)+b
      zs.append(z)
      activation=sigmoid(z)
      activations.append(activation)

    delta=(self.cost).delta(zs[-1],activations[-1],y)
    nabla_b[-1]=delta
    nabla_w[-1]=np.dot(delta,activations[-2].transpose())

    for l in range(2,self.num_layers):
      z=zs[-l]
      sp=sigmoid_prime(z)
      delta=np.dot(self.weights[-l+1].transpose(),delta)*sp
      nabla_b[-l]=delta
      nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())
    return (nabla_b,nabla_w)

  def accuracy(self,data,convert=False):
    if convert:
      results=[(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y) in data]
    else:
      results=[(np.argmax(self.feedforward(x)),y) for (x,y) in data]
      for (x,y) in results:
          print(str(x)+"<predict-real> "+str(y))

    result_accuracy=sum(int(x==y) for (x,y) in results)
    return result_accuracy

  def total_cost(self,data,lmbda,convert=False):
    cost=0.0
    for x,y in data:
      a=self.feedforward(x)
      if convert:
        y=vectorized_result(y)
      cost+=self.cost.fn(a,y)/len(data)
      cost+=0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
    return cost

  def make_plots(self,evaluation_cost,evaluation_accuracy,training_cost,training_accuracy,num_epochs,training_data_size,test_data_size):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(np.arange(0, num_epochs),training_cost[0:num_epochs],color='#2A6EA6',label="Cost-training Data")
    ax.plot(np.arange(0, num_epochs),evaluation_cost[0:num_epochs],color='#FFA933',label="Cost-test data")
    ax.set_xlim([0, num_epochs])
    ax.set_title('Cost plot')
    plt.legend(loc="lower right")
    plt.show()

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(np.arange(0, num_epochs),[accuracy*100.0/training_data_size for accuracy in training_accuracy[0:num_epochs]],color='#2A6EA6',label="Accuracy on the training data")
    ax.plot(np.arange(0,num_epochs),[accuracy*100.0/test_data_size for accuracy in evaluation_accuracy[0:num_epochs]],color='#FFA933',label="Accuracy on the test data")
    ax.set_xlim([0, num_epochs])
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy plot')
    plt.legend(loc="lower right")
    plt.show()

#testing on mnist data
'''
training_data,validation_data,test_data=load_data()
training_data=list(training_data)
test_data=list(test_data)
validation_data=list(validation_data)
net = Network([784, 30, 10],cost=CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data,5,10,0.5,evaluation_data=test_data,monitor_training_accuracy=True,monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True)
'''

#testing on mobile snapshots
training_data,validation_data,_=load_data()
test_data_X=preprocess_data()
test_data_y=[0,5,2,2,1,8,5,9]
# test_data_y=[2,3,1,4,9,8,7,6,5,1,4,7]
# test_data_y=[2,3,6,5]
# test_data_y=[2,3,9,4]
test_data=load_test_data(test_data_X,test_data_y)
training_data=list(training_data)
validation_data=list(validation_data)
test_data=list(test_data)
net=Network([784,30,10],cost=CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data,30,10,0.5,lmbda=5.0,evaluation_data=test_data,monitor_training_accuracy=True,monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True)
