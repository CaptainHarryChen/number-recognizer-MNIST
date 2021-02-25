import numpy as np
import Network
import mnist_loader

images,labels=mnist_loader.load_data("data","train")
train_data=mnist_loader.standardize(images,labels)
images,labels=mnist_loader.load_data("data","t10k")
test_data=mnist_loader.standardize(images,labels)

ann=Network.Network([784,30,10])

'''print(test_data[0][0])
print(ann.feedforward(test_data[0][0]))
print(test_data[0][1])'''

ann.SGD(train_data,epochs=30,mini_batch_size=10,eta=0.1,test_data=test_data)
