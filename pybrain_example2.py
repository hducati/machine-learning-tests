from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer, SigmoidLayer

def main():
    """
        network = buildNetwork(2, 3, 1, outclass=SoftmaxLayer)
        print(network['in'])
        print(network['hidden0'])
        print(network['out'])
        print(network['bias'])
    """
    network = buildNetwork(2, 3, 1)
    data = SupervisedDataSet(2, 1)
    data.addSample((0, 0), (0, ))
    data.addSample((0, 1), (1, ))
    data.addSample((1, 0), (1, ))
    data.addSample((1, 1), (0, ))
    print(data['input'])
    print(data['target'])
    
    training = BackpropTrainer(network, dataset=data,
                               learningrate=0.01, momentum=0.06)
    
    for i in range(1, 30000):
        error = training.train()
        if i % 1000 == 0:
            print('Error: %s' % error)
    network.activate([0, 0])
    network.activate([0, 1])
    network.activate([1, 0])
    network.activate([1, 1])

if __name__ == '__main__':
    main()