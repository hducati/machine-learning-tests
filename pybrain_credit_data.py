from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure.connections import FullConnection


def main():
    network = FeedForwardNetwork()
    input_layer = LinearLayer(2)
    hidden_layer = SigmoidLayer(3)
    output_layer = SigmoidLayer(1)
    # one for the entry_level
    bias_one = BiasUnit()
    # one for the hidden_layer
    bias_two = BiasUnit()
    list_network = [input_layer, hidden_layer, output_layer, bias_one, bias_two]
    for x in list_network:
        network.addModule(x)
    input_hidden = FullConnection(input_layer, hidden_layer)
    output_hidden = FullConnection(hidden_layer, output_layer)
    hidden_bias = FullConnection(bias_one, hidden_layer)
    output_bias = FullConnection(bias_two, output_layer)
    # establish connection
    network.sortModules()
    print_neural_network(network, input_hidden, output_hidden, hidden_bias, output_bias)
    

def print_neural_network(network, input_hidden, output_hidden, hidden_bias, output_bias):
    print(network)
    print(input_hidden.params)    
    print(output_hidden.params)
    
if __name__ == '__main__':
    main()