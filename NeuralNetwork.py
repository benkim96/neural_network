#import libraries
from numpy import exp, array, random, dot
import pygame

SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 900

BLACK, WHITE, RED, GREEN, BLUE = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)
random.seed(1)

class NeuralLayer():
    def __init__(self, num_nodes = 1):
        self.num_nodes = num_nodes

class NeuralBridge():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

        self.synaptic_weights = []
        self.__create_synaptic_weights()
        print("Synaptic Weights: ", self.synaptic_weights)

    def __create_synaptic_weights(self):
        for node in range(self.layer2.num_nodes):
            self.synaptic_weights.append((2 * random.random((1, self.layer1.num_nodes)) - 1)[0].tolist())

    def hypothesis_function(self, input_data, synaptic_weight):
        return self.__sigmoid(dot(input_data, synaptic_weight))

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def bridge_forward_propagate(self, input_data):
        output = []
        for node in range(self.layer2.num_nodes):
            output.append(self.hypothesis_function(input_data, self.synaptic_weights[node]))
        output = array(output).T
        return output

    def bridge_backward_propagate(self, delta, layer_input, layer_output):
        return

class NeuralNetwork():
    def __init__(self, neural_layers, training_input, training_output):
        self.layers = neural_layers
        self.training_input = training_input
        self.training_output = training_output
        self.bridges = []
        self.layer_inputs = []

        for index in range(len(self.layers) - 1):
            self.bridges.append(NeuralBridge(self.layers[index], self.layers[index + 1]))

    def train(self, iterations=1):
        for _ in range(iterations):
            self.layer_inputs = []
            self.layer_inputs.append(self.training_input)
            self.forward_propagate()
            self.backward_propagate()
            print(self.layer_inputs)

    def forward_propagate(self, bridge_index=0):
        if bridge_index == len(self.bridges):
            return
        self.layer_inputs.append(self.bridges[bridge_index].bridge_forward_propagate(self.layer_inputs[bridge_index]))
        self.forward_propagate(bridge_index + 1)

    def sigmoid_gradient(self, x):
        return x * (1 - x)

    def backward_propagate(self, bridge_index=0, delta=0):
        layer_output = self.layer_inputs[-1 - bridge_index]
        if bridge_index == len(self.bridges):
            return
        elif bridge_index == 0:
            error = self.training_output - layer_output
        else:
            error = dot(delta, self.bridges[-bridge_index].synaptic_weights)

        delta = error * self.sigmoid_gradient(layer_output)
        adjustment = dot(self.layer_inputs[-2 - bridge_index].T, delta).T
        self.bridges[-1 - bridge_index].synaptic_weights += adjustment
        self.backward_propagate(bridge_index + 1, delta)

class NeuralNetworkVisualizer():
    def __init__(self, neural_network):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 100))
        self.neuron_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.neuron_surface = self.neuron_surface.convert_alpha()
        pygame.display.set_caption("Neural Network Visualizer")
        clock = pygame.time.Clock()

        self.neural_network = neural_network
        self.layer_coordinates = []

        self.input_display()
        self.synapse_display()

    def input_display(self):
        radius = 30
        x_pos = 150
        x_gap = (SCREEN_WIDTH - (len(self.neural_network.layers) * radius * 2)) // (len(self.neural_network.layers) - 1)

        for layer in self.neural_network.layers:
            neuron_coordinates = []
            y_gap = (SCREEN_HEIGHT - (layer.num_nodes * radius * 2)) // (layer.num_nodes +  1)
            y_pos = radius + y_gap
            for node in range(layer.num_nodes):
                coordinate = (x_pos, y_pos)
                pygame.draw.circle(self.neuron_surface, WHITE, coordinate, radius, 5)
                neuron_coordinates.append((x_pos, y_pos))
                y_pos += y_gap + (2 * radius)
            x_pos += x_gap
            self.layer_coordinates.append(neuron_coordinates)
        print(self.layer_coordinates)

    def synapse_display(self):
        for _ in range(1):
            self.screen.fill(BLACK)
            self.neural_network.train(1)
            synaptic_surface = pygame.Surface.copy(self.neuron_surface)
            Font = pygame.font.SysFont(None, 48)


            for layer_index in range(len(self.neural_network.layers) - 1):
                if layer_index == 0:
                    text = Font.render('INPUT', True, WHITE, BLACK)
                else:
                    text = Font.render('HIDDEN LAYER', True, WHITE, BLACK)
                textrect = text.get_rect()
                textrect.centerx  = self.layer_coordinates[layer_index][0][0]
                textrect.centery = 50
                self.screen.blit(text, textrect)

                for node_index in range(self.neural_network.layers[layer_index].num_nodes):
                    for out_node_index in range(self.neural_network.layers[layer_index + 1].num_nodes):
                        color, width = self.__synapse_preprocess(self.neural_network.bridges[layer_index].synaptic_weights[out_node_index][node_index])
                        pygame.draw.line(synaptic_surface, color, self.layer_coordinates[layer_index][node_index],
                                self.layer_coordinates[layer_index + 1][out_node_index], width)

            self.screen.blit(synaptic_surface, (0, 100))

            text = Font.render("OUTPUT", True, WHITE, BLACK)
            textrect = text.get_rect()
            textrect.centerx = self.layer_coordinates[-1][0][0]
            textrect.centery = 50

            self.screen.blit(text, textrect)
            pygame.display.update()

    def __synapse_preprocess(self, synaptic_weight):
        color = (0, 0, 0)
        width = int(synaptic_weight)
        if synaptic_weight > 0:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
            width = -1 * width
        return color, ((width + 1) * 3)

    def train(self, iteration=1):
        for count in range(iteration):
            self.synapse_display()

    def display(self):
        running = True
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            pygame.quit()
        except SystemExit:
            pygame.quit()

if __name__ == "__main__":
    # initialize training data
    input_data = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    output_data = array([[0, 1, 1, 1, 1, 0, 0]]).T

    input_layer = NeuralLayer(3)
    hidden_layer_1 = NeuralLayer(10)
    output_layer = NeuralLayer(1)

    layers = [input_layer, hidden_layer_1, output_layer]
    layer = [NeuralLayer(3), NeuralLayer(5), NeuralLayer(10), NeuralLayer(1)]

    neural_network = NeuralNetwork(layer, input_data, output_data)
    #neural_network.train(1000)
    visualizer = NeuralNetworkVisualizer(neural_network)
    visualizer.train(1000000)
    visualizer.display()
