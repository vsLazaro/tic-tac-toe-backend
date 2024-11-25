import torch

class NeuralLink:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Pesos e biases inicializados aleatoriamente entre -1 e 1
        self.weights_hidden = torch.empty(hidden_size, input_size).uniform_(-1, 1)
        self.bias_hidden = torch.empty(hidden_size).uniform_(-1, 1)
        self.weights_output = torch.empty(output_size, hidden_size).uniform_(-1, 1)
        self.bias_output = torch.empty(output_size).uniform_(-1, 1)

    def forward(self, x):
        # Camada oculta com função limiar
        hidden = torch.matmul(self.weights_hidden.float(), x) + self.bias_hidden.float()
        hidden = (hidden > 0).float()

        # Camada de saída com função limiar
        output = torch.matmul(self.weights_output.float(), hidden) + self.bias_output.float()
        return (output > 0).float()
