import numpy
import scipy.special

class neuralNetwork:
  def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
    self.inodes = inputnodes

    self.hnodes = hiddennodes
    self.onodes = outputnodes
          
    self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
    self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

    self.lr = learningrate


    self.activation_function = lambda x: scipy.special.expit(x)
    self.inverse_activation_function = lambda x: scipy.special.logit(x)

    pass

  def train(self, inputs_list, targets_list):
          
    inputs = numpy.array(inputs_list, ndmin=2).T
    targets = numpy.array(targets_list, ndmin=2).T

    hidden_inputs = numpy.dot(self.wih, inputs)
    hidden_outputs = self.activation_function(hidden_inputs)

    final_inputs = numpy.dot(self.who, hidden_outputs)
    final_outputs = self.activation_function(final_inputs)

    output_errors = targets - final_outputs
    hidden_errors = numpy.dot(self.who.T, output_errors) 


    self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

    self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    pass

    
    
  def query(self, inputs_list):
    inputs = numpy.array(inputs_list, ndmin=2).T


    hidden_inputs = numpy.dot(self.wih, inputs)
    hidden_outputs = self.activation_function(hidden_inputs)

    final_inputs = numpy.dot(self.who, hidden_outputs)
    final_outputs = self.activation_function(final_inputs)

    return final_outputs

  # FUNÇÃO PARA FAZER O BACKQUERYING
  # O target SERÃO OS VALORES DO LADO FINAL DIREITO DA RN, SERÃO USADOS COMO INPUTS
  # hidden_output É O SINAL AO LADO DIREITO DAS CÉLULAS DA HIDDEN LAYER
  def backquery(self, targets_list):
    # AQUI É FEITA A TRANSPOSIÇÃO DA LISTA targets PARA UMA MATRIX VERTICAL
    final_outputs = numpy.array(targets_list, ndmin=2).T

    # AQUI É CALCULA O SINAL PARA DENTRO DA CAMADA DE OUTPUT, USANDO A FUNÇÃO INVERSA DE ATIVAÇÃO
    final_inputs = self.inverse_activation_function(final_outputs)

    # AQUI É CALCULADO O SINAL AGORA PRA FORA DAS CÉLULAS DA CAMADA DO MEIO (HIDDEN LAYER)
    hidden_outputs = numpy.dot(self.who.T, final_inputs)
    # E AQUI É FEITA A RE-ESCALA DOS VALORES PARA FICAREM ENTRE 0.01 E 0.99
    hidden_outputs -= numpy.min(hidden_outputs)
    hidden_outputs /= numpy.max(hidden_outputs)
    hidden_outputs *= 0.98
    hidden_outputs += 0.01

    # E AQUI É CALCULADO PARA DENTRO DA CAMADA DE INPUT 
    hidden_inputs = self.inverse_activation_function(hidden_outputs)

    # E AQUI É ENTÃO TERMINADA O BACKQUERYING, SENDO FEITO O CALCULO DO SINAL PARA FORA DA CAMADA DE INPUT (A ESQUERDA)
    inputs = numpy.dot(self.wih.T, hidden_inputs)
    # ENTÃO RE-ESCALA OS VALORES NOVAMENTE
    inputs -= numpy.min(inputs)
    inputs /= numpy.max(inputs)
    inputs *= 0.98
    inputs += 0.01

    return inputs
    #PIMBA.