import numpy
import matplotlib.pyplot as plt
from neuralNetwork import *


input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.2

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

training_data_file = open("mnist_test.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 3

for e in range(epochs):
  for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass
    pass


test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
  all_values = record.split(',')
  correct_label = int(all_values[0])
  inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
  outputs = n.query(inputs)
  label = numpy.argmax(outputs)
  if (label == correct_label):
    scorecard.append(1)
  else:
    scorecard.append(0)
    pass
  pass

scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)

# AQUI ACONTE O PROCESSO INVERSO, O BACKQUERYING

# O RÓTULO QUE SERÁ TESTADO
label = 7
# CRIANDO OS SINAIS DE OUTPUT PARA ESSE RÓTULO, UMA LISTINHA DE 0.01s
targets = numpy.zeros(output_nodes) + 0.01
# AQUI ELE SEGUIRÁ O VALOR DO all_values PARA CONSIDERAR O RÓTULO
targets[label] = 0.99
# AQUI ELE PRINTA A LISTA COM 0.01s E O RÓTULO ESCOLHIDO COM 0.99
print(targets)

# AQUI ELE EXECUTA A FUNÇÃO CRIADA LÁ EM CIMA, FAZENDO O PROCESSO DE BACKQUERY COM O TARGET NAQUELE VALOR DE RÓTULO ESCOLHIDO, GERANDO UMA LISTONA DE 784 NÚMEROS
image_data = n.backquery(targets)

# E AQUI ELE GERA A IMAGEM TRANSFORMANDO OS 784 EM UMA MATRIX 28X28 E PIMBA.
plt.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')
plt.show()