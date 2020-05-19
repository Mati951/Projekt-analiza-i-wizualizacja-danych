import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from enum import Enum

class Columns(Enum):
    instant = "instant"
    dteday = "dteday"
    pora_roku = "season"
    rok = "yr"
    miesiac = "mnth"
    wakacje = "holiday"
    powszedni = "weekday"
    roboczy = "workingday"
    weathersit = "weathersit"
    temp = "temp"
    atemp = "atemp"
    hum = "hum"
    predkosc_wiatru = "windspeed"
    casual = "casual"
    zarejestrowany = "registered"
    cnt = "cnt"

#
# # „pd_” jako przedrostek zmiennej oznacza „pochodna częściowa”
# # „d_” jako przedrostek zmiennej oznacza „pochodną”
# # „_wrt_” to skrót od „w odniesieniu do”
# # „w_ho” i „w_ih” są indeksem wag odpowiednio od neuronów ukrytej do wyjściowej i danych wejściowych do neuronów ukrytej warstwy

class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # Wykorzystuje uczenie się online, tj. Aktualizację ciężarów po każdym przypadku szkoleniowym
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Wyjściowe delty neuronów
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Ukryte delty neuronów
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # Musimy obliczyć pochodną błędu w odniesieniu do wyjścia każdego neuronu warstwy ukrytej
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

        # 3. Zaktualizuj wyjściowe wagi neuronów
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Zaktualizuj ukryte wagi neuronów
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Każdy neuron w warstwie ma tę samą tendencję
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # Zastosuj funkcję logistyczną
    # Wynik jest czasami określany jako „netto”
    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # Określ, ile całkowitego wkładu neuronu musi się zmienić, aby zbliżyć się do oczekiwanego wyniku
    #
    #      Teraz, gdy mamy częściową pochodną błędu w odniesieniu do wyniku (∂E / ∂yⱼ) i
    #      pochodna wyniku w stosunku do całkowitego wkładu netto (dyⱼ / dzⱼ), który możemy obliczyć
    #      częściowa pochodna błędu w stosunku do całkowitego wkładu netto.
    #      Ta wartość jest również znana jako delta (δ) [1]
    #      δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    # Błąd dla każdego neuronu oblicza się za pomocą metody Mean Square Error:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # Częściowa pochodna błędu w odniesieniu do rzeczywistej produkcji jest następnie obliczana przez:
    #      = 2 * 0,5 * (wyjście docelowe - wyjście rzeczywiste) ^ (2-1) * -1
    #      = - (wyjście docelowe - wyjście rzeczywiste)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)


    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)


    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

###

def prepare_inputs(record,  MIN_MAX_NORMALIZER):

    return[#record[Columns.instant.value],
    #record[Columns.dteday.value],
    MIN_MAX_NORMALIZER[Columns.pora_roku.value].normalize(record[Columns.pora_roku.value]),
    MIN_MAX_NORMALIZER[Columns.rok.value].normalize(record[Columns.rok.value]),
    MIN_MAX_NORMALIZER[Columns.miesiac.value].normalize(record[Columns.miesiac.value]),
    MIN_MAX_NORMALIZER[Columns.wakacje.value].normalize(record[Columns.wakacje.value]),
    MIN_MAX_NORMALIZER[Columns.powszedni.value].normalize(record[Columns.powszedni.value]),
    MIN_MAX_NORMALIZER[Columns.roboczy.value].normalize(record[Columns.roboczy.value]),
    MIN_MAX_NORMALIZER[Columns.weathersit.value].normalize(record[Columns.weathersit.value]),
    MIN_MAX_NORMALIZER[Columns.temp.value].normalize(record[Columns.temp.value]),
    MIN_MAX_NORMALIZER[Columns.hum.value].normalize(record[Columns.hum.value]),
    MIN_MAX_NORMALIZER[Columns.predkosc_wiatru.value].normalize(record[Columns.predkosc_wiatru.value]),
    MIN_MAX_NORMALIZER[Columns.casual.value].normalize(record[Columns.casual.value]),
    MIN_MAX_NORMALIZER[Columns.zarejestrowany.value].normalize(record[Columns.zarejestrowany.value]),
    MIN_MAX_NORMALIZER[Columns.cnt.value].normalize(record[Columns.cnt.value])]


def zwroc_etykiete(rekord):
    ilosc_osob = rekord[Columns.zarejestrowany.value]
    if ilosc_osob > 0 and ilosc_osob < 1099:
        return "MALO"
    if ilosc_osob > 1100 and ilosc_osob < 2099:
        return "SREDNIO"
    if ilosc_osob > 2100 and ilosc_osob < 3099:
        return "DUZO"
    if ilosc_osob > 3100:
        return "TLUM"

# Oczekiwane wyjście sieci
def get_expected_output(expected_value):

    if (expected_value=="MALO"):
        return[1, 0, 0, 0]
    if (expected_value=="SREDNIO"):
        return[0, 1, 0, 0]
    if (expected_value=="DUZO"):
        return[0, 0, 1, 0]
    if (expected_value=="TLUM"):
        return[0, 0, 0, 1]

def clean(data):
    data.dropna(how="all", axis='index', inplace=True)
    data.dropna(how="all", axis='columns', inplace=True)
    data.fillna(inplace=True, method="ffill")
    data.fillna(inplace=True, method="bfill")

def odczytaj_wszystkie_dane():
    data = pd.read_csv("day.csv", delimiter=",")
    return data

def odczytaj_dane_uczace():
    data = pd.read_csv("uczace.csv", delimiter=",")
    return data

def odczytaj_dane_testujące():
    data = pd.read_csv("testowe.csv", delimiter=",")
    return data

def odczytaj_dane_walidujace():
    data = pd.read_csv("walidujace.csv", delimiter=",")
    return data

def change_value(val, max):
    if val == max:
        return 1
    else:
        return 0


def round_outputs(outputs):
    max_v = max(outputs)
    return [change_value(o, max_v) for o in outputs]

class Normalizer:
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.max_minus_min = max - min




    def normalize(self, value):
        if self.max_minus_min == 0:
            return 0
        return (value - self.min) / self.max_minus_min


def main():
    records = odczytaj_wszystkie_dane()
    clean(records)

    MIN_MAX_NORMALIZER = {}
    for col in [Columns.pora_roku, Columns.rok, Columns.miesiac, Columns.wakacje, Columns.powszedni, Columns.roboczy,
                Columns.weathersit, Columns.temp, Columns.hum, Columns.predkosc_wiatru, Columns.casual,
                Columns.zarejestrowany, Columns.cnt]:
        tmp_col = records[col.value]
        MIN_MAX_NORMALIZER[col.value] = Normalizer(min(tmp_col), max(tmp_col))

    nn = NeuralNetwork(13, 13, 4, hidden_layer_weights=None, hidden_layer_bias=0.6,
                       output_layer_weights=None, output_layer_bias=0.7)

    print("UCZENIE START")

    records = odczytaj_dane_uczace()
    clean(records)

    PLOT_DATA = []
    for i in range(500):
        print("EPOKA: " + str(i))

        for index, row in records.iterrows():
            inputs = prepare_inputs(row, MIN_MAX_NORMALIZER)
            output = get_expected_output(zwroc_etykiete(row))

            nn.train(inputs, output)

        success_counter = 0
        for index, row in records.iterrows():
            inputs = prepare_inputs(row, MIN_MAX_NORMALIZER)
            output = get_expected_output(zwroc_etykiete(row))
            result = nn.feed_forward(inputs)
            if np.array_equal(output, round_outputs(result)):
                success_counter += 1

        PLOT_DATA.append(success_counter)

    print("UCZENIE STOP")

    # =============================================================================================================================
    print("TESTOWANIE START")
    testing_records = odczytaj_dane_testujące()
    clean(testing_records)

    testing_success_counter = 0;
    for index, row in testing_records.iterrows():
        inputs = prepare_inputs(row, MIN_MAX_NORMALIZER)
        output = get_expected_output(zwroc_etykiete(row))

        result = nn.feed_forward(inputs)
        if np.array_equal(output, round_outputs(result)):
            testing_success_counter += 1

    print("TESTOWANIE STOP")
    print("SUKCESY DANYCH TESTOWYCH: " + str(testing_success_counter))

    # =============================================================================================================================
    print("WALIDUJACE START")
    validating_records = odczytaj_dane_walidujace()
    clean(validating_records)

    validating_success_counter = 0;
    for index, row in validating_records.iterrows():
        inputs = prepare_inputs(row, MIN_MAX_NORMALIZER)
        output = get_expected_output(zwroc_etykiete(row))

        result = nn.feed_forward(inputs)
        if np.array_equal(output, round_outputs(result)):
            validating_success_counter += 1

    print("WALIDUJACE STOP")
    print("SUKCESY DANYCH WALIDUJACYCH: " + str(validating_success_counter))
    # =============================================================================================================================

    plt.plot(PLOT_DATA)
    plt.show()
# ======================================================================
# print("EXPECTED " + str(zwroc_etykiete(row)))
# print("RESULT " + str(nn.feed_forward(inputs)))

if __name__ == '__main__':
    main()
