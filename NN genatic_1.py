import numpy as np
import random
from copy import deepcopy

#for activation function use sigmoid if you use other you have to change manually in code
class Nureal():
    lerning_rate = 0.4

    # here size = brain should be list of neuron of each layer
    def __init__(self, size, id=None,weight= None,baises=None):
        if weight == None :
            self.weight = np.array([np.random.rand(size[i + 1], size[i],) for i in range(len(size) - 1)],dtype=np.ndarray)
        else:
            self.weight = np.array(weight,dtype=np.ndarray)
        if  baises == None:
            self.biases = np.array([np.random.rand(j, 1) for j in size[1:]],dtype=np.ndarray)
        else:
            self.biases = np.array(baises,dtype=np.ndarray)
        # self.weight = [np.ones((size[i + 1], size[i])) for i in range(len(size) - 1)]
        # self.biases = [np.ones((j, 1)) for j in size[1:]]
        self.changed_weight = []
        self._output = []
        self._input = []
        self.point = 0
        self.id = id
        self.hidden_layers = []
        self.all_layers = []
        self.hidden_layers_error = []

    # input should be 1d array
    def forward(self, input):
        self._output = []
        self.hidden_layers = []
        self.all_layers = []
        self.hidden_layers_error = []
        input = [[i] for i in input]
        self._input = np.array(input, dtype=np.int64)
        self.all_layers = [self._input]
        input = np.array(input)
        for l in range(len(self.weight)):
            if l == len(self.weight) - 1:
                self._output = np.add(np.matmul(self.weight[l], input), self.biases[l])
                continue
            input = np.add(np.matmul(self.weight[l], input), self.biases[l])
            input = self.activation_sigmoid(input)
            self.hidden_layers.append(input)
            self.all_layers.append(input)
        self._output = self.activation_sigmoid(self._output)
        # self._output = self.activation_relue(self._output)
        self.all_layers.append(self._output)
        return self._output

    def activation_relue(self, layer):
        layer = np.array(layer, dtype=np.ndarray)
        k = layer.argmax()
        for i in range(len(layer)):
            if i == k:
                for j in range(len(layer[i])):
                    layer[i][j] = float(1.0)
            else:
                for j in range(len(layer[i])):
                    layer[i][j] = float(0.0)
        return layer

    def activation_softmax(self, layer):
        m = (np.exp(layer))
        return m / m.sum()

    def activation_sigmoid(self, layer):
        layer = layer.astype(float)
        try:
            np.exp(-layer)
        except RuntimeWarning:
            print(layer)
        return 1 / (1 + np.exp(-layer))

    # this sigmoiddelta is partial differentiation of sigmoid function
    @staticmethod
    def sigmoiddelta(error, output):
        ones = [[1] for i in range(len(output))]
        return np.multiply(error, np.multiply(output, np.subtract(ones, output)))

    @staticmethod
    def error_calculation(result, output):
        error = np.subtract(output, result)
        return error

    #how much part of loss we are correcting
    @staticmethod
    def delta(error):
        return (np.multiply(Nureal.lerning_rate, error))

    @staticmethod
    def transpose(m):
        return np.matrix.transpose(m)

    #result should be 1d list
    def backward(self, result):
        result = [[i] for i in result]
        error = Nureal.error_calculation(result, self._output)
        self.hidden_layers_error.append(error)
        for i in range(len(self.hidden_layers)):
            temp = np.matmul(Nureal.transpose(self.weight[-1 - i]), self.hidden_layers_error[0])
            # temp = np.matmul(Nureal.transpose(self.weight[-1 - i]),
            #                  Nureal.sigmoiddelta(self.hidden_layers_error[0],self.all_layers[-1-i]))
            self.hidden_layers_error.insert(0, temp)

        for i in range(len(self.weight)):
            temp = np.matmul(Nureal.sigmoiddelta(self.hidden_layers_error[i], self.all_layers[i + 1]),
                             Nureal.transpose(self.all_layers[i]))

            self.weight[i] = np.subtract(self.weight[i], Nureal.delta(temp))
            self.biases[i] = np.subtract(self.biases[i],
                                         Nureal.delta(
                                              Nureal.sigmoiddelta(self.hidden_layers_error[i], self.all_layers[i + 1])))

        self._output = []
        return error


class genatic():
    mutation_no = 3
    # ids different from position of creature but in this code we are not using id to access creature
    # brain should be list of neuron of each layer
    def __init__(self, size, brain):
        self.ids = 0
        self.brain = brain
        self.weights = [[self.brain[i+1],self.brain[i]]for i in range(len(self.brain)-1)]
        self.size = 0
        self.population = np.array([], dtype=object)
        self.Makepopulation(size)

    def Makepopulation(self,size = None):
        for i in range(size):
            temp = Nureal(self.brain, self.ids)
            self.ids +=1
            self.population = np.append(self.population, temp)
        self.size = self.population.size
        # this below code is for i don"t whana my creature id like 1425536377829929
        # id is just for our understanding in this code position in population matter
        if self.ids>self.size*5:
            self.ids = 0

    # if you don't want back propagation  output == None
    def Train(self, position_of_creature, input, output=None):
        if output != None:
            work = self.population[position_of_creature].forward(input)
            self.population[position_of_creature].backward(output)
        else:
            work = self.population[position_of_creature].forward(input)
        return work

    # assign points to specific neuron model
    def Evaluation(self,position_of_creature, points):
        self.population[position_of_creature].point += float(points)

    # No of kill  = kill_size
    # condition is different for different so write your own
    def kill(self, kill_size, condition=None):
        topcreature = {}
        if self.population.size == 0:
            print("all are kiled")
            return
        if condition == None and kill_size < self.size - 1:
            self.size = self.population.size
            for position_of_creature in range(self.population.size):
                point = self.population[position_of_creature].point
                # again all models point = 0 for new age
                self.population[position_of_creature].point = 0
                if point not in topcreature:
                    topcreature[point] = [position_of_creature]
                else:
                    # two models can have same points
                    topcreature[point].append(position_of_creature)
            # gives soreted points
            keys = sorted(topcreature)
            #  position_of_creature_top it is for stored position
            position_of_creature_top =[]
            for i in keys:
                position_of_creature_top+=topcreature[i]
            it = 0
            # top it is for reassign creature based on points
            top = []
            for i in position_of_creature_top:
                if it < kill_size:
                    self.population[i]=0
                    it+=1
                else:
                    top.append(self.population[i])
            print(position_of_creature_top[0], position_of_creature_top[-1], keys[0], keys[-1])
            # reassign creature based on points
            self.population = np.array(top[::-1],dtype=object)
        else:
            pass

    #gives random position for weight in 3d array of weight
    def random_pos_weight(self):
        weight_layer = random.randint(0, len(self.brain)-2)
        weight = self.weights[weight_layer]
        weight_layer_row = random.randint(0,weight[0]-1)
        weight_element = random.randint(0, weight[1]-1)
        return [weight_layer, weight_layer_row, weight_element]

    # gives random position for biases in 3d array of weight
    def random_pos_biases(self):
        biases_layer = random.randint(0,len(self.brain)-2)
        biases = random.randint(0,self.brain[biases_layer+1]-1)
        return [biases_layer,biases,0]

    # change random element by adding random number positive or negative
    # lr is how much you want random weight
    def self_Mutation(self, id,lr=2,mutation_neuron_no=mutation_no):
        for i in range(mutation_neuron_no):
            w_layer_no, row, element = self.random_pos_weight()
            ind = self.random_pos_biases()
            sign = random.randint(0,1)
            if sign == 0:
                sign = -1
            gate = random.randint(0, 2)
            if gate == 0 or gate == 2:
                self.population[id].weight[w_layer_no][row][element] += random.random()*sign*lr
            if gate == 1 or gate == 2:
                self.population[id].biases[ind[0]][ind[1]][ind[2]] +=random.random()*sign*lr


    # mutent will give weight element to normal creature
    def MutualMutation(self, normal_id,mutent_id , mutation_neuron_no=3):
        for i in range(mutation_neuron_no):
            sign = random.randint(0,2)
            if sign == 2 or sign == 1:
                ind = self.random_pos_weight()
                self.population[normal_id].weight[ind[0]][ind[1]][ind[2]]=self.population[mutent_id].weight[ind[0]][ind[1]][ind[2]]
            if sign == 0 or sign == 2:
                ind = self.random_pos_biases()
                self.population[normal_id].biases[ind[0]][ind[1]][ind[2]]=self.population[mutent_id].biases[ind[0]][ind[1]][ind[2]]

    # n_o_w_l_recive_f_m = number of ELEMENT change in child
    # self mutation occure when female is not present
    #  N_O_w_r_f_f no of weight recive from female
    def child(self, male_id,n_o_w_l_recive=3,lr = 5,female_id = None, N_O_w_r_f_f = 0):
        new_child = Nureal(self.brain, self.ids,deepcopy(self.population[male_id].weight),deepcopy(self.population[male_id].biases))
        self.ids+=1
        if self.ids>self.size*5:
            self.ids = 0
        self.population = np.append(self.population, new_child)
        self.size  = self.population.size
        if female_id == None:
            self.self_Mutation(self.size-1,lr,n_o_w_l_recive)
        else:
            for i in range(N_O_w_r_f_f):
                sign = random.randint(0, 2)
                if sign == 2 or sign == 1:
                    ind = self.random_pos_weight()
                    self.population[self.size-1].weight[ind[0]][ind[1]][ind[2]] = \
                    self.population[female_id].weight[ind[0]][ind[1]][ind[2]]
                if sign == 0 or sign == 2:
                    ind = self.random_pos_biases()
                    self.population[self.size-1].biases[ind[0]][ind[1]][ind[2]] = \
                    self.population[female_id].biases[ind[0]][ind[1]][ind[2]]

    #for predicting in last
    def predict(self,id,input):
        return self.population[id].forward(input)

# xor model
def xor():
    # 50 creature with brain [2,4,1]
    a = genatic(50,[2,4,1])
    b=[[1,0],[0,1],[1,1],[0,0]]
    # 80 ages
    for m in range(80):
        for k in range(a.size):
            random.shuffle(b)
            for h in range(4):
                j = b[h]
                output = a.Train(k,j)
                if j == [0,0 ]or j ==[1,1]:
                    points = -round(float(output),5)
                    a.Evaluation(k,points)
                elif j == [0,1]or j == [1,0]:
                    points = round(float(output),5)
                    a.Evaluation(k,points)

        # killing half of population
        a.kill(25)
        for i in range(20):
            parent1 = random.randint(0,24)
            a.child(parent1)
        #making new creature show every time we got some randomness
        a.Makepopulation(5)
        # mutent weights will passed to normal
        for i in  range(0,5):
            normalid = random.randint(24,49)
            mutentid = random.randint(0,24)
            a.MutualMutation(normalid,mutentid,5)
        print("AGE :", m,'\n')
        print("\n")
    creatures = {}


    for i in range(a.population.size):
        for j in b:
            output = a.predict(i, j)
            if j == [0, 0] or j == [1, 1]:
                points = -round(float(output), 5)
                a.Evaluation(i, points)
            elif j == [0, 1] or j == [1, 0]:
                points = round(float(output), 5)
                a.Evaluation(i, points)
            print(i,j,a.predict(i,j),a.population[i].point)
        print()

if __name__ == "__main__":
    xor()
