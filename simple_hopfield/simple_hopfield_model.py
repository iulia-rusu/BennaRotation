import torch
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(1)  


#assuming input pattern = single feature 




# feature_input = 50 #number of dimensions in initial input vector f
# n_neurons = 1000 #number of neurons in the network/ number of desired final dimensions
# n_input_vectors = 100 #number of f 






#set up stable states with feedforward matrix. Random feature vectors of n=50 dimensions are generated,
#expanded to 

class feedforward_matrix:
        def __init__ (self, n_neurons, feature_input, n_input_vectors):
            self.feature_input = feature_input
            self.n_neurons = n_neurons
            self.n_input_vectors = n_input_vectors
            self.Weights = torch.zeros((self.n_input_vectors, self.n_neurons))
            
            

    # Sets up continuous feedforward matrix
        def update_weights(self, n_neurons, feature_input):
            self.Weights = torch.randn((self.n_input_vectors, self.n_neurons))
            return self.Weights
          
    # report shape of weights matrix






        def generate_f(self, feature_input, n_input_vectors):
        # Ensure f is generated with a shape that matches the expectations for matrix multiplication
            f = torch.randn(self.feature_input, self.n_input_vectors)
            return f
        

        
#pass f through feedforward matrix to get x, transpose it, and name the returned value x
        def generate_x(self, f):
            x = torch.sign(f @self.Weights).T
            return x
        
        
        
# generate noisy x from x, by subsampling from the each vexctor in x
    



class Hopfield:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.Weights = torch.zeros((self.n_neurons, self.n_neurons))
        


        

    def create_matrix_static(self, n_states):
          
        sum_outer_products = torch.zeros((self.n_neurons, self.n_neurons))
    #samples a random distribution of input patterns     
        for _ in range(n_states):
            state = torch.randint(0, 2, self.n_neurons) #state vector, dimensioned, shaped by n_neurons
            state[state == 0] = -1
            outer_product = state.view(-1, 1) @ state.view(-1, 1).T
            sum_outer_products += outer_product
        
        self.Weights = sum_outer_products / self.n_neurons
        self.Weights.fill_diagonal_(0)

 # Update     
    
    def update(self, input, number_of_iterations):
        for _ in range(number_of_iterations):
            input = torch.sign(self.Weights @ input)
        return input
    

#function to retrieve the hopfield output pattern based of input pattern and number of iterations
    def retrieve(self, input, number_of_iterations) -> np.array:
        for _ in range(number_of_iterations):
            input = torch.sign(self.Weights @ input)
        return input
    print(input)
    
 
#calculate inner product of retrieved pattern input and noisy input pattern
    
    
    


      
# Plot the input patterns and the corresponding output patterns after 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 iterations







