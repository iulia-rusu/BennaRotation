import torch
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(1)  


#assuming input pattern = single feature 

# n_neurons = 1000 #dimensions/ neurons in final result
# feature_input = 2 #dimensions
# n_input_vectors = 10 # amount of feature vectors


#set up stable states with feedforward matrix. Random feature vectors of n=50 dimensions are generated,
#expanded to 

class feedforward_matrix:
        def __init__ (self, n_neurons, feature_input, n_input_vectors):
            self.feature_input = feature_input
            self.n_neurons = n_neurons
            self.n_input_vectors = n_input_vectors
            self.weights = torch.randn(self.n_neurons, self.feature_input)
            print(self.weights.shape)
            
        # def weights(self):
        #     return self.Weights
            
            
        # Sets up continuous feedforward matrix
        # def update_weights(self):
        #     self.Weights = torch.randn(self.n_input_vectors, self.n_neurons)
        #     return self.Weights
          
        def generate_f(self, feature_input, n_input_vectors):
        # Ensure f is generated with a shape that matches the expectations for matrix multiplication
            f = torch.randn(feature_input, n_input_vectors)
            return f
        
        #Generate noisy f by subsampling from the each vector in f
        def generate_noisy_f(self, f, noise_std=0.1):
        # Generate noise with the same shape as f
            noise = torch.randn(f.size()) * noise_std
        # Add the noise to f to create a noisy version of f
            noisy_f = f + noise
            return noisy_f
        
  
    
        def noise(self, f, noise_std=0.1, percent_of_vector=0.5):
        # Calculate the number of elements to perturb in each vector
            n_indexes = int(self.feature_input * percent_of_vector)
        # Generate a list of indexes to perturb for each vector
            idx_id = torch.randint(0, self.feature_input, (n_indexes,))
        # Generate noise only for selected elements
            selective_noise = torch.randn(n_indexes, self.n_input_vectors) * noise_std
        #copy f and only apply noise to the copy    
            noisy_f = f.clone()
        # Add noise only to selected indices
            for i in range(f.shape[1]):
                noisy_f[idx_id, i] += selective_noise[:, i]
        
            return noisy_f




        #pass f through feedforward matrix to get x, transpose it, and name the returned value x
        def generate_x(self, f):
            
                X = torch.sign(self.weights @  f)
                return X
        
        #Genrerate noisy_x
        def generate_noisy_x(self, noisy_f):
            
            X = torch.sign(self.weights @ noisy_f)
            return X
            



        

    



class Hopfield:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.Weights = torch.zeros((self.n_neurons, self.n_neurons))
        


        

    # def create_matrix_static(self, n_states):
          
    #     sum_outer_products = torch.zeros((self.n_neurons, self.n_neurons))
    # #samples a random distribution of input patterns     
    #     for _ in range(n_states):
    #         state = torch.randint(0, 2, self.n_neurons) #state vector, dimensioned, shaped by n_neurons
    #         state[state == 0] = -1
    #         outer_product = state.view(-1, 1) @ state.view(-1, 1).T
    #         sum_outer_products += outer_product
        
    #     self.Weights = sum_outer_products / self.n_neurons
    #     self.Weights.fill_diagonal_(0)


    # def create_hopfield_matrix(self, X):
    #     sum_outer_products = torch.zeros((self.n_neurons, self.n_neurons))

    #     for i, x in enumerate(X):
    #         outer_product = x.view(-1, 1) @ x.view(-1, 1).T
    #         sum_outer_products += outer_product
    #     self.Weights = sum_outer_products / self.n_neurons
    #     self.Weights.fill_diagonal_(0)
     
    # updated create_hopfield_matrix. take in tensor X, and iterate through each column, take i and j, calculate outer product, and sum the outer products
    def create_hopfield_matrix(self, X):
        sum_outer_products = torch.zeros((self.n_neurons, self.n_neurons))
        for i in range(X.shape[1]): 
            x = X[:, i]  # Select the i-th column/vector
            outer_product = x.view(-1, 1) @ x.view(-1, 1).T
            sum_outer_products += outer_product
            self.Weights = sum_outer_products / self.n_neurons
            self.Weights.fill_diagonal_(0) 



    def retrieve_states(self, input, n_iterations):
        
    # Returns:
    #     A tensor with shape [n_iterations, n_neurons, n_vectors] where each "slice" [i, :, :] is the state of the network
    #     at iteration i for each input vector.
    # """
        n_neurons, n_vectors = input.shape
    # Prepare an output tensor to hold the states at each iteration for each vector
        output_states = torch.zeros((n_iterations, n_neurons, n_vectors))
    
    # Iterate over each vector (column in X)
        for vector_index in range(n_vectors):
            current_state = input[:, vector_index]  # Initial state for this vector
        # Run through iterations, updating state and storing in output tensor
            for iteration in range(n_iterations):
                current_state = torch.sign(self.Weights @ current_state)
                output_states[iteration, :, vector_index] = current_state
    
        return output_states



    
    def compare_states(self, output_states, comparison_vectors):
        n_iterations, n_neurons, n_vectors = output_states.shape
        dot_products = torch.zeros((n_iterations, n_vectors))
    
        for iteration in range(n_iterations):
            for vector_index in range(n_vectors):
            # Extract the state vector at the current iteration
                state_vector = output_states[iteration, :, vector_index]
            # Extract the corresponding comparison vector
                comparison_vector = comparison_vectors[:, vector_index]
            # Calculate the dot product and store it
                dot_products[iteration, vector_index] = torch.dot(state_vector, comparison_vector)/n_neurons
    
        return dot_products
    

#Plotting

def plot_dot_products( dot_products):
    # """
    # Plot the dot products for each vector across iterations.
    
    # Args:
    #     dot_products (torch.Tensor): A tensor of shape [n_iterations, n_vectors] containing 
    #                                  dot products for each state vector at each iteration 
    #                                  with the corresponding comparison vector.
    # """
    n_iterations, n_vectors = dot_products.shape
    
    plt.figure(figsize=(12, 8))
    
    for vector_index in range(n_vectors):
        plt.plot(range(n_iterations), dot_products[:, vector_index], label=f'Vector {vector_index + 1}')
    
    plt.title('Dot Product Evolution Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Dot Product')
    plt.legend()
    plt.grid(True)
    plt.show()
      
# Plot the input patterns and the corresponding output patterns after 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 iterations







