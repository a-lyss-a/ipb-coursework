import numpy as np
import matplotlib.pyplot as plt
import argparse


def main(args):
    myNetwork = Network()
    myNetwork.forward(1000)
    

class PEUnit:
    def __init__(self, r_size, I_size, alpha, beta, gamma, epsilon, zeta, eta, theta):
        
        self.r = np.zeros((r_size))
        self.U = np.random.rand(r_size, I_size) # i think this might be wrong
        self.V = np.random.rand(r_size, r_size)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.zeta = zeta
        self.eta = eta
        self.theta = theta

        '''okay listen i know that a lot of these constants have factors in common,
        but i don't know what they should be so i'm just using these greek letters
        as stand-ins.
        these are all called "learning rates" in the optimal estimation paper, but
        they all share a common, normalising(?) value in the other two.'''

    # Update r, V, U based on a new input I
    def update(self, I, err_td):
        # estimate of the last value of true r (r' or r-dash)
        # this is now self.r so i don't need to calculate it so many times
        # est_r = np.matmul(self.V, self.r) # this should be passed through a nonlinearity
        # error from bottom-up
        err_bu = I - np.matmul(self.U, self.r)
        # r-hat post-Kalman filter
        r_hat = (self.r
                 + self.alpha * (self.U.T * err_bu)
                 + self.beta * err_td
                 - self.gamma * self.r)

        # transition matrix from previous timestep to now
        new_V = self.V + self.epsilon * np.matmul( (r_hat - self.r), r_hat.T ) - self.eta
        # weights; i'm not sure why we give this r-hat rather than r', just that they do in the paper
        new_U = self.U + self.zeta * ( (I - np.matmul(self.U, r_hat) ) * r_hat.T ) - self.theta

        self.r = np.matmul(self.V, r_hat) # we should still probably pass this through a nonlinearity
        self.V = new_V
        self.U = new_U

        print("U =",self.U)
        print("V =",self.V)
        print("r =",self.r)

        # we need to output predictions to hand forwards, errors to hand backwards
        return self.predict(), err_bu
    
    # generate a prediction
    def predict(self):
        return np.matmul(self.U, self.r)

        
# for arranging and running the network
class Network:
    def __init__(self):
        self.step = 0

        # noise generator
        self.noise = np.random.default_rng()
        # define the units and their outputs
        self.layer1 = [PEUnit(5, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)]
        self.layer2 = [PEUnit(5, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)]

    def generate_input(self):
        return np.sin(self.step) + self.noise.normal(scale=0.05)
    
    # this is stacked right now because i haven't thought hard enough about the hierarchy
    def forward(self, steps):
        # outputs to be given to the next layers
        # THIS WILL HAVE TO BE CHANGED FOR MULTITHREADING
        # initialise everything
        errs_21 = [0]
        preds_12 = [0]
        errs_32 = [0]
        preds_23 = [0]
        while self.step < steps:
            
            # if r(t) = [0] then i guess our first real r is 1
            self.step += 1

            # generate input ( imagine this is I(t) )
            It = self.generate_input()

            # update everything and put the predictions into an array
            for i in range(len(self.layer1)):
                preds_12[i], _ = self.layer1[i].update(It, errs_21[i])

            # ideally, you wouldn't be doing this step-by-step, but i guess this works for now
            print("layer 2")

            # hand predictions up and errors down
            for i in range(len(self.layer2)):
                preds_23[i], errs_21[i] = self.layer2[i].update(preds_12[i], errs_32[i])

            if self.step % 10 == 0:
                print(f"input: {It}, layer1: {preds_12}, layer2: {preds_23}")

# i forgot how the shell works
main(1)