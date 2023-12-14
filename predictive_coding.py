import numpy as np
import matplotlib.pyplot as plt
import argparse


def main(args):
    myNetwork = Network()
    myNetwork.forward(args)
    

class PEUnit:
    def __init__(self, r_size, I_size, alpha, beta, gamma, epsilon, zeta, eta, theta):
        
        self.r = np.zeros((r_size, 1))
        self.U = np.random.rand(I_size, r_size)
        self.V = np.identity(5)
        #self.V = np.random.rand(r_size, r_size)
        # learning rates
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.zeta = zeta
        self.eta = eta
        self.theta = theta

        # noise generator
        self.noise = np.random.default_rng()

    def update(self, I, err_td):
        "Update r, V, U based on a new input I"

        # error from bottom-up
        err_bu = I - np.matmul(self.U, self.r) #1x1
        # r-hat post-Kalman filter
        cursed_r = np.matmul(np.linalg.inv(self.V), self.r)
        r_hat = (self.r
                 + self.alpha * (self.U.T * err_bu)
                 + self.beta * err_td)
                 #- self.gamma * self.r)

        # transition matrix from previous timestep to now
        new_V = self.V + self.epsilon * np.matmul( (r_hat - self.r), r_hat.T )# - self.eta * self.V
        # generative matrix
        new_U = self.U + self.zeta * ( (I - np.matmul(self.U, r_hat)) * r_hat.T ) #- self.theta * self.U

        self.r = np.tanh( np.matmul(self.V, r_hat) )# + self.noise.normal(scale=0.005)
        self.V = new_V
        self.U = new_U

        print("U =",self.U)
        print("V =",self.V)
        print("r =",self.r)

        # we need to output predictions to hand forwards, errors to hand backwards
        return self.predict(), err_bu
    
    def predict(self):
        "generate a prediction"
        return np.matmul(self.U, self.r)


class Network:
    "for arranging and running the network"
    def __init__(self):
        self.step = 0

        # noise generator
        self.noise = np.random.default_rng()
        # define the units and their outputs
        self.layer1 = [PEUnit(5, 1, 0.1, 0.1, 0.001, 0.1, 0.1, 0.1, 0.1)]
        self.layer2 = [PEUnit(5, 1, 0.1, 0.1, 0.001, 0.1, 0.1, 0.1, 0.1)]

    def generate_input(self, out_type = 0):
        step_forward = self.step/10
        if(out_type == 0): out = np.sin(step_forward) + self.noise.normal(scale=0.005) #was 0.05
        if(out_type == 1): out = np.sin(step_forward)    
        if(out_type == 2): out = 0.5
        return out
    
    # this is stacked right now because i haven't thought hard enough about the hierarchy
    def forward(self, steps):
        # outputs to be given to the next layers
        # initialise everything
        errs_21 = [0]
        preds_12 = [0]
        errs_32 = [0]
        preds_23 = [0]
        while self.step < steps:
            print("Step:", self.step, "~~~~~~~~LAYER 1~~~~~~~~")

            # if r(t) = [0] then i guess our first real r is 1
            self.step += 1

            # generate input
            It = self.generate_input(out_type= 0)

            # update everything and put the predictions into an array
            for i in range(len(self.layer1)):
                preds_12[i], _ = self.layer1[i].update(It, errs_21[i])

            print("~~~~~~~~LAYER 2~~~~~~~~")

            # hand predictions up and errors down
            for i in range(len(self.layer2)):
                preds_23[i], errs_21[i] = self.layer2[i].update(preds_12[i], errs_32[i])
            # check on the predictions
            if self.step % 10 == 0:
                print(f"input: {It}, layer1: {preds_12}, layer2: {preds_23}")

# i forgot how the shell works
rnaaaaa = np.random.default_rng()
random_length = np.floor(rnaaaaa.normal(loc = 700, scale=100))
main(random_length)