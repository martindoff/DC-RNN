""" Difference of Convex deep Recurrent Neural Network (DC-RNN) model - coupled tank

Approximation of the coupled tank dynamics by a RNN model with DC structure.

(c) Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.constraints import NonNeg
from tank_model import f
from control_custom import eul
import matplotlib.pyplot as plt
import param_init as param


# RNN with convex structure 
def convex_RNN(N_units, N_state, N_input):
    """ Create a recurrent neural network with convex input-output map
    Input: 
        - N_units: number of units
        - N_state: number of states
        - N_input: number of inputs
    Output: keras neural network model
    """
    
    # Initial condition layer
    x_init = keras.Input(shape=(N_state, ))
    state_init = layers.Dense(N_units)(x_init)
    
    # Input sequence
    input = keras.Input(shape=(None, N_input))
    
    # Simple RNN state update layer (with nonnegatively constrained weights)
    state = layers.SimpleRNN(N_units, activation='relu', recurrent_constraint=NonNeg(),
                             return_sequences=True)(input, initial_state=state_init)
    
    # Dense output layer (with nonnegatively constrained weights)
    output = layers.Dense(N_state, kernel_constraint=NonNeg())(state)
    
    return keras.Model([x_init, input], output)

# main
if __name__ == "__main__":
    """ 
    Test the DC neural network architecture on an example
    
    """
    load = False # set to False if model has to be retrained
    
    # Load data (generated with "generate_data.py")
    X = np.load('input.npy')
    Y = np.load('output.npy')
    N_tot = X.shape[0]
    N_input = X.shape[-1]
    N_state = Y.shape[-1]
    
    # Separate training, validation and test data (70 : 15 : 15)
    N_train = 7*N_tot // 10
    N_val = (N_tot - N_train) // 2
    N_test = N_tot - N_train - N_val
    
    X_train  = X[0:N_train, :, :]            # Training input sequences
    X0_train = Y[0:N_train, 0, :]            # Training input initial conditions
    Y_train  = Y[0:N_train, 1:, :]           # Training output sequences
    
    X_val  = X[N_train:N_train+N_val, :, :]  # Validation input sequences
    X0_val = Y[N_train:N_train+N_val, 0, :]  # Validation input initial conditions
    Y_val  = Y[N_train:N_train+N_val, 1:, :] # Validation output sequences
    
    X_test  = X[N_train+N_val:, :, :]        # Test input sequences
    X0_test = Y[N_train+N_val:, 0, :]        # Test input initial conditions
    Y_test  = Y[N_train+N_val:, 1:, :]       # Test output sequences
    
    # Normalisation (was not necessary in this case)
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train -= mean
    X_train /= std
    X_val -= mean
    X_val /= std
    X_test -= mean
    X_test /= std"""  
    
    # Build DC-RNN model
    N_units = 8                                                # number of hidden units
    x_init = keras.Input(shape=(N_state, ))                    # Initial condition
    input = keras.Input(shape=(None, N_input))                 # input sequence
    model_f1 = convex_RNN(N_units, N_state, N_input)           # input-convex RNN #1 
    model_f2 = convex_RNN(N_units, N_state, N_input)           # input-convex RNN #2
    f1 = model_f1([x_init, input])                             # convex function f1
    f2 = model_f2([x_init, input])
    output = layers.Subtract()([f1, f2])                       # f = f1 - f2
    f_DC = keras.Model(inputs=[x_init, input], outputs=output) # DC-RNN 
    f_DC.summary()
    
    # Compile model
    f_DC.compile(optimizer='rmsprop', loss=['mse', 'mse'], metrics=['mae'])
        
    
    # Load or train model
    if load:  # load existing model
    
        # Restore the weights
        f_DC.load_weights('./model/f_DC').expect_partial()

    else:  # train new model
        
        # Train model
        history = f_DC.fit([X0_train, X_train], Y_train, batch_size=64, epochs=20, 
                       validation_data=([X0_val, X_val], Y_val))
        
        # Save the weights
        f_DC.save_weights('./model/f_DC')
        
        # Plot training and validation loss
        loss_train = history.history['loss']
        loss_val = history.history['val_loss']
        epochs = range(1, len(loss_train)+1)
        fig, ax = plt.subplots()
        ax.plot(epochs, loss_train, 'bo', label='Training loss')
        ax.plot(epochs, loss_val, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        ax.set(xlim=(epochs[0], epochs[-1]), ylim=(0, 20))
        ax.legend()
        #plt.savefig('plot/loss.eps', format='eps')
        plt.show()
    
    # Evaluate model
    test_scores = f_DC.evaluate([X0_val, X_val], Y_val, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test metrics:", test_scores[1]) 
    
    
    # Plot results
    for i in range(10):
        k = random.randint(0, N_test)  # get random entry for the test dataset
        
        # Test data (sampled from dynamical model)
        u = X_test[k, :, :]  # pump voltage (input sequence)
        h = Y_test[k, :, :]  # tank height (output sequence)
        h0 = X0_test[k, :]   # initial tank height (input RNN initial condition)
        steps = range(0, h.shape[0]+1)
        
        # RNN model prediction
        y = f_DC.predict([h0[None, :], u[None, :, :]])
        
        # Plot of the DC decomposition
        fig = plt.figure()
        fig.suptitle("""Coupled tank trajectory prediction for a random voltage 
        input sequence, $x_0 = [{}, {}]$ (cm)""".format(round(h0[0], 1), round(h0[1], 1)))
        fig.tight_layout()
        
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(steps, np.hstack([h0[0], h[:, 0]]), '--b', label='nonlinear model (ref)')
        ax.plot(steps, np.hstack([h0[0], y[0, :, 0]]), '-r', label='DCRNN $f=g-h$')
        ax.set_xlabel('Step [-]')
        ax.set_ylabel('$x_1$')
        ax.legend()
        
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(steps, np.hstack([h0[1], h[:, 1]]), '--b', label='nonlinear model (ref)')
        ax.plot(steps, np.hstack([h0[1], y[0, :, 1]]), '-r', label='DCRNN $f = g-h$')
        ax.set_xlabel('Step [-]')
        ax.set_ylabel('$x_2$')
        ax.legend()
        
        #plt.savefig('plot/RNN{}.eps'.format(i), format='eps')
           
    plt.show()
    
    # Generate a graph of the network
    keras.utils.plot_model(f_DC, "f_DC.png", show_shapes=True)
    
    # Go through the weights from the model
    """print("Weights: ")
    for w in model_f1.get_weights():
        print("new w: ")
        print(w)"""
