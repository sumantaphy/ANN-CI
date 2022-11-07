# Introduction :

Here, a feed-forward artificial neural network (ANN) is used to find the configuration interaction coefficients of a wave function for some model systems. The most important configurations are then found and used to get the ground state energy.

We have found that the error is minimized when the amplitude and sign of the coefficients are separately trained.
# Prerequisites :
1. Python 2
2. PyTorch
3. scikit-learn

# Contributors :
1. Sumanta Kumar Ghosh
2. Debashree Ghosh

# Installation :
        git clone https://github.com/sumantaphy/ANN-CI.git

#How to run this code ?
Modify input arguments in "input.in" file.
python amplitude_signANN.py input.in &

# Input arguments
1. test_fraction                :       FLOAT
                                        Fraction of the total data for testing purposes. So, the (1-test_fraction) part of the data is for training.
2. inp_dim                      :       INT
                                        The number of input nodes. For our case, the number of input nodes is equal to the number of sites.
3. node_amplitude               :       INT
                                        The number of hidden nodes for the sign training.
4. node_sign                    :       INT
                                        The number of hidden nodes for the sign training. For positive (negative) values of coefficients, we have taken 1.0 (0.0).
5. batch_size                   :       INT
                                        Batch size gives the amount of train data in a given batch while training of ANN.
6. num_epochs_amplitude         :       INT
                                        The number of epochs/iterations for amplitude training.
7. num_epochs_sign              :       INT
                                        The number of epochs/iterations for sign training.
8. learn_rate                   :       FLOAT
                                        Learning rate of the model.
9. csv_column                   :       (INT a, INT b, INT c, INT d, INT e)
                                        The range of input, output descriptor and determinant serial number should be given. Where, input ranges from 'a' to 'b' column of a csv file.
                                        'c' and 'e' indicate the output descriptors for the amplitude and sign training, respectively. 'd' is for the determinant column.
10. input_file                  :       STR
                                        Name of input data file in csv format.

# Generated output files

The "model.pth" file is the converged ANN model. One can use this file for future purposes. "output_Amplitude_test.out" file contains the actual and predicted amplitude value in the second and third column, respectively. Similarly, "output_sign_test.out" file contains the actual and predicted amplitude value in the second and third column, respectively.
