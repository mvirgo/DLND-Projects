# Project #3 - Generating a Simpson's TV script

In this project, I built a recurrent neural network (RNN) capable of creating ridiculous Simpson's tv scripts. 

### Summary

#### Dataset

The dataset for this project was a selection of scenes in Moe's Tavern in *The Simpson's*. See [here](tv-script-generation/data/simpsons/moes_tavern_lines.txt).

#### Project Steps

All the work done in this project is in the [dlnd_tv_script_generation](tv-script-generation/dlnd_tv_script_generation.ipynb) jupyter notebook.

1. Pre-process the data by:
    - Making look-up dictionaries where a word is represented by an integer and vice versa, as the vocabulary list is converted to integers.
    - Tokenizing the punctuation within the scripts to make sure the network knows "bye" and "bye!" are the same, regardless of surrounding punctuation.
2. Build the inner workings of the RNN, including determining how many LSTM layers to use (I initally wanted two but found one trained much easier) and utilizing dropout with a wrapper function.
3. Create an embedding layer for use with the inputs before the LSTM layer(s).
4. Build the full RNN, using the embedding, LSTM layer(s), and a fully connected layer for determining the following word based on an input.
5. Making a function to grab batches of training data.
6. Tuning hyperparameters.
7. Training the neural network.
8. Testing it out on a feed word to see whether it can generate a somewhat comprehensible script.
