# Convolution-Music-AI
Using Convolutional Neural Network to Generate Music:

Generated music samples are in the Music folder.

experiments.ipynb is the jupyter notebook that was used while designing some of the algorithms / network architecture.

utils.py contains the algorithms for generating piano rolls from musicXML files as well as for generating MIDI files from piano rolls.

ConvNetwork.py contains the code for the network (loss functions etc.)

evaluate.py contains the code for training the network and for producing outputs based on using Gibbs Sampling on random samples of Bach using block erases of the score:  To train, run evaluate.py.

To load a model, $python3 evaluate.py -load path-to-model-directory (save files as model.json and model.h5).
