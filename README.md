# Simple_neural_nets
some simple neural net implementations for instructive purposes

The first file, neural nets by hand, makes a neural net with 2 input nodes, a hidden layer with 8 hidden nodes and a bias, and an output
node which outputs <1 if the 2 inputs represent a point inside the unit circle and >1 if it represents a point outside the unit circle.
When running the code, it prints the percent correct, and the data structure that stores the network (a list of dictionaries, where each dictionary
represents a layer of weights)

The second file does the same thing, except it uses numpy arrays. Every print statement is the number of incorrect points categorized
out of the 10000 pairs txt file. 
