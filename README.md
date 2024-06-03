# Siamese Network for One-Shot Learning for Traffic Sign Recognition
(this repository is forked)

Siamese networks, a unique subset of neural network architectures, are designed to facilitate the comparison of inputs to determine their similarity. The defining characteristic of these networks is their 'Siamese' twin structure, where two identical sub-networks come together at their outputs. This architecture is particularly suited for tasks that require comparison of pairs of inputs, such as face recognition, signature verification, and prescription pill identification.

The basic architecture of a Siamese network consists of two inputs and two branches, often referred to as "sister networks." Each of these sister networks is identical to the other, sharing the same architecture and parameters. This mirrored structure ensures that if the weights in one sub-network are updated, the weights in the other sub-network are updated as well, maintaining consistency across the network.

The outputs of the two sub networks are combined, and then the final similarity score output is returned. This score is typically computed by measuring the Euclidean distance between the outputs of the two sub networks and feeding them through a sigmoid activation function. The sigmoid activation function values closer to "1" imply more similarity, while values closer to "0" indicate less similarity.

TensorFlow of version 2.15.0 and Python 3.11.9 along with multiple addons were used for our implementation. Functions from Pllow (PIL) library\footnote{\url{https://python-pillow.org}} were used specifically for augmentations to manipulate image data. We use standard versions of SciPy, pandas, seaborn and TensorBoard for plotting results.
