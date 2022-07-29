# GIB

This repo contains a rewrite of the original GIB source code. The rewrite intents to update the existing code to the latest Pytorch and Pytorch Geometric versions. Moreover, we attempt to integrate the training with Pytorch Lightining in order to provide a cleaner API and an easier way of logging metrics.

At the moment we have implemented the following:

* GCN: only diag reparametrization and Gaussian prior.
* GAT: only diag reparametrization and Gaussian prior.
* training: no adversarial attack (yet).


Author:
Victor Faraggi, @stepp1