# nHDP

This repository is a modification of the [MATLAB nHDP code by John
Paisley](http://www.columbia.edu/~jwp2128/code/nHDP.zip), an implementation
of the model introduced in ["Nested Hierarchical Dirichlet Processes" by Paisley,
Wang, Blei, and Jordan (2014)](https://ieeexplore.ieee.org/document/6802355)
([arxiv preprint](https://arxiv.org/abs/1210.6738)).  Changes include:

* Using a sparse matrix format for the corpus
* Creating a function for the training loop
* Further parametrizing the code for flexibility/reusability
* Comments and other readability improvements

Use the following command to see all changes: `git diff 0d196e6`

Alternatively, use [GitHub's `compare` tool](https://github.com/ccmaymay/nhdp/compare/0d196e6...main).

## Notes

* Note K-means initialization is EM where E-step (document assignment to clusters) is L1 minimization and M-step (reestimation of cluster centers) is L2 minimization
* K-means initialization is tweaked such that clusters are returned in descending order of size
* In init, both beta ss and V ss are scaled by a specified constant "scale" (100*D/K by default)
* In init, beta ss is a probability vector (before scaling)
* In init, V ss is set to the number of documents in the subtree rooted at the node, divided by the total number of documents in the initialization set (before scaling)
* In first iteration of e-step, prior is ignored (just theta ss are considered)
* Based on func_process_tree, it does seem nodes are reordered, and by subtree
* Based on func_doc_weight_up, the local subtrees are also reordered, by subtree
* Noisy global param update is scaled by "scale" e.g. 100D/K, also used in init... makes update consistent with init, but weird...
* Global param update includes a uniform term: rho/10 of the batch estimate is unif and the rest is from data (this is then scaled by rho and added to (1 - rho) times the existing global estimate)
