# hybrid_LDA metric learning package

## Description 

Implementation of hybrid metric learning algorithm described in:

```bibtex
B. Bue and E. Merényi, "An Adaptive Similarity Measure for
Classification of Hyperspectral Signatures," IEEE Geoscience and
Remote Sensing Letters, 2012.
```

Please cite the above reference if you publish work that uses this code.

## Tested on: 
- OSX 10.6 
- Python 2.6, numpy 1.6, scipy 0.9
- Matlab R2011a


## Example usage 

To run the Python demo for several different lambda values:

```python
   lambdas = [1e-5,1e-3,0,0.1,0.5,0.9]
   minDistHybrid([dat_ci,dat_cr],labels,lambdas,Dlab=["CI","CR"])
```
where: 
- dat_ci and dat_cr: [N x d] arrays containing a set of N
  continuum-intact spectral signatures and their corresponding
  continuum-removed representations, respectively; 
- labels: N-dimensional vector of class labels for each signature; 
- lambdas: regularization values, each in the [0,1] range.

Functions to calculate the continuum-removed representation of a
spectral signature are provided in the LINCR library available at:

https://github.com/dsmbgu8/LINCR

or

http://www.ece.rice.edu/~bdb1/#code.


## Contact 

Please contact the author (bbue@alumni.rice.edu) if you have any questions
regarding this code.
