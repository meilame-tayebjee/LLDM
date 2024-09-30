## Implementation of "Longitudinal Latent Diffusion Models"


<p align="center">
    <a>
	    <img src='lib/plots/to_gif/sprites/case_20/movie.gif' width="60" 
     height="60"/>
	</a>
    <a>
	    <img src='lib/plots/to_gif/sprites/case_0/movie.gif' width="60" 
     height="60"/>
	</a>
    <a>
	    <img src='lib/plots/to_gif/sprites/case_23/movie.gif' width="60" 
     height="60"/>
	</a>
     <a>
	    <img src='lib/plots/to_gif/starmen/case_8/movie.gif' width="60" 
     height="60"/>
	</a>
    <a>
	    <img src='lib/plots/to_gif/starmen/case_12/movie.gif' width="60" 
     height="60"/>
	</a>
    <a>
	    <img src='lib/plots/to_gif/starmen/case_49/movie.gif' width="60" 
     height="60"/>
    </a>
</p>
<p align="center">
  <b>Generated sequences</b>
</p>
	

**Disclaimer**: 
- The code in `lib` is an adaptation from [1]. In particular, we adapted their implementation of the IAF flows, the VAE and VAMP models and their trainer to plug our method.

[1] Chadebec, C., Vincent, L. J., and Allassonni `ere, S. Pythae:
Unifying generative autoencoders in python–a bench-
marking use case. Proceedings of the Neural Information
Processing Systems Track on Datasets and Benchmarks,
2022


## Setup

First create a virtual env and activate it 

```bash
conda create -n env python=3.8
conda activate env
```
