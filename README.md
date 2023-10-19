# UPbeat

UPbeat is a user-friendly MATLAB GUI that estimates continuous
time-temperature paths from U-Pb depth profiles of accessory minerals
such as rutile and apatite. The program uses a Markov Chain Monte
Carlo (MCMC) algorithm based on Aslak Grindsted's implementation of
the Goodman and Weare (2010) ensemble sampler (a.k.a. the "MCMC
Hammer"; Foreman-Mackey et al., 2013).

The repo contains the following files and folders:


- `UPbeat.m` and `UPbeat.fig`: the Matlab GUI. To run this code, simply enter `UPbeat` at the MATLAB command prompt.

- `data1`: a directory with a first example dataset, including three
  kinetic data files (in an `.xls` format) and the corresponding three
  depth profiles (in a `.txt` format).

- `data2` - a directory with a second example dataset of four depth
  profiles.