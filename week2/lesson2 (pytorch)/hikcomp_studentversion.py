import os,sys,numpy as np

import torch

import time

def forloopdists(feats1,feats2):

  #your code here

  return dist

def numpydists(feats1,feats2):

  #your code here

  return dist
  
def pytorchdists(feats1,feats2,device):
  
  #your code here

  return dist.cpu().numpy()


def run():

  ########
  ##
  ## if you have less than 8 gbyte, then reduce from 250k
  ##
  ###############

  numdata1=2500
  numdata2=500
  dims=30

  # genarate some random histogram data
  feats1=np.random.normal(size=(numdata1,dims))**2
  feats2=np.random.normal(size=(numdata2,dims))**2

  feats1=feats1/np.sum(feats1,axis=1)[:,np.newaxis]
  feats2=feats2/np.sum(feats2,axis=1)[:,np.newaxis]

  
  since = time.time()
  dists0=forloopdists(feats1,feats2)
  time_elapsed=float(time.time()) - float(since)
  print('for loop, Comp complete in {:.3f}s'.format( time_elapsed ))
  



  device=torch.device('cpu')
  since = time.time()

  dists1=pytorchdists(feats1,feats2,device)


  time_elapsed=float(time.time()) - float(since)

  print('pytorch, Comp complete in {:.3f}s'.format( time_elapsed ))
  print(dists1.shape)

  #print('df0',np.max(np.abs(dists1-dists0)))


  since = time.time()

  dists2=numpydists(feats1,feats2)


  time_elapsed=float(time.time()) - float(since)

  print('numpy, Comp complete in {:.3f}s'.format( time_elapsed ))

  print(dists2.shape)

  print('df',np.max(np.abs(dists1-dists2)))


if __name__=='__main__':
  run()
