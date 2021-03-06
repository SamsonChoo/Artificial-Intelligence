import os,sys,numpy as np

import torch

import time

def forloopdists(feats1,feats2):
    dist=np.empty((len(feats1),len(feats2)))
    for i,feat1 in enumerate(feats1):
        for j,feat2 in enumerate(feats2):
            mindtot=0    #total of minimum
            for d in range(30):
                mind = min(feat1[d],feat2[d])    #minimum between comparison of each d
                mindtot += mind
            dist[i][j]=mindtot
    return dist



def numpydists(feats1,feats2):

    D = len(feats1)
    L = len(feats2)
    f1 = feats1.reshape(D,1,30)
    f2 = feats2.reshape(1,L,30)
    dist = np.sum(np.minimum(f1,f2), axis = 2)

    return dist
  
def pytorchdists(feats1,feats2,device):

    f1 = torch.unsqueeze(torch.tensor(feats1),1)
    f2 = torch.unsqueeze(torch.tensor(feats2),0)
    dist=torch.sum((torch.min(f1,f2)),2)
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

  print('df0',np.max(np.abs(dists1-dists0)))


  since = time.time()

  dists2=numpydists(feats1,feats2)


  time_elapsed=float(time.time()) - float(since)

  print('numpy, Comp complete in {:.3f}s'.format( time_elapsed ))

  print(dists2.shape)

  print('df',np.max(np.abs(dists1-dists2)))


if __name__=='__main__':
  run()
