import matplotlib.pyplot as plt
import copy
import numpy as np
import random
import matplotlib
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

class simpleprob1():
  # all actions into one single state, the keystate, give a high reward

  def __init__(self,numh,numw, keystate):
  
    self.numh=numh
    self.numw=numw

    if (keystate[0]<0) or (keystate[0]>=self.numh):
      print('illegal')
      exit()
    if (keystate[1]<0) or (keystate[1]>=self.numw):
      print('illegal')
      exit()

    #state space: set of tuples (h,w) 0<=h<=numh, 0<=w<=numw
    self.statespace=[ (h,w) for h in range(self.numh) for w in range(self.numw) ]
 
    self.statespace2index=dict()
    for i,s in enumerate(self.statespace):
      self.statespace2index[s]=i



    self.actions=['stay','left','down','right','up']
    self.actdict2index=dict()
    for i,a in enumerate(self.actions):
      self.actdict2index[a]=i


    self.highrewardstate=keystate
    self.rewardtogothere=10.

    #only for RL
    #self.state=[np.random.randint(0,self.numh),np.random.randint(0,self.numw)]
    self.reset()

  def transition_deterministic(self,oldstate_index,action):
    #P(s'|s,a) is 1 for one specific s'

    if action not in self.actions:
      print('illegal')
      exit()


    oldstate=self.statespace[oldstate_index]
    
    # all deterministic

    if self.actdict2index[action]==0:
      newstate=list(oldstate)

    elif self.actdict2index[action]==1:
      newstate=list(oldstate)
      newstate[1]=min(self.numw-1,newstate[1]+1)


    elif self.actdict2index[action]==2:
      newstate=list(oldstate)
      newstate[0]=min(self.numh-1,newstate[0]+1)


    elif self.actdict2index[action]==3:
      newstate=list(oldstate)
      newstate[1]=max(0,newstate[1]-1)


    elif self.actdict2index[action]==4:
      newstate=list(oldstate)
      newstate[0]=max(0,newstate[0]-1)

    #can return probs or set of new states and probabilities

    done=False # can play forever
    return self.statespace2index[tuple(newstate)]
  

  def reward(self,oldstate_index,action,newstate_index):
    #P(R|s,a)
    onlygoalcounts=True

    if False==onlygoalcounts: #one gets  a reward when one jumps into the golden state or stays there
      r=self.tmpreward1(oldstate_index, action, newstate_index)
    else: #one gets only a reward when one stays in the golden state
      r=self.tmpreward2(oldstate_index, action, newstate_index) 

    return r
  
  def tmpreward1(self,oldstate_index,action,newstate_index):

    newstate=self.statespace[newstate_index]
    if (newstate[0]==self.highrewardstate[0]) and (newstate[1]==self.highrewardstate[1]):
      return self.rewardtogothere
    else:
      return 0

  def tmpreward2(self,oldstate_index,action,newstate_index):

    newstate=self.statespace[newstate_index]
    if (newstate[0]==self.highrewardstate[0]) and (newstate[1]==self.highrewardstate[1]) and (action=='stay'):
      return self.rewardtogothere
    else:
      return 0



  ##################################
  # for RL
  #####################################
  def reset(self):
    #randomly set start point
    self.state=np.random.randint(0,len(self.statespace))
    return self.state

  def getstate(self):
    return self.state

  def step(self,action):
    #print(self.state,action)
    done=False
    tmpstateind=self.transition_deterministic(self.state,action)
    reward=self.reward(self.state, action, tmpstateind)

    self.state=tmpstateind

    return self.state, reward, done



def plotqvalstable(qvals, simpleprob_instance, block):
  # input is numpy of shape (5,h,w)  
  #plotted into 3x3 + boundary  qvals[c,h,w] c=center,l,d,r,up
  plt.ion()

  offsets=[ [1,1],[1,2],[2,1],[1,0],[0,1] ]  
  symbols=[ 'o','->','\ ','<-','^' ]  

  mh=simpleprob_instance.numh
  mw=simpleprob_instance.numw

  plotvals=-np.ones((3*mh,3*mw))

  for i in range(len(simpleprob_instance.statespace)):
    h=simpleprob_instance.statespace[i][0]
    w=simpleprob_instance.statespace[i][1]

    for c in range( len(simpleprob_instance.actions)):
        plotvals[3*h + offsets[c][0] ,3*w+ offsets[c][1]]=qvals[ i,c]

  plotvals = np.ma.masked_where(plotvals<0,plotvals)

  fig, (ax0) = plt.subplots(1, 1, figsize = (20,20))

  ax0.imshow(plotvals, cmap=plt.get_cmap('summer'),interpolation='nearest')



  #c = ax0.pcolor(plotvals, edgecolors='white', linewidths=1)
  ax0.patch.set(hatch='xx', edgecolor='red')

  for i in range(len(simpleprob_instance.statespace)):
    h=simpleprob_instance.statespace[i][0]
    w=simpleprob_instance.statespace[i][1]

    for c in range( len(simpleprob_instance.actions)):
      if c==0:
        printstr= "{:.2f}".format(qvals[ i,c]) #str(qvals[c,h,w])
      elif c==1:
        printstr="{:.2f}".format(qvals[ i,c])+symbols[c]
      elif c==2:
        printstr=symbols[c]+"{:.2f}".format(qvals[ i,c])
      elif c==3:
        printstr=symbols[c]+ "{:.2f}".format(qvals[ i,c])
      elif c==4:
        printstr=symbols[c]+"{:.2f}".format(qvals[ i,c])
      
              
      ax0.text( 3*w+ offsets[c][1], 3*h + offsets[c][0],printstr,
                     ha="center", va="center", color="k")

  plt.savefig("Plot trainsth")
  plt.draw()
  plt.pause(0.01)

  if True==block:
    input("Press [enter] to continue.")




def plotonlyvalstable2(qvals, simpleprob_instance,  block):
  # input is numpy of shape (5,h,w)  
  #plotted into 3x3 + boundary  qvals[c,h,w] c=center,l,d,r,up
  plt.ion()

  mh=simpleprob_instance.numh
  mw=simpleprob_instance.numw


  plotvals=-np.ones((mh,mw))
  for i in range(len(simpleprob_instance.statespace)):
    h=simpleprob_instance.statespace[i][0]
    w=simpleprob_instance.statespace[i][1]
    for c in range( len(simpleprob_instance.actions)):
      plotvals[h,w]=np.max(qvals[ i,:])

  #print(qvals)
  plotvals = np.ma.masked_where(plotvals<0,plotvals)

  fig, (ax0) = plt.subplots(1, 1)

  ax0.imshow(plotvals, cmap=plt.get_cmap('summer'),interpolation='nearest')

  #c = ax0.pcolor(plotvals, edgecolors='white', linewidths=1)
  #ax0.patch.set(hatch='xx', edgecolor='black', color='red')

  for h in range(mh):
    for w in range(mw):
      printstr= "{:.2f}".format(plotvals[h,w])   
      ax0.text( w, h ,printstr,ha="center", va="center", color="k")

  plt.draw()
  plt.pause(0.01)
  if True==block:
    #pass
    input("Press [enter] to continue.")



def plotonlyvalstable2b(qvals, simpleprob_instance,  block):
  # input is numpy of shape (5,h,w)  
  #plotted into 3x3 + boundary  qvals[c,h,w] c=center,l,d,r,up
  plt.ion()

  mh=simpleprob_instance.numh
  mw=simpleprob_instance.numw


  plotvals=-np.ones((mh,mw))
  for i in range(len(simpleprob_instance.statespace)):
    h=simpleprob_instance.statespace[i][0]
    w=simpleprob_instance.statespace[i][1]
    for c in range( len(simpleprob_instance.actions)):
      plotvals[h,w]=np.max(qvals[ i,:])

  #print(qvals)
  plotvals = np.ma.masked_where(plotvals<0,plotvals)

  fig= plt.figure(1)
  plt.clf()
  #fig, (ax0) = plt.subplots(1, 1)
  ax0=plt.axes()
  ax0.imshow(plotvals, cmap=plt.get_cmap('summer'),interpolation='nearest')

  #c = ax0.pcolor(plotvals, edgecolors='white', linewidths=1)
  #ax0.patch.set(hatch='xx', edgecolor='black', color='red')

  for h in range(mh):
    for w in range(mw):
      printstr= "{:.2f}".format(plotvals[h,w])   
      ax0.text( w, h ,printstr,ha="center", va="center", color="k")


  plt.draw()
  plt.pause(0.001)
  if True==block:
    #pass
    input("Press [enter] to continue.")


def plotmoves(statesseq, simpleprob_instance,  block):
  fig= plt.figure(5)
  ax0=plt.axes()

  plt.clf()
  mh=simpleprob_instance.numh
  mw=simpleprob_instance.numw

  for s in statesseq:

    #plt.clf()

    h=simpleprob_instance.statespace[s][0]
    w=simpleprob_instance.statespace[s][1]
    plotvals=np.zeros((mh,mw))
    plotvals[h,w]=1
    print(h,w)
    ax0.imshow(plotvals, cmap=plt.get_cmap('summer'),interpolation='nearest')

    plt.draw()

    plt.pause(0.05)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def valueiter_mdp_q2(problemclass,gamma,  delta , showeveryiteration):
    l = len(problemclass.statespace)
    
    qsa = np.zeros((l, len(problemclass.actions)))
    new_qsa = np.zeros((l, len(problemclass.actions)))
    diff = 1
    while diff > delta:
    
        for state_index in range(l):          
            action_values = []
            for action_index in range(len(problemclass.actions)):               
                action = problemclass.actions[action_index]
                new_state_index = problemclass.transition_deterministic(state_index, action)
                state_action_reward = problemclass.reward(state_index,action,new_state_index)
                new_qsa[state_index, action_index] = state_action_reward + gamma*(max(qsa[new_state_index]))            
        diff = np.sum(np.square(qsa-new_qsa))
        qsa = copy.deepcopy(new_qsa) 
    print("qsa: ", qsa)  
    
    values = np.zeros(l)
    new_values = np.zeros(l)
    diff = 1
    
    while diff > delta:
        for state_index in range(l):
            action_values = []
            for action in problemclass.actions:               
                new_state_index = problemclass.transition_deterministic(state_index, action)
                state_action_reward = problemclass.reward(state_index,action,new_state_index)
                action_values.append(state_action_reward + gamma*(values[new_state_index]))
            new_values[state_index] = max(action_values)
        diff = np.sum(np.square(values-new_values))
        values = copy.deepcopy(new_values)
    print("Values: ", values)  
    
    return values, qsa




def plot_rewards2(episode_rewards,means100):
    plt.figure(3)
    plt.clf()

    plt.title('training or testing...')
    plt.xlabel('Episode')
    plt.ylabel('averaged reward')
    plt.plot( np.asarray( episode_rewards ))
    # Take 100 episode averages and plot them too
    if len(episode_rewards) >= 100:
        #means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        #means = torch.cat((torch.zeros(99), means))
        #plt.plot(means.numpy())
        mn=np.mean(episode_rewards[-100:])
    else:
        mn=np.mean(episode_rewards)
    means100.append(mn)
    plt.plot(means100)
        #print('100 mean:')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

class agent_Qlearn_thattable( ):
  def __init__(self, simpleprob_instance):

    self.gamma=0.9
    self.softupdate_alpha=0.2

    self.delta=1e-3 # Q-convergence

    self.numactions=len(simpleprob_instance.actions)
    self.statespace= [i for i in range(len(simpleprob_instance.statespace))]
    self.Q=np.zeros(( len(self.statespace) , self.numactions  ))

    #how should the eps for exploration vs exploitation decay?
    self.epsforgreediness_start=0.9
    self.epsforgreediness_end=0.01
    self.epsforgreediness_maxdecaytime=100



  def currenteps(self,episode_index):
    
    if episode_index<0:
      v=self.epsforgreediness_end
    else:
      v=self.epsforgreediness_end + (self.epsforgreediness_start-self.epsforgreediness_end)* max(0,self.epsforgreediness_maxdecaytime-episode_index)/ float(self.epsforgreediness_maxdecaytime)

    return v

  def actionfromQ(self,state_index,episode_index):
    #episode_index for decay
    eps=self.currenteps(episode_index)

    #
    # ME
    #
    
    rand = random.random()
    if rand > eps:
        action = np.argmax(self.Q[state_index])
    else:
        #pick random action
        action = np.random.randint(0, len(self.Q[state_index]))

    return action
    
  def train(self, simpleprob_instance):

    numepisodes=120 #250
    maxstepsperepisode=100
    

    episode_rewards=[]
    means100=[]

    for ep in range(numepisodes):

      #
      # ME
      #

      step = 0
      ep_reward = 0
      state_index = simpleprob_instance.reset()  #S
      
      for step in range(maxstepsperepisode):

          action = simpleprob_instance.actions[self.actionfromQ(state_index,ep)]  #A
          new_state_index = simpleprob_instance.transition_deterministic(state_index,action)
          reward = simpleprob_instance.reward(state_index, action, new_state_index)  #R
          action_index = self.actionfromQ(state_index,ep)
          added_term = self.softupdate_alpha*(reward + self.gamma*np.max(self.Q[new_state_index]) - self.Q[state_index, action_index])
          self.Q[state_index, action_index] = self.Q[state_index, action_index] + added_term
          state_index = new_state_index  #S'

          ep_reward += reward
                
      # outside of playing one episode
      avgreward = ep_reward/maxstepsperepisode
      episode_rewards.append(avgreward)
      plot_rewards2(episode_rewards,means100)

      print('episode',ep,'averaged reward',avgreward)

      if ep%10==0:
        plotonlyvalstable2b(self.Q, simpleprob_instance,  block=False)

    plotqvalstable(self.Q, simpleprob_instance, False)

  def runagent(self, simpleprob_instance):
    maxstepsperepisode=20

    state_index=simpleprob_instance.reset()

    episode_rewards=[]
    means100=[]
    statesseq=[state_index]

    avgreward=0
    for playstep in range(maxstepsperepisode):

        #
        # ME
        #

        action = simpleprob_instance.actions[np.argmax(self.Q[state_index])]
        next_state, reward, done = simpleprob_instance.step(action)
        avgreward += reward
        statesseq.append(next_state)

        state_index = next_state

    avgreward = avgreward/maxstepsperepisode
    # outside of playing one episode
    episode_rewards.append(avgreward)
    #plot_rewards2(episode_rewards,means100)
    print('post training run averaged reward',avgreward)
    plotmoves(statesseq, simpleprob_instance,  block=False)

def runmdp():

  plotbig=True
  showeveryiteration=True

  mdp=simpleprob1(5,6,keystate=[1,4])
  values,qsa=valueiter_mdp_q2(mdp,gamma=0.5, delta=3e-2, showeveryiteration=showeveryiteration)

  
  if False==plotbig:
    if False==showeveryiteration:
      plotonlyvalstable(qsa,  mdp, block=False)
  else:
    plotqvalstable(qsa, mdp,block=False)

  for i in range(3):
    print('FINISHED')
  input("Press [enter] to continue.")


def trainsth():

  plotbig=True
  showeveryiteration=True

  problem=simpleprob1(5,6,keystate=[1,4])

  ag=agent_Qlearn_thattable(problem)
  ag.train(problem)

  '''
  if False==plotbig:
    if False==showeveryiteration:
      plotonlyvalstable(ag.Q,  problem, block=False)
  else:
    plotqvalstable(ag.Q, problem,block=False)
  '''

  for i in range(3):
    print('FINISHED')

  ag.runagent( problem)

  for i in range(3):
    print('FINISHED')
  input("Press [enter] to continue.")


if __name__=='__main__':
  #tester()
  #runmdp()
  trainsth()