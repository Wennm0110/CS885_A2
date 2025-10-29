import numpy as np
import MDP

class RL2:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy 
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs: 
        V -- final value function
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.mdp.nStates)
        policy = np.zeros(self.mdp.nStates,int)
        reward, s_next = 0, 0
        T_hat = np.copy(defaultT)
        R_hat = np.copy(initialR)

        N_sa = np.zeros((self.mdp.nActions, self.mdp.nStates))
        N_sas = np.zeros((self.mdp.nActions, self.mdp.nStates, self.mdp.nStates))
        R_sum = np.zeros((self.mdp.nActions, self.mdp.nStates))

        learned_mdp = MDP.MDP(T_hat, R_hat, self.mdp.discount)
        [V, _, _] = learned_mdp.valueIteration(initialV=np.zeros(self.mdp.nStates), tolerance=0.01)
        policy = learned_mdp.extractPolicy(V)

        for episode in range(nEpisodes) :
            s = s0
            for step in range(nSteps) :
                if np.random.rand() < epsilon :
                    action = np.random.randint(self.mdp.nActions)
                else : 
                    action = policy[s]
            
                [reward, s_next] = self.sampleRewardAndNextState(s, action)

                N_sa[action, s] += 1
                N_sas[action, s, s_next] += 1
                R_sum[action, s] += reward

                count = N_sa[action, s]
                T_hat[action, s, :] = N_sas[action, s, :] / count
                R_hat[action, s] = R_sum[action, s] / count

                learned_mdp = MDP.MDP(T_hat, R_hat, self.mdp.discount)

                [V, _, _] = learned_mdp.valueIteration(initialV=V, tolerance=0.01)

                policy = learned_mdp.extractPolicy(V)

                s = s_next



        return [V,policy]    

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor).  Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        empiricalMeans = np.zeros(self.mdp.nActions)
        actiontime = np.zeros(self.mdp.nActions)
        action = np.argmax(empiricalMeans)
        for iterId in range(nIterations) :
            epislon = 1 / (iterId + 1)
            if np.random.rand() < epislon :
                action = np.random.randint(self.mdp.nActions)
            else : 
                action = np.argmax(empiricalMeans)
            
            reward = self.sampleReward(self.mdp.R[action, 0])
            actiontime[action] += 1
            empiricalMeans[action] = ((empiricalMeans[action] * (actiontime[action] - 1) ) + reward) / actiontime[action]
            
        return empiricalMeans

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''
        empiricalMeans = np.zeros(self.mdp.nActions)
        beliefs = np.copy(prior)
        for iter in range(nIterations) :
            sample = np.random.beta(beliefs[:, 0], beliefs[:, 1])
            action = np.argmax(sample)
            reward = self.sampleReward(self.mdp.R[action, 0])
            if reward == 1 :
                beliefs[action, 0] += 1
            else :
                beliefs[action, 1] += 1
        
        empiricalMeans = beliefs[:, 0] / (beliefs[:, 0] + beliefs[:, 1])

        return empiricalMeans

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        nActions = self.mdp.nActions
        empiricalMeans = np.zeros(nActions)
        actionCounts = np.zeros(nActions)

            
        for action in range(nActions):           
            reward = self.sampleReward(self.mdp.R[action, 0])
            actionCounts[action] = 1
            empiricalMeans[action] = reward

        for t in range(nActions, nIterations):
            

            
            bonus_term = np.sqrt((2 * np.log(t)) / actionCounts)
            ucb_scores = empiricalMeans + bonus_term

            action = np.argmax(ucb_scores)
            

            reward = self.sampleReward(self.mdp.R[action, 0])
            

            old_count = actionCounts[action]
            mean_old = empiricalMeans[action]
            
            new_count = old_count + 1
            empiricalMeans[action] = (mean_old * old_count + reward) / new_count
            actionCounts[action] = new_count

        return empiricalMeans