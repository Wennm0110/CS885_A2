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
        rewards_per_episode = np.zeros(nEpisodes)
        for episode in range(nEpisodes) :
            s = s0
            cumulative_discounted_reward = 0.0
            for step in range(nSteps) :
                if np.random.rand() < epsilon :
                    action = np.random.randint(self.mdp.nActions)
                else : 
                    action = policy[s]
            
                [reward, s_next] = self.sampleRewardAndNextState(s, action)
                cumulative_discounted_reward += (self.mdp.discount ** step) * reward
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
            rewards_per_episode[episode] = cumulative_discounted_reward



        return [V, policy, rewards_per_episode]  

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor).  Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''
        rewards_history = np.zeros(nIterations)
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
            rewards_history[iterId] = reward
            empiricalMeans[action] = ((empiricalMeans[action] * (actiontime[action] - 1) ) + reward) / actiontime[action]
            
        return empiricalMeans,rewards_history

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''
        rewards_history = np.zeros(nIterations)
        empiricalMeans = np.zeros(self.mdp.nActions)
        beliefs = np.copy(prior)
        for iter in range(nIterations) :
            sample = np.random.beta(beliefs[:, 0], beliefs[:, 1])
            action = np.argmax(sample)
            reward = self.sampleReward(self.mdp.R[action, 0])
            rewards_history[iter] = reward
            if reward == 1 :
                beliefs[action, 0] += 1
            else :
                beliefs[action, 1] += 1
        
        empiricalMeans = beliefs[:, 0] / (beliefs[:, 0] + beliefs[:, 1])

        return empiricalMeans,rewards_history

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''
        rewards_history = np.zeros(nIterations)
        nActions = self.mdp.nActions
        empiricalMeans = np.zeros(nActions)
        actionCounts = np.zeros(nActions)

            
        for action in range(nActions):           
            reward = self.sampleReward(self.mdp.R[action, 0])
            actionCounts[action] = 1
            empiricalMeans[action] = reward
            rewards_history[action] = reward

        for t in range(nActions, nIterations):
            

            
            bonus_term = np.sqrt((2 * np.log(t)) / actionCounts)
            ucb_scores = empiricalMeans + bonus_term

            action = np.argmax(ucb_scores)
            

            reward = self.sampleReward(self.mdp.R[action, 0])
            

            old_count = actionCounts[action]
            mean_old = empiricalMeans[action]
            rewards_history[t] = reward
            new_count = old_count + 1
            empiricalMeans[action] = (mean_old * old_count + reward) / new_count
            actionCounts[action] = new_count

        return empiricalMeans, rewards_history
    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        rewards_per_episode -- A list containing the cumulative discounted reward for each episode
        '''

        Q = np.copy(initialQ)
        N = np.zeros([self.mdp.nActions, self.mdp.nStates])
        
        # 新增：用於儲存每個 episode 的獎勵
        rewards_per_episode = []

        for episode in range(nEpisodes):
            state = s0
            # 新增：用於計算當前 episode 的累積折扣獎勵
            cumulative_discounted_reward = 0.0

            for step in range(nSteps):
                # 1. 選擇動作 (ε-greedy)
                if np.random.rand() < epsilon:
                    action = np.random.randint(self.mdp.nActions)
                else:
                    if temperature > 0:
                        q_values = Q[:, state]
                        q_values_stable = q_values - np.max(q_values)
                        probabilities = np.exp(q_values_stable / temperature)
                        probabilities /= np.sum(probabilities)
                        action = np.random.choice(self.mdp.nActions, p=probabilities)
                    else:
                        action = np.argmax(Q[:, state])

                # 2. 執行動作並觀察結果
                [reward, nextState] = self.sampleRewardAndNextState(state, action)
                
                cumulative_discounted_reward += (self.mdp.discount ** step) * reward

                # 3. 更新 Q-value
                N[action, state] += 1
                alpha = 1.0 / N[action, state]

                max_q_next_state = np.max(Q[:, nextState])
                
                td_target = reward + self.mdp.discount * max_q_next_state
                Q[action, state] += alpha * (td_target - Q[action, state])

                # 4. 更新狀態
                state = nextState
            
            # 將這個 episode 的總獎勵記錄下來
            rewards_per_episode.append(cumulative_discounted_reward)

        # 導出最終策略
        policy = np.argmax(Q, axis=0)

        # 回傳 Q 表、策略以及每個 episode 的獎勵歷史
        return [Q, policy, rewards_per_episode]