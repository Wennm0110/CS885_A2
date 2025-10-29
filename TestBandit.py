import numpy as np
import MDP
import RL2
import matplotlib.pyplot as plt

def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0


# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.3],[0.5],[0.7]])
discount = 0.999
mdp = MDP.MDP(T,R,discount)
banditProblem = RL2.RL2(mdp,sampleBernoulli)


nTrials = 1000
nIterations = 200

eps_greedy_rewards = np.zeros([nTrials, nIterations])
thompson_rewards = np.zeros([nTrials, nIterations])
ucb_rewards = np.zeros([nTrials, nIterations])

print("Running Bandit Experiments...")
for trial in range(nTrials):
    if (trial + 1) % 100 == 0: 
        print(f"  Bandit Trial {trial + 1}/{nTrials}")
    
    # 1. Epsilon-Greedy
    [_, rewards_eps] = banditProblem.epsilonGreedyBandit(nIterations=nIterations)
    eps_greedy_rewards[trial, :] = rewards_eps
    
    # 2. Thompson Sampling
    prior = np.ones([mdp.nActions, 2]) 
    [_, rewards_thomp] = banditProblem.thompsonSamplingBandit(prior=prior, nIterations=nIterations)
    thompson_rewards[trial, :] = rewards_thomp
    
    # 3. UCB
    [_, rewards_ucb] = banditProblem.UCBbandit(nIterations=nIterations)
    ucb_rewards[trial, :] = rewards_ucb


avg_eps_greedy_rewards = np.mean(eps_greedy_rewards, axis=0)
avg_thompson_rewards = np.mean(thompson_rewards, axis=0)
avg_ucb_rewards = np.mean(ucb_rewards, axis=0)


plt.figure(figsize=(10, 6))
plt.plot(avg_eps_greedy_rewards, label='Epsilon-Greedy (1/t)')
plt.plot(avg_thompson_rewards, label='Thompson Sampling (Beta(1,1))')
plt.plot(avg_ucb_rewards, label='UCB')
plt.xlabel('Iteration #')
plt.ylabel('Average Reward Earned')
plt.title('Bandit Algorithm Comparison (1000 Trials)')
plt.legend()
plt.grid(True)
plt.savefig('images/bandit_comparison.png') 


# # Test epsilon greedy strategy
# empiricalMeans = banditProblem.epsilonGreedyBandit(nIterations=200)
# print("\nepsilonGreedyBandit results")
# print(empiricalMeans)

# # Test Thompson sampling strategy
# empiricalMeans = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=200)
# print("\nthompsonSamplingBandit results")
# print(empiricalMeans)

# # Test UCB strategy
# empiricalMeans = banditProblem.UCBbandit(nIterations=200)
# print("\nUCBbandit results")
# print(empiricalMeans)
