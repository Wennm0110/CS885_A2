# CS885 Assignment 2 

## Part I：Bandit & Model-Based RL

### 1.迷宮 (Maze) 問題：Model-Based vs. Q-Learning

* **Model-Based RL :** Model-based RL 是學習環境模型（Transition $T(s, a, s')$ 和 Reward $R(s, a)$），一但有了模型，就可以直接進行planning。

* **Q-learning :** Q-learning 則是model-free的方法，他不會學習環境的 Transition 和 Reward，而是透過大量的試錯 (Trial-and-Error) 從經驗中學習如何估計Q-value。

* 由結果可以觀察出，model-based RL能有較快的收斂速度，但只要給Q-learning訓練次數，他也能夠學到接近最佳的策略。
![Model-Based vs Q-learning](/images/maze_comparison.png)

### 2.Bandit
* **epsilon-greedy :** 它的探索是隨機的，它會以 $\epsilon$ 的機率隨機選一個arm。他在最初時 $\epsilon=1$ ，也就是一定會隨機選取一個arm去拉下，而在一定回合後， $\epsilon$ 會逐漸下降，而agent會以 $\epislon$ 的機率去探索，並以 $1-\epsilon$ 的機率去選取目前有最高期望值的arm去拉下。
* **Thompson Sampling :**
* **UCB (Upper Confidence Bound) :**
![Bandit](/images/bandit_comparison.png)

## Part II：REINFORCE (w/o baseline) & PPO
