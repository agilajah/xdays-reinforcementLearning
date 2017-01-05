Today, we are going to be attempting to solve the FrozenLake environment from the OpenAI gym.

The FrozenLake environment consists of a 4x4 grid of blocks, each one either being the start block, the goal block, a safe frozen block, or a dangerous hole. The objective is to have an agent learn to navigate from the start to the goal without moving onto a hole. At any given time the agent can choose to move either up, down, left, or right. The catch is that there is a wind which occasionally blows the agent onto a space they didn’t choose. As such, perfect performance every time is impossible, but learning to avoid the holes and reach the goal are certainly still doable. The reward at every step is 0, except for entering the goal, which provides a reward of 1. 


Note :

1. In this case, we will need an algorithm that learns long-term expected rewards. So, we use Q-Learning. Because Q-Learning designed to such case.

2. We will use table of values for every state and action possible in the environment. State: row. Action: column. (16x4 table, 16 possible states. 4 possible actions each state).

3. We will update our Q-table using Bellman equation which states that "the expected long-term reward for a given action is equal to the immediate reward from the qurrent action combined with the expected reward from the best future action taken at the following state." equation: `Q(s, a) = r + γ (max(Q (s', a')))`.

4. The equation above says that the Q-value for a given state (s) and action (a) should represent the current reward (r) plus the maximum discounted (γ) future reward expected according to our own table for the next state (s’) we would end up in. The discount variable allows us to decide how important the possible future rewards are compared to the present reward. By updating in this way, the table slowly begins to obtain accurate measures of the expected future reward for a given action in a given state. 

5. If we think of the Q algorithm in the context of gradient descent, then `r + γ(max(Q(s’,a’))` is what we would like to approach, but we know that it is a noisy estimate of the true Q value for that given region. So instead of directly updating toward it, we take a small step in the direction that will make the Q value closer to the desired one. So we will use `Q[s,a] ←Q[s,a] + α(r+ γ * max Q[s',a'] - Q[s,a])` or, equivalently, `Q[s,a] ←(1-α) Q[s,a] + α(r+ γ * max Q[s',a'])`. Where `α` is step-size parameter which influences the rate of learning, or, simply, learning rate, and `γ` is the discount-rate parameter.