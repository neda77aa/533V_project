import gym
import math
import random
from itertools import count
import torch
from eval_policy import eval_policy, device
from model import MyModel
from replay_buffer import ReplayBuffer
from torch.nn import MSELoss
import numpy as np


BATCH_SIZE = 256
# False: Without replay
# True: With replay
Replay = True
GAMMA = 0.99
EPS_EXPLORATION = 0.2
TARGET_UPDATE = 5
NUM_EPISODES = 2000
TEST_INTERVAL = 25
LEARNING_RATE = 10e-3
RENDER_INTERVAL = 20
ENV_NAME = 'CartPole-v0'
PRINT_INTERVAL = 10
env = gym.make(ENV_NAME)
state_shape = len(env.reset())
n_actions = env.action_space.n
model = MyModel(state_shape, n_actions).to(device)
target = MyModel(state_shape, n_actions).to(device)
target.load_state_dict(model.state_dict())
target.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = MSELoss()
rb = ReplayBuffer()
score_plot = []
def choose_action(state,step ,test_mode=False):
    # TODO implement an epsilon-greedy strategy
    epsilon = 0.05
    x = random.uniform(0,1)
    if x<epsilon:
        qval = model(torch.tensor(state).to(device)).cpu().detach().numpy()
        prob_action = (qval-np.max(qval))/np.sum((qval-np.max(qval)))
        action = torch.tensor(int(np.random.choice(prob_action,p=prob_action)))
        return action.view(1, 1)
    else:
        return torch.argmax(model(torch.tensor(state).to(device))).view(1, 1)


def optimize_model(state, action, next_state, reward, done):
    # TODO given a tuple (s_t, a_t, s_{t+1}, r_t, done_t) update your model weights
    (values, indices) = torch.max(target(next_state.detach().clone()).view(-1,2),dim=1)
    y = reward + (torch.logical_not(done)) * GAMMA* values
    y_pred = torch.gather(model(state.detach().clone()).view(-1,2), dim=1, index=action.type(torch.LongTensor).squeeze().to(device).view(-1,1)).squeeze()
    loss = loss_function(y_pred , y.squeeze())/BATCH_SIZE
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_reinforcement_learning(render=False):
    steps_done = 0
    best_score = -float("inf")
    for i_episode in range(1, NUM_EPISODES+1):
        episode_total_reward = 0
        state = env.reset()
        for t in count():
            action = choose_action(state,steps_done+1)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0][0])
            steps_done += 1
            episode_total_reward += reward
            if Replay:
                rb.push(state, action, next_state, reward, done) 
                if steps_done>=BATCH_SIZE:
                    state1, action1, next_state1, reward1, done1 = rb.sample(batch_size=BATCH_SIZE)
                    optimize_model(state1, action1, next_state1, reward1, done1)
            else: 
                state_t = torch.tensor(state, device=device)
                next_state_t = torch.tensor(next_state, device=device, dtype=torch.float32)
                action_t = action.detach().clone()
                reward_t = torch.tensor(reward, device=device)
                done_t = torch.tensor(done, device=device)
                optimize_model(state_t, action_t, next_state_t, reward_t, done_t)

            state = next_state

            if render:
                env.render(mode='human')
            
            if done:
                if i_episode % PRINT_INTERVAL == 0:
                    print('[Episode {:4d}/{}] [Steps {:4d}] [reward {:.1f}]'
                        .format(i_episode, NUM_EPISODES, t, episode_total_reward))
                break
                 
        if i_episode % TARGET_UPDATE == 0:
            target.load_state_dict(model.state_dict())

        if i_episode % TEST_INTERVAL == 0:
            print('-'*10)
            score = eval_policy(policy=model, env=ENV_NAME, render=render)
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), "best_model_{}.pt".format(ENV_NAME))
                print('saving model.')
            print("[TEST Episode {}] [Average Reward {}]".format(i_episode, score))
            print('-'*10)
            score_plot.append(score)
    return score_plot


if __name__ == "__main__":
    score = train_reinforcement_learning()
    with open("replay_score_soft.txt", 'w') as f:
        for s in score:
            f.write(str(s) + '\n')

