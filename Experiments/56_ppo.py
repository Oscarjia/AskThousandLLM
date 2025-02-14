
'''
pip install gym==0.26.2 matplotlib torch numpy

Here is example python code of PPO:Proximal Policy Optimization 
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import  gym 
#Categorical用于模拟智能体做选择的过程
from torch.distributions.categorical import Categorical

DEVICE=torch.device("cuda") if torch.cuda.is_available() else 'cpu'

# print(f"DEVICE:{DEVICE}")

#Action Space: Defines the possible actions the agent can take (discrete or continuous).
#Observation Space: Defines the possible observations the agent can receive from the environment (discrete or continuous).
#policy and value model
class ActorCriticNetwork(nn.Module):

    def __init__(self,obs_space_size,action_space_size):
        super().__init__()

        self.shared_layers=nn.Sequential(
            nn.Linear(obs_space_size,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU()
        )
        self.policy_layers=nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,action_space_size)
        )

        self.value_layers=nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def value(self,obs):
        z=self.shared_layers(obs)
        value=self.value_layers(z)
        return value
    
    def policy(self,obs):
        z=self.shared_layers(obs)
        policy_logits=self.policy_layers(z)
        return policy_logits
    
    def forward(self,obs):
        z=self.shared_layers(obs)
        policy_logits=self.policy_layers(z)
        value=self.value_layers(z)
        return policy_logits,value

class PPOTrainner():
    def __init__(self,actor_critic,ppo_clip_val=0.2,target_kl_div=0.01,max_policy_train_iters=80,value_train_iters=80,policy_lr=3e-4,value_lr=1e-2):
        self.ac=actor_critic
        # 设定一个初始值，这个值用于控制更新的范围
        self.ppo_clip_val=ppo_clip_val
        # 这个值用于设定kl所允许更新的差异值。
        self.target_kl_div=target_kl_div
        self.max_policy_train_iters=max_policy_train_iters
        self.value_train_iters=value_train_iters
        policy_params=list(
            self.ac.shared_layers.parameters()
        )+list(self.ac.policy_layers.parameters())

        self.policy_optim=optim.Adam(policy_params,lr=policy_lr)
        value_params=list(self.ac.shared_layers.parameters())+list(self.ac.value_layers.parameters())
        self.value_optim=optim.Adam(value_params,lr=value_lr)


    def train_policy(self,obs,acts,old_log_probs,gaes,):

        for _ in range(self.max_policy_train_iters):

            self.policy_optim.zero_grad()
            new_logits=self.ac.policy(obs)
            new_logits=Categorical(logits=new_logits)
            new_log_probs=new_logits.log_prob(acts)
            policy_ratio=torch.exp(new_log_probs-old_log_probs)
            # 限制更新的比率避免更新太大
            clipped_ratio=policy_ratio.clamp(1-self.ppo_clip_val,1+self.ppo_clip_val)
        
        
            clipped_loss=clipped_ratio*gaes
            full_loss=policy_ratio*gaes
            # 在 PPO 算法中，我们的目标是最大化某个期望值（例如，优势函数加权后的策略比率），也就是说我们希望模型的策略变得越来越好。但是，常见的优化器（比如 Adam）是通过最小化损失函数来更新参数的。为此，我们需要把“最大化”问题转换成“最小化”问题，这就需要取目标函数的负值。
            policy_loss=-torch.min(full_loss,clipped_loss).mean()
            policy_loss.backward()
            self.policy_optim.step()
            kl_div=(old_log_probs-new_log_probs).mean()
            ##如果策略更新超过某一个值，就结束训练。
            if kl_div>=self.target_kl_div:
                break

    def train_value(self,obs,returns):
        for _ in range(self.value_train_iters):
            self.value_optim.zero_grad()
            values=self.ac.value(obs)
            value_loss=(returns-values)**2
            value_loss=value_loss.mean()
            value_loss.backward()
            self.value_optim.step()


def rollout(model,env,max_steps=1000):
    '''
    Performs a single rollout.
    returns traning data in the shape(n_steps,observation_shape)
    and the cumulative reward
    '''
    #create data storage
    train_data=[[],[],[],[],[]] # obs,act,reward,values,act_log_probs
    # env feedback
    obs,info=env.reset()
    # print(obs.shape)

    # print(info)
    ep_reward=0

    for _ in range(max_steps):
        # logits 模拟 policy model 的输出，可以有多个actions
        logits,val=model(torch.tensor([obs],dtype=torch.float32,device=DEVICE))
        act_distribution=Categorical(logits=logits)
        # 智能体随机选择从一系列actions里面选择一个。
        act=act_distribution.sample()
        act_log_prob=act_distribution.log_prob(act).item()

        act,val=act.item(),val.item()

        next_obs,reward,terminated, truncated,_=env.step(act)
        done = terminated or truncated  # 处理 done 逻辑

        #record data for training
        # 循环列表
        for i,item in enumerate((obs,act,reward,val,act_log_prob)):
            train_data[i].append(item)
        # 更新state的值
        obs=next_obs
        ep_reward+=reward
        if done:
            break
    train_data=[np.asarray(x) for x in train_data]

    # 根据Reards 和 value的值 用GAE来替换 Values

    train_data[3]=calculate_gaes(train_data[2],train_data[3])

    return train_data,ep_reward

def discount_rewards(rewards,gamma=0.99):
    """
    Return discounted rewards based on the given rewards and gamma param
    """
    new_rewards=[float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i])+gamma*new_rewards[-1])
    return np.array(new_rewards[::-1])


def calculate_gaes(rewards,values,gamma=0.99,decay=0.97):
    next_values=np.concatenate([values[1:],[0]])
    deltas=[rew+gamma*next_val-val for rew,val,next_val in zip(rewards,values,next_values)]
    gaes=[deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i]+decay*gamma*gaes[-1])

    return np.array(gaes[::-1])

  

if __name__=='__main__':

    env=gym.make('CartPole-v0')

    model=ActorCriticNetwork(env.observation_space.shape[0],env.action_space.n)

    model=model.to(DEVICE)

    train_data,reward=rollout(model,env)

    # print(np.array(train_data[0]).shape)

    # define training params

    n_episodes=200
    print_freq=20


    ppo=PPOTrainner(model,policy_lr=3e-4,value_lr=1e-3,target_kl_div=0.02,max_policy_train_iters=40,value_train_iters=40)

      # Traning loop
    ep_rewards=[]
    for episode_idx in range(n_episodes):
        #perform rollout
        train_data,reward=rollout(model,env)
        ep_rewards.append(reward)
        #shuffle
        # 这行代码的作用是对训练数据的索引进行 随机排列，以确保训练数据的顺序在每个训练周期（episode）内是 不同的，从而提升训练的随机性，防止模型过拟合到某个特定的顺序模式。
        # train_data[0] 存储的是 观测数据（observations），它的长度等于本次 rollout（执行策略收集数据）中的步数 n_steps。
        permute_idxs=np.random.permutation(len(train_data[0]))
        # Data formatting
        obs=torch.tensor(train_data[0][permute_idxs],dtype=torch.float32,device=DEVICE)
        acts=torch.tensor(train_data[1][permute_idxs],dtype=torch.float32,device=DEVICE)
        gaes=torch.tensor(train_data[3][permute_idxs],dtype=torch.float32,device=DEVICE)
        act_log_probs=torch.tensor(train_data[4][permute_idxs],dtype=torch.float32,device=DEVICE)

        returns=discount_rewards(train_data[2])[permute_idxs]
        returns=torch.tensor(returns,dtype=torch.float32,device=DEVICE)

        ppo.train_policy(obs,acts,act_log_probs,gaes)
        ppo.train_value(obs,returns)


        # Policy + value network training
        if (episode_idx+1)%print_freq==0:
            print('Episode {} | Avg Reward {:.1f}'.format(episode_idx+1,np.mean(ep_rewards[-print_freq:])))

