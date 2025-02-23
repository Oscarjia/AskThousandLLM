
'''
Here is example python code of PPO:Proximal Policy Optimization 
and Use the code to finetune a LLM
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import GradScaler, autocast, nn
from torch import optim
#Categorical用于模拟智能体做选择的过程
from torch.distributions.categorical import Categorical
from transformers import  AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# GPT have 115 M 
# openai-community/gpt2
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
SEQ_LENGTH = 128
BATCH_SIZE = 1  # 根据显存调整
PPO_CLIP = 0.2
GAMMA = 0.99
LAMBDA = 0.95
KL_BETA = 0.1
LR = 1e-6
EPOCHS = 3
TARGET_KL = 0.01
class LLMPPO(nn.Module):

    def __init__(self,proxies):
        super().__init__()
        # 以 LLM 模型为基础
        # 策略和价值网络
        self.actor=AutoModelForCausalLM.from_pretrained(MODEL_NAME,proxies=proxies)
        self.critic=nn.Linear(self.actor.config.hidden_size,1) # 价值网络

        # 冻结部分参数（可选）
        for param in self.actor.parameters():
            param.requires_grad = False
        for param in self.actor.get_output_embeddings().parameters():
            param.requires_grad = True

    def forward(self,input_ids,attention_mask=None):
        outputs=self.actor(input_ids,attention_mask=attention_mask,output_hidden_states=True)
        # 获取所有token的隐藏状态 [batch, seq_len, dim]
        all_hidden =outputs.hidden_states[-1]
        # 计算每个位置的价值 [batch, seq_len]
        values=self.critic(all_hidden).squeeze(-1)
        return outputs.logits,values

class RewardCalculator:
    '''
    design a reward calulator of a model.
    '''

    def __init__(self,tokenizer,reward_model=None):
        self.tokenizer=tokenizer
        self.reward_model=reward_model
    
    def __call__(self, sequences):
        batch_rewards=[]
        for seq in sequences:
            #转换token 序列为文本
            text=self.tokenizer.decode(seq,skip_special_tokens=True)
            # 实际需要根据任务设计的奖励逻辑
            # 示例1：基于规则
            step_rewards=[]
            for i in range(1,len(seq)):
                partial_text=self.tokenizer.decode(seq[:i+1])
                reward=self._rule_based_reward(partial_text)
                step_rewards.append(reward)
            
            # 示例二：使用奖励模型
            if self.reward_model:
                with torch.no_grad():
                    inputs=self.tokenizer(text,return_tensors='pt').to(DEVICE)
                    reward=self.reward_model(**inputs).logits.item()
                step_rewards=[reward/len(seq)]*len(seq) # 平均分配
            
            batch_rewards.append(step_rewards)

        return torch.tensor(batch_rewards,device=DEVICE)

    def _rule_based_reward(self, text):
        """示例规则奖励函数"""
        reward = 0
        # 鼓励长文本但惩罚冗余
        reward += len(text) * 0.01
        # 惩罚重复词
        unique_words = len(set(text.split()))
        reward += unique_words * 0.05
        # 特殊关键词奖励
        if "千问LLM" in text: reward += 0.5
        return reward



# 奖励函数
def calculate_rewards(sequences,tokenizer):
    '''
    计算 actor 的输出是 batch sequences。
    '''
    rewards=[]
    for seq in sequences:
        text=tokenizer.decode(seq,skip_special_tokens=True)
        # 鼓励生成长文本
        reward=len(text.split())/100
        rewards.append(reward)
    return torch.tensor(rewards,device=DEVICE)

#数据生成
def generate_rollout(model,tokenizer,prompt):
    '''
    model is PPO model

    '''
    inputs=tokenizer(prompt,return_tensors="pt").to(DEVICE)
    #生成序列,让模型根据输入来输出。
    # Model output Shape :[batch_size, seq_length]
    generate_ids=model.actor.generate(
        inputs.input_ids,
        max_length=SEQ_LENGTH,
        do_sample=True,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,  # 关键参数
        output_hidden_states=True
    )
    # 计算对数概率
    log_probs = []
    for step, (scores, tokens) in enumerate(zip(generate_ids.scores, generate_ids.T)):
    
    # 计算奖励
    rewards = model.reward_calculator(seq.unsqueeze(0))[0]
    
    # 构建终止标记（示例：假设生成到最大长度时终止）
    dones = torch.zeros(len(rewards), device=DEVICE)
    dones[-1] = 1
    
    return {
        "sequences": seq.unsqueeze(0),
        "log_probs": log_probs,
        "rewards": rewards,
        "dones": dones
    }


class PPOTrainner():
    def __init__(self,model,tokenizer):
        self.model=model
        self.tokenizer=tokenizer
        self.scaler=GradScaler()
        self.reward_calculator=RewardCalculator(tokenizer)
        self.kl_beta=KL_BETA
        self.optimizer=AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=LR)

    def compute_addvantages(self,rewards,values):
        '''
        计算GAE优势值
        rewards: [seq_len]
        values: [seq_len+1]
        dones: [seq_len]
        '''
        advantages=[]
        last_advantage=0
        for t in reversed(range(len(rewards))):
            delta=rewards[t]+GAMMA*values[t+1]-values[t] if t<len(rewards)-1 else rewards[t]-values[t]
            advantage=delta+GAMMA*LAMBDA*last_advantage
            advantages.insert(0,advantage)
            last_advantage=advantage

        return torch.stack(advantages)
    
    def train_step(self,rollouts):
        total_loss=0
        self.optimizer.zero_grad()

        for rollout in rollouts:
            #解包数据
            seq=rollout["sequences"]
            old_log_probs=rollout["log_probs"]
            rewards=rollout["rewards"]
            dones=rollout["dones"]

            #前向计算
            with autocast():
                logits,values=self.model(seq[:,:-1])# 排除最后一个预测
                values=values.squeeze()

                #计算新策略概率
                dist=Categorical(logits=logits)
                new_log_probs=dist.log_prob(seq[:,1:])

                # 计算优势
                advantages = self.compute_addvantages(rewards, values)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # PPO 损失
                ratios=torch.exp(new_log_probs-old_log_probs)
                clipped_ratios=torch.clamp(ratios,1-PPO_CLIP,1+PPO_CLIP)
                policy_loss=-torch.min(ratios*advantages,clipped_ratios*advantages)

                #价值损失
                value_loss=0.5*(values[:-1]-rewards).pow(2).mean()

                #KL 散度惩罚

                kl_div=(old_log_probs-new_log_probs).mean()
                kl_penalty=self.kl_beta*kl_div

                #总损失
                loss=policy_loss+value_loss+kl_penalty

                total_loss+=loss/len(rollouts)
            #梯度累积
            self.scaler.scale(loss).backward()

        # 梯度裁剪和参数更新
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # 动态调整KL系数
        if kl_div > 1.5 * TARGET_KL:
            self.kl_beta *= 2
        elif kl_div < TARGET_KL / 1.5:
            self.kl_beta /= 2
        
        return total_loss.item()


    def update(self,rollouts):
        all_losses=[]
        for _ in range(EPOCHS):
            for rollout in rollouts:
                #计算优势
                advantages=self.compute_addvantages(
                    rollout["rewards"],
                    rollout["values"]
                )
                #标准化优势
                advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)

                #计算新旧策略概率比
                old_log_probs=rollout["log_probs"].detach()
                logits,_=self.model(rollout["sequences"][:,:-1]) # 排除最后一个token
                new_dist=Categorical(logits=logits)
                new_log_probs=new_dist.log_prob(rollout["sequences"][:,1:])
                ratios=torch.exp(new_log_probs-old_log_probs)
                #计算策略损失
                surr1=ratios*advantages
                surr2=torch.clamp(ratios,1-PPO_CLIP,1+PPO_CLIP)*advantages
                policy_loss=-torch.min(surr1,surr2).mean()

                #计算价值损失
                _,current_values=self.model(rollout["sequences"][:,:-1])
                value_loss=0.5*(current_values-rollout["values"]).pow(2).mean()
                #计算KL惩罚
                kl_penalty=KL_BETA*(old_log_probs-new_log_probs).mean()
                #总损失
                total_loss=policy_loss+value_loss+kl_penalty

                #反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                self.optimizer.step()
                all_losses.append(total_loss.item())
            
        return np.mean(all_losses)        

    

def discount_rewards(rewards,gamma=0.99):
    """
    Return discounted rewards based on the given rewards and gamma param
    """
    # 最后一个直接获得折扣奖励。
    new_rewards=[float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        #然后依次计算上一个的折扣奖励的值
        new_rewards.append(float(rewards[i])+gamma*new_rewards[-1])
       
    return np.array(new_rewards[::-1])  # 这行代码将折扣奖励的列表 new_rewards 从 最后一个奖励 到 第一个奖励 反转，并将其转化为 NumPy 数组返回。



if __name__=='__main__':

    proxies={}

    #初始化组件
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,proxies=proxies)
    model=LLMPPO(proxies=proxies).to(DEVICE)
    trainer=PPOTrainner(model,tokenizer)

    #训练循环
    for epoch in range(100):
        #生成数据
        rollouts=[generate_rollout(model,tokenizer,"Human: ") for _ in range(BATCH_SIZE)]

        #参数更新
        avg_loss=trainer.train_step(rollouts)

        #监控训练过程
        if epoch % 5==0:
            avg_reward=sum(torch.mean(rollout["rewards"]).item() for rollout in rollouts)/len(rollouts)
            print(f"Epoch {epoch} | Loss: {avg_loss:.3f} |Avg Reward: {avg_reward:.3f}")

    

#
# Epoch 0 | Loss: 29.402 |Avg Reward: 0.170
# Epoch 5 | Loss: 24.997 |Avg Reward: 0.180
# Epoch 10 | Loss: 18.031 |Avg Reward: 0.190
# Epoch 15 | Loss: 16.673 |Avg Reward: 0.190
# Epoch 20 | Loss: 20.196 |Avg Reward: 0.170
# Epoch 25 | Loss: 25.567 |Avg Reward: 0.130
# Epoch 30 | Loss: 38.930 |Avg Reward: 0.170
# Epoch 35 | Loss: 18.160 |Avg Reward: 0.210
# Epoch 40 | Loss: 21.719 |Avg Reward: 0.190
# Epoch 45 | Loss: 25.167 |Avg Reward: 0.210