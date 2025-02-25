
'''
Here is example python code of PPO:Proximal Policy Optimization 
and Use the code to finetune a LLM

Core Package
pip install transformers==4.49.0 torch==2.5.1 flash_attn==2.7.4.post1
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
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
        last_hidden=outputs.hidden_states[-1][:,-1,:] # 取最后一个token的隐藏状态
        value=self.critic(last_hidden)
        return outputs.logits,value

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
        output_hidden_states=True
    )
    # 提取各时间步信息
    log_probs=[]
    values=[]
    rewards=[]
    # 获取模型输出的长度。
    for t in range(1,generate_ids.sequences.shape[1]):
        #获取历史序列
        partial_seq=generate_ids.sequences[:,:t]
        with torch.no_grad():
            # 逐token计算 value
            logits,value=model(partial_seq)
            # why ?
            dist=Categorical(logits=logits[:,-1,:])
            token_prob=dist.log_prob(generate_ids.sequences[:,t])
        log_probs.append(token_prob)
        values.append(value.squeeze())
    
    #计算最终奖励
    final_rewards=calculate_rewards(generate_ids.sequences,tokenizer)

    return {
        "sequences":generate_ids.sequences,
        "log_probs":torch.stack(log_probs),
        "values":torch.stack(values),
        "rewards":final_rewards.repeat(len(log_probs)) # 简单分配奖励
    }


class PPOTrainner():
    def __init__(self,model,tokenizer):
        self.model=model
        self.tokenizer=tokenizer
        self.optimizer=AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=LR)

    def compute_addvantages(self,rewards,values):
        advantages=[]
        last_advantage=0
        for t in reversed(range(len(rewards))):
            delta=rewards[t]+GAMMA*values[t+1]-values[t] if t<len(rewards)-1 else rewards[t]-values[t]
            advantage=delta+GAMMA*LAMBDA*last_advantage
            advantages.insert(0,advantage)
            last_advantage=advantage

        return torch.stack(advantages)
    
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
        avg_loss=trainer.update(rollouts)

        #监控训练过程
        if epoch % 5==0:
            avg_reward=sum(torch.mean(rollout["rewards"]).item() for rollout in rollouts)/len(rollouts)
            print(f"Epoch {epoch} | Loss: {avg_loss:.3f} |Avg Reward: {avg_reward:.3f}")

    

    