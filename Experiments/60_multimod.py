'''

# requirements file
# note which revision of python, for example 3.9.6
# in this file, insert all the pip install needs, include revision

#for example:
#torch==2.0.1
#matplotlib==3.7.2
openai
torch
numpy
pandas
torchvision
matplotlib
tqdm
python-mnist
scikit-learn
plotly
weaviate-client==4.5.4
umap-learn
bokeh ## needed for umap-learn
holoviews ## needed for umap-learn
scikit-image ## needed for umap-learn
colorcet ## needed for umap-learn
datashader ## needed for umap-learn
opencv-python
python-dotenv==1.0.0
google-generativeai
python-dotenv==1.0.0
dask-expr


'''
import warnings
warnings.filterwarnings("ignore")
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Import Basic computation libraries along with data visualization and ploting library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import umap.plot
from torch.utils.data import Dataset
from tqdm import tqdm
import os
#Define dataset class
class MNISTDataset(Dataset):
    def __init__(self,data_df:pd.DataFrame,transform=None,is_test=False):
        super().__init__()
        dataset=[]
        labels_positive={}
        labels_negative={}
        if is_test==False:
            # for each label create a set of same label images.
            # unique the values in the label:0-9
            for i in list(data_df.label.unique()):
                labels_positive[i]=data_df[data_df.label==i].to_numpy()
            # for each label create a set of image of different label.
            for i in list(data_df.label.unique()):
                labels_negative[i]=data_df[data_df.label!=i].to_numpy()

        for i, row in tqdm(data_df.iterrows(),total=len(data_df)):
            data=row.to_numpy()
            #if is test then only image will be returned
            if is_test:
                label=-1
                first=data.reshape(28,28)
                second=-1
                dis=-1
            else:
                # If is train.
                #label and image of the index for each row in df
                # This is because data is saved to array.
                label=data[0]
                first=data[1:].reshape(28,28)
                #prabability of same label image==0.5
                # 这是 NumPy 的一个函数，用于生成一个随机整数，范围是从 0 到 1（包括 0，不包括 2）
                if np.random.randint(0,2)==0:
                    # randomly select same label image
                    second=labels_positive[label][np.random.randint(0,len(labels_positive[label]))]
                else:
                    # randomly select different(negative) label 
                    second=labels_negative[label][
                        np.random.randint(0,len(labels_negative[label]))
                    ]
                #cosine is 1 for same and 0 for different label
                dis=1.0 if second[0]==label else 0.0
                #reshape image
                second=second[1:].reshape(28,28)
            # apply transform on both images
            if transform is not None:
                first=transform(first.astype(np.float32)) 
                if isinstance(second,np.ndarray) and second.size>0:
                    second=transform(second.astype(np.float32))
            #append to dataset list
            # return the distance between first and second and the label
            dataset.append((first,second,dis,label))

        # append to dataset list. 
        # this random list is created once and used in every epoch
        self.dataset=dataset
        self.transform=transform
        self.is_test=is_test

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
# Define a neural network architecture with two convolution layers and two fully connected layers
# Input to the network is an MNIST image and Output is a 64 dimensional representation. 
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,32,5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2),stride=2),
            nn.Dropout(0.3)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(32,64,5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2),stride=2),
            nn.Dropout(0.3)
        )
        self.linear1=nn.Sequential(
            nn.Linear(64*4*4,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512,64)
        )
    def forward(self,x):
        x=self.conv1(x) # x: d * 32 * 12 * 12
        x=self.conv2(x) # x: d * 64 * 4  * 4 
        x=x.view(x.size(0),-1)  # x: d * (64*4*4)
        x=self.linear1(x) # x: d * 64
        return x
    
# The ideal distance metric for a positive sample is set to 1, for a negative sample it is set to 0   
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.similarity=nn.CosineSimilarity(dim=-1,eps=1e-7)
    
    def forward(self,anchor,contrastive,distance):
        # use cosine similarity from torch to get score
        score=self.similarity(anchor,contrastive)
        #after cosine apply MSE between distance and score
        return nn.MSELoss()(score,distance) #Ensures that the calculated score is close to the ideal distance (1 or 0)



#visualize datapoints
def show_images(images,title=''):
    num_images=len(images)
    fig,axes=plt.subplots(1,num_images,figsize=(9,3))
    for i in range(num_images):
        img=np.squeeze(images[i])
        axes[i].imshow(img,cmap='gray')
        axes[i].axis('off')
    fig.suptitle(title)
    plt.show()

def load_model_from_checkpoint(path,device):
    checkpoint = torch.load(path)
    
    net = Network()
    net.to(device=device)
    net.load_state_dict(checkpoint)
    net.eval()

    return net
  


if __name__ == '__main__':

    # Load data from csv
    data=pd.read_csv("AskThousandLLM/Experiments/train.minist.csv")
    val_count=1000
    #common transformation for both val and  train
    default_transform=transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(0.5,0.5)
        ]
    )
    #Split data into val and train
    dataset=MNISTDataset(data.iloc[:-val_count],default_transform)
    val_dataset=MNISTDataset(data.iloc[-val_count:],default_transform)

    #create torch dataloaders
    trainLoader=DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        prefetch_factor=100
    )

    valLoader=DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=100
    )

    #visualize some examples
    for batch_idx,(anchor_images,contrastive_images,distances,labels) in enumerate(trainLoader):
        #convert tensors to numpy arrays
        anchor_images=anchor_images.numpy()
        contrastive_images=contrastive_images.numpy()
        labels=labels.numpy()

        #Display some samples from the batch
        show_images(anchor_images[:4],title="anchor images")
        show_images(contrastive_images[:4],title='+/- Example')
        break

    # Define the training configuration
    net=Network()
    device="cpu"
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    net=net.to(device)

    #Define the traning configugration
    optimizer=optim.Adam(net.parameters(),lr=0.005)
    loss_function=ContrastiveLoss()
    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.3)


    #Train Loop
    checkpoint_dir="AskThousandLLM/Experiments/MultiModaCheckPoint"

    #ENsure the directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    lrs=[]
    losses=[]
    for epoch in range(1,2):
        epoch_loss=0
        batches=0
        print(f"Epoch - {epoch}")
        lrs.append(optimizer.param_groups[0]['lr'])
        print(f"learing rate:{lrs[-1]}")
        for anchor,contranstive,distance,label in tqdm(trainLoader):
            batches+=1
            optimizer.zero_grad()
            anchor_out=net(anchor.to(device))
            contrastive_out=net(contranstive.to(device))
            distance=distance.to(torch.float32).to(device)
            loss=loss_function(anchor_out,contrastive_out,distance)
            epoch_loss+=loss
            loss.backward()
            optimizer.step()
            losses.append(epoch_loss.cpu().detach().numpy()/batches)
        scheduler.step()
        print('epoch_loss', losses[-1])
        #save a checkpoint of the model
        checkpoint_path=os.path.join(checkpoint_dir,f"model_epoch_{epoch}.pt")
        torch.save(net.state_dict(),checkpoint_path)
        #show loss trend
        plt.plot(losses)
        plt.show()

