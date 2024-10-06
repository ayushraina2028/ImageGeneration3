#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch_snippets import *
device = "cuda" if torch.cuda.is_available() else "cpu"
from torchvision.utils import make_grid


# In[2]:


print(device)


# In[3]:


# Import MNIST data
from torchvision.datasets import MNIST 
from torchvision import transforms


# In[4]:


transform = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize(mean=(0.5,), std=(0.5,))
])


# In[5]:


data_loader = torch.utils.data.DataLoader(MNIST('data', train=True, download=True, transform=transform),batch_size=64, shuffle=True, drop_last=True)


# # Discriminator 

# In[6]:


class Discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        return self.model(x)


# In[7]:


from torchsummary import summary
discriminator = Discriminator().to(device)


# In[8]:


# print the summary of the model 
summary(discriminator, (784,))


# # Generator Class

# In[9]:


class Generator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,784),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.model(x)


# In[10]:


generator = Generator().to(device)
summary(generator, torch.zeros(1,100))


# # Random Noise Generator

# In[11]:


def noise(size):
    n = torch.randn(size, 100)
    return n.to(device)


# In[12]:


def trainDiscriminator(realData, fakeData):
    
    # Reset the Gradients
    d_optimizer.zero_grad()
    
    # predicting on real data and get the loss
    prediction1 = discriminator(realData)
    error1 = loss(prediction1, torch.ones(realData.size(0), 1).to(device))
    error1.backward()
    
    # predicting on fake data and get the loss
    prediction2 = discriminator(fakeData)
    error2 = loss(prediction2, torch.zeros(fakeData.size(0), 1).to(device))
    error2.backward()
    
    # update weights and return overall loss
    d_optimizer.step()
    return error1 + error2


# In[13]:


def trainGenerator(fakeData):
    
    # Reset the Gradients
    g_optimizer.zero_grad()
    
    # prediction
    prediction = discriminator(fakeData)
    
    # calculate the loss
    error = loss(prediction, torch.ones(fakeData.size(0), 1).to(device))
    
    # perform backpropagation
    error.backward()
    g_optimizer.step()
    
    return error


# In[14]:


discriminator = Discriminator().to(device)
generator = Generator().to(device)

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()
num_epochs = 200

log = Report(num_epochs)


# In[32]:


import matplotlib.pyplot as plt

# Initialize lists to store loss values
d_losses = []
g_losses = []

idx = 0
# Training loop
for epoch in range(num_epochs):
    N = len(data_loader)
    for i, (images, _) in enumerate(data_loader):
        real_data = images.view(len(images), -1).to(device)
        fake_data = generator(noise(len(real_data))).to(device)
        fake_data = fake_data.detach()
        
        d_loss = trainDiscriminator(real_data, fake_data)
        fake_data = generator(noise(len(real_data))).to(device)
        g_loss = trainGenerator(fake_data)
        
        # Append losses to lists
        if(i==idx):
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
        
        log.record(epoch+(1+i)/N, d_loss=d_loss.item(), g_loss=g_loss.item(), end='\r')
    idx = idx + 1
    log.report_avgs(epoch+1)

# save d_losses and g_losses in a csv file
import pandas as pd
df = pd.DataFrame({'d_losses':d_losses, 'g_losses':g_losses})
df.to_csv('losses.csv', index=False)



# In[33]:


# import save_image
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI environments
import torch
from torchvision.utils import make_grid, save_image

# Generate images
z = torch.randn(256, 100).to(device)
sample_images = generator(z).data.view(256, 1, 28, 28)

# Create a grid of images
grid = make_grid(sample_images, nrow=16, normalize=True)

# Save the grid of images to a file
save_image(grid, 'gan.png')

# Comment out the following line when running as a script
# show(grid.cpu().detach().permute(1, 2, 0), sz=5)

     


# In[ ]:




