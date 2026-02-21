import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

IM_WIDTH = 1024
IM_HEIGHT = 737
NUM_CHANNELS = 3

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        #encoder
        self.enc1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)   #737x1024 to 368x512
        self.enc2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)

        #decoder
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        #encoder
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))

        #decoder
        d1 = self.relu(self.dec1(e3))
        d1 = torch.cat([d1, e2], dim=1)

        d2 = self.relu(self.dec2(d1))
        d2 = torch.cat([d2, e1], dim=1)

        d3 = self.dec3(d2)

        if d3.size()[2:] != x.size()[2:]:
            d3 = F.interpolate(d3, size=x.size()[2:], mode='bilinear', align_corners=False)

        
        return torch.sigmoid(d3)

def train(model, dataloader, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for night, day in dataloader:
            night = night.to(device)
            day = day.to(device)

            optimizer.zero_grad()

            output = model(night)
            loss = criterion(output, day)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss {total_loss / len(dataloader)}")

def show_results(model, night_img, day_img):
    model.eval()

    with torch.no_grad():
        input_tensor = night_img.unsqueeze(0).to(device)
        output = model(input_tensor)[0].cpu()

    night_np = night_img.permute(1,2,0).cpu().numpy()
    day_np = day_img.permute(1,2,0).cpu().numpy()
    output_np = output.permute(1,2,0).numpy()

    plt.figure(figsize=(18,6))

    plt.subplot(1,3,1)
    plt.imshow(night_np)
    plt.title("Original Night")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(day_np)
    plt.title("Original Day")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(output_np)
    plt.title("Transformed Day")
    plt.axis("off")

    plt.show()

def loadImages():
    folderPath = 'ML Code/training_data'
    labelsPath = 'ML Code/labels/label.txt'
    outputPath = 'ML Code/transformedFiles'
    
    labels = np.loadtxt(labelsPath, dtype=str)
    
    nightList = []
    dayList = []
    
    for label in labels:
        img = Image.open(folderPath + "/" + label[0]).convert("RGB")
        img_np_array = np.array(img)
        print(img_np_array.shape)
        
        imgName = outputPath + "/" + label[0][0:len(label[0])-4]
        
        if int(label[2]) < 7 or int(label[2]) > 17:
            imgName += " Night"
            nightList.append(img_np_array)
        else:
            imgName += " Day"
            dayList.append(img_np_array)
        
        np.save(imgName, img_np_array)
    
    nightSet = np.array(nightList)
    daySet = np.array(dayList)
    
    print(nightSet.shape)
    print(daySet.shape)
    
    print(labels)
    
    return nightSet, daySet

def RGB_MSE(output, day):
    pass

if __name__ == "__main__":
    nightSet, daySet = loadImages()
    
    #allow randomness
    #this variable can be changed to false if desired
    #if True: will randomly match day and night time images
    #if False: will cut off the excess night time images and match them according to input order
    randomness = True
    
    if randomness:
        nightSet = nightSet[nightSet.shape[0] - daySet.shape[0]:]
        
        shuffledIndices = np.random.permutation(nightSet.shape[0])
        
        nightSet = nightSet[shuffledIndices]
    else:
        nightSet = nightSet[nightSet.shape[0] - daySet.shape[0]]
        pass
    
    #convert to tensors
    #change from (number, height, width, channel) to (number channel, height, width) cause pytorch wants that
    #also divide by 255 to normalize for faster training
    nightSet = torch.tensor(nightSet / 255.0, dtype=torch.float32).permute(0,3,1,2)
    daySet   = torch.tensor(daySet / 255.0, dtype=torch.float32).permute(0,3,1,2)
    
    #divide into training and validation sets
    print(nightSet.shape)
    portion = int(nightSet.shape[0] * 0.8)
    trainX = nightSet[:portion]
    trainY = daySet[:portion]
    testX = nightSet[portion:]
    testY = daySet[portion:]
    
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)
    
    #prepare the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()   # L1 works better than MSE for images
    
    #load the data
    trainTensor = TensorDataset(trainX, trainY)
    trainLoader = DataLoader(trainTensor, batch_size = 2, shuffle = True)
    
    train(model, trainLoader, epochs = 100)
    
    show_results(model, testX[0], testY[0])
