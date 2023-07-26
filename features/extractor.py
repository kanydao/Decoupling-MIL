from YOUR_DATASET import DATASET
from models import resnet
import torch.nn as nn
import torch
import numpy as np
import os
#-----------------------------------------------------------------
device = torch.device("cpu")
#-----------------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers=['fc']):
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule
        self.submodule.to(device)
        self.extracted_layers = extracted_layers

    def forward(self, x):
        x = x.to(device)
        outputs = []
        for name, module in self.submodule._modules.items():
            # print(name)
            if "fc" in name:
                x = x.view(x.size(0), -1)
            if name in self.extracted_layers:
                outputs.append(x.to('cpu'))
            x = module(x)
        return outputs
    
def get_feature():
    #setting 
    dataset = DATASET
    net = resnet.i3_res50_nl(num_classes=400)
    myexactor = FeatureExtractor(submodule=net)
    # extract features:
    j = -1
    for dataname in dataset.data:
        j = j + 1
        
        feature_path = dataname.split('/')[-2:]
        feature_path = '/mnt/889cdd89-1094-48ae-b221-146ffe543605/xrs/12121_project/data/baseline/features/' + feature_path[0] + '/' + feature_path[1] + '_i3d.npy'

        if os.path.exists(feature_path): 

            print('loading ' + dataname)
            video = dataset[j]
            print('finish reading: {}'.format(video['location']))

            expandedVideo = video['frames'].unsqueeze(dim=0)
            frames = torch.split(expandedVideo, 16, dim=2)

            features = []
            frameLength = len(frames)
        
            block_size = 16
            for i in range(frameLength):
                if frames[i].size()[2] < block_size:
                    padsize = (frames[i].size()[0], frames[i].size()[1], block_size - frames[i].size()[2], frames[i].size()[3],frames[i].size()[4])
                    padtensor = torch.zeros(padsize)
                    frame = np.concatenate((frames[i], padtensor), axis=2)
                else: frame = frames[i]

                x = myexactor(torch.tensor(frame))
                features = features + x
                print('extracting---{}/{}'.format(i+1, frameLength))
            
                if i % block_size == block_size-1 or i == frameLength-1:
                    if i % block_size != 0: features = torch.stack(features)
                    else : features = torch.tensor([feature.cpu().detach().numpy() for feature in features])
                    np.save(feature_path + 'temp_{}.npy'.format(i // block_size) , features.detach().numpy())
                    features = []
            
            for i in range((frameLength + block_size - 1) // block_size):
                if i == 0: features = np.load(feature_path + 'temp_{}.npy'.format(i))
                else: features = np.concatenate((features , np.load(feature_path + 'temp_{}.npy'.format(i))),axis=0)

            np.save(feature_path,features)

            for i in range((frameLength + block_size - 1) // block_size):
                os.remove(feature_path + 'temp_{}.npy'.format(i))

        else: 
            print(dataname + '\'s feature is already extracted, skip.')
        
if __name__ == "__main__":    
    get_feature()
