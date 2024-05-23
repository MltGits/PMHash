import torch

torch.set_printoptions(profile="full")
import torch.nn as nn
import torchvision
from torch.nn.parameter import Parameter
from torch.nn import Linear
from feature_loader import autoDataset
from torch.utils.data import DataLoader

import os


class AE(nn.Module):
    # enc_dims = [2048, 512, 512, 256, 256]
    def __init__(self, laten_dim, enc_dims=[2048, 1024, 512, 256, 128]):
        super(AE, self).__init__()

        self.laten_dim = laten_dim

        self.encoder = nn.Sequential(
            Linear(enc_dims[0], enc_dims[1]), nn.ReLU(),
            Linear(enc_dims[1], enc_dims[2]), nn.ReLU(),
            Linear(enc_dims[2], enc_dims[3]), nn.ReLU(),
            Linear(enc_dims[3], enc_dims[4]), nn.ReLU(),
            Linear(enc_dims[4], laten_dim),
        )
        self.decoder1 = nn.Sequential(
            Linear(laten_dim, enc_dims[4]), nn.ReLU(),
            Linear(enc_dims[4], enc_dims[3]), nn.ReLU(),
            Linear(enc_dims[3], enc_dims[2]), nn.ReLU(),
            Linear(enc_dims[2], enc_dims[1]), nn.ReLU(),
            Linear(enc_dims[1], enc_dims[0])
        )

    def forward(self, x):
        x_compressed = self.encoder(x)
        x_reconstruct1 = self.decoder1(x_compressed)

        return x_reconstruct1, x_compressed


def main():
    path = "dataset/NUS-WIDE/all_feature_train.pickle"
    train_dataset = autoDataset(path, multi=True, class_cnt=21, start=0, end=5000, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

    device = torch.device("cuda")
    model = AE(laten_dim=64, enc_dims=[train_dataset.x1_list.shape[1], 512, 512, 256, 256]).to(device)
    criterion_mse = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    minloss = 1
    last_path = ""
    print(model)
    for epoch in range(200):
        for batch_idx, feature in enumerate(train_loader):
            feature = feature.to(torch.float32).to(device)
            x_reconstruct1, x_compressed = model(feature)

            loss = criterion_mse(x_reconstruct1, feature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch: {} batch_idx:{} loss: {:.8f} '.format(epoch, batch_idx, loss))

            if minloss > loss:
                minloss = loss
                if last_path != "" and os.path.exists(last_path):
                    os.remove(last_path)
                path = 'dataset/NUS-WIDE/all_features_train'+"_" + str(minloss.item())
                torch.save(model.state_dict(), path)
                last_path = path
                print('save model in: %s' % path)


if __name__ == '__main__':
    main()
