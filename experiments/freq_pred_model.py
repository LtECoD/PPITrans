import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class FreqPredictor(pl.LightningModule):
    def __init__(self, input_dim=None, out_dim=None):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim))

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        # features: B x D
        # freq: B x L 
        features, freq = batch
        logits = self.layers(features)
        probs = torch.softmax(logits, dim=-1)

        kl = probs * torch.log(probs / (freq+1e-6))
        kl = torch.sum(kl, dim=-1)

        return kl

    def training_step(self, batch, batch_idx):
        kls = self.forward(batch)
        loss = torch.mean(kls)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        kls = self.forward(batch)
        loss = torch.mean(kls)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class FeatureDataset(Dataset):
    def __init__(self, features, freq):
        """
        features: ndarray, N x D
        freq:   ndarray, N x L
        """
        self.features = torch.Tensor(features)
        self.freq = torch.Tensor(freq)

    def __getitem__(self, idx):
        return self.features[idx], self.freq[idx]

    def __len__(self):
        return self.features.size(0)


def get_dataloader(features, freq, shuffle=True):
    dataset = FeatureDataset(features, freq)
    loader = DataLoader(dataset=dataset, batch_size=16, shuffle=shuffle)
    return loader


class MyTrainer:
    def __init__(self, input_dim, out_dim):
        self.model = FreqPredictor(input_dim=input_dim, out_dim=out_dim)

        early_stop_callback = EarlyStopping(monitor="val_loss", \
            min_delta=0.00, patience=5, verbose=False, mode="min")
        self.trainer = pl.Trainer(gpus=1, max_epochs=50, \
            callbacks=[early_stop_callback])
    
    def fit(self, features, freq, val_feat, val_freq):
        loader = get_dataloader(features, freq)
        val_loader = get_dataloader(val_feat, val_freq, False)
        self.trainer.fit(self.model, loader, val_loader)

    def predict(self, features, freq):
        loader = get_dataloader(features, freq)

        losses = []
        with torch.no_grad():
            for feat, fq in loader:
                loss = self.model((feat, fq))
                losses.extend(loss.cpu().numpy().tolist())
        return losses 

    def load(self, ckpt):
        self.model.load_state_dict(torch.load(ckpt))
    
    def save(self, ckpt):
        torch.save(self.model.state_dict(), ckpt)


# from torchvision.datasets import MNIST
# dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
# train, val = random_split(dataset, [550, 50])

# autoencoder = FreqPredictor()
# trainer = pl.Trainer(gpus=1, max_epochs=3)
# trainer.fit(autoencoder, DataLoader(train), DataLoader(val))