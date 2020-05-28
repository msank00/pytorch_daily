import torch
from torch.nn import functional as F 
from torch import nn
import pytorch_lightning as pl 
from torchvision import datasets, transforms

class VAE(pl.LightningModule):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def training_step(self, batch, batch_index):
        x, _ = batch
        recon_batch, mu, logvar = self(x)
        loss = self.loss_function(recon_batch, x, mu, logvar)
        
        return {'loss':loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
        
if __name__ == "__main__":
    
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--cuda', default=False)
    args = parser.parse_args()

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

    vae = VAE()
    trainer = pl.Trainer()
    trainer.fit(vae, train_dataloader=train_loader)