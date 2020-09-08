from __future__ import print_function
import argparse
import math
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.distributions as tdist


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--use_UT', action='store_true', default=False,
                    help='the model uses unscented transformation for sampling')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(40, 400)
        self.fc41 = nn.Linear(400, 784)
        self.fc42 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) 
        return mu + eps*std

    def decode(self, z, istrain=True):
        h3 = F.relu(self.fc3(z))
        if istrain:
            return torch.sigmoid(self.fc4(h3))
        else:
            return self.fc41(h3), self.fc42(h3)
      
      def svdsqrtm(self, x, eps=1e-15):
        #Return the matrix square root of x calculating using the svd.
    
        #Set singular values < eps to 0.
        #This prevents numerical errors from making sqrtm(x) complex
        #(small eigenvalues of x accidently negative).
        u, s, v = torch.svd(x)
        s_notneg = torch.zeros_like(s)
        for i in range(s.size):
            if s[i] > eps:
                s_notneg[i] = s[i]

        return torch.dot(u, torch.dot(torch.diag(torch.sqrt(s_pos)), torch.transpose(v)))
    
    def unscented(self, mu, logvar):
        #For a vector mu of length N with covariance matrix logvar,
        #form 2N sigma points used for taking the unscented transform.
        mu = mu.view(-1) # Force shape
        N = mu.shape[1]
        varsqrt = scale * self.svdsqrtm(N * logvar)
        x_sigma = []
        
        for i in xrange(N):
            x_sigma.append(mu + varsqrt[:, i])

        for i in xrange(N):
            x_sigma.append(mu - varsqrt[:, i])

        return x_sigma
      
    def norm_dist(self, x, mu, var):
        return (1/(var*torch.sqrt(2*math.pi())))*torch.exp(-(1/(2*var))*(x - mu)^2)
    
    def sample_loss(self, z, mu_z, var_z, x):
        K = z.shape[0]
        with torch.no_grad():
            mu_x, var_x = self.decode(z)
        
        #q_z_x = tdist.Normal(torch.tensor(mu_z), torch.tensor(var_z))
        #p_x_z = tdist.Normal(torch.tensor(mu_x), torch.tensor(var_x))
        #p_z = tdist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        
        for sample in z:
            #q_z_x = self.norm_dist(sample, mu_z, var_z)
            q_z_x = (1/(var_z*torch.sqrt(2*math.pi())))*torch.exp(-1/2)
            print(q_z_x.shape)
            p_x_z = self.norm_dist(x, mu_x, var_x)
            print(p_x_z.shape)
            p_z = self.norm_dist(sample, 0, 1)
            print(p_x_z.shape)
                 
        sampled_ELBO = torch.log((1/K)*torch.sum((p_x_z*p_z)/q_z_x)
        
    def unscented_mu_cov(self, x_sigma):
        #Approximate mean, covariance from 2N sigma points transformed through
        #an arbitrary non-linear transformation.
        #Returns a flattened 1d array for x.
        N = len(x_sigma)
        pts = torch.tensor(x_sigma)

        x_mu = torch.mean(pts, axis=0)
        diff = pts - x_mu
        x_cov = torch.dot(diff.T, diff) / N
        return x_mu, x_cov
  

    def forward(self, args, x, istrain=True):
        mu, logvar = self.encode(x.view(-1, 784))
        if istrain:
            z = self.reparameterize(mu, logvar)
        else:
            z = self.unscented(mu, logvar)
        return self.decode(z, istrain=istrain), mu, logvar


model = VAE(args).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(args, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(args, data, istrain=False)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(args, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(args, data, istrain=False)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(args, epoch)
        test(args, epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
