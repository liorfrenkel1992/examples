from __future__ import print_function
import argparse
import math
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions.multivariate_normal import MultivariateNormal


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
        self.fc3 = nn.Linear(20, 400)
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
        #return torch.sigmoid(self.fc4(h3))
        return self.fc41(h3), self.fc42(h3)
      
    def svdsqrtm(self, x, eps=1e-15):
        #Return the matrix square root of x calculating using the svd.
    
        #Set singular values < eps to 0.
        #This prevents numerical errors from making sqrtm(x) complex
        #(small eigenvalues of x accidently negative).
        u, s, v = torch.svd(x)
        s_notneg = torch.zeros_like(s)
        for i in range(s.shape[0]):
            if s[i] > eps:
                s_notneg[i] = s[i]
        print(u.shape, s_notneg.shape, torch.transpose(v, 0, 1).shape)
        return torch.dot(u, torch.dot(torch.diag(torch.sqrt(s_notneg)), torch.transpose(v, 0, 1)))
    
    def unscented(self, mu, logvar):
        #For a vector mu of length N with covariance matrix logvar,
        #form 2N sigma points used for taking the unscented transform.
        #mu = mu.view(-1) # Force shape
        bs = mu.shape[0]
        N = mu.shape[1]
        #scale = 1.0
        #varsqrt = scale * self.svdsqrtm(N * logvar)
        logvar_diag = torch.zeros(bs, N, N).to(device)
        for i in range(logvar.shape[0]):
            logvar_diag[i, :] = torch.diag(logvar[i, :])
        varsqrt = math.sqrt(N)*logvar_diag
        x_sigma = []
        
        for i in range(N):
            x_sigma.append(mu + varsqrt[:, i])

        for i in range(N):
            x_sigma.append(mu - varsqrt[:, i])

        return x_sigma
    
    def batch_det(self, x, var):
        k = x.shape[1]
        bs = x.shape[0]
        Sigma = torch.zeros(bs, k, k).to(device)
        for i in range(var.shape[0]):
            Sigma[i, :] = torch.diag(var[i, :])
        
        return Sigma
    
    def norm_dist_exp(self, x, mu, var):
        Sigma = self.batch_det(x, var)
        return torch.squeeze((-1/2)*torch.sum(torch.log(var), dim=1)
                             -(1/2)*torch.bmm(torch.bmm(torch.transpose((x - mu).unsqueeze(-1), 1, 2), torch.inverse(Sigma)), (x - mu).unsqueeze(-1)))
    
    def norm_dist(self, x, mu, var, max_x):
        exp_norm = self.norm_dist_exp(x, mu, var)
        #Sigma = self.batch_det(x, var)
        #sqrt_det = torch.sqrt(torch.det(Sigma))
        diff = exp_norm - max_x
        
        return torch.exp(diff), diff
    
    def UT_sample_loss(self, x, z, mu_z, var_z):
        K = len(z)       
        bs = x.shape[0]
        x_exps = []
        z_exps = []
        means_x = []
        vars_x = []
        with torch.no_grad():
            for sample in z:
                mu_x, var_x = self.decode(sample)
                means_x.append(mu_x)
                vars_x.append(var_x)
                x_exp = self.norm_dist_exp(x, mu_x, var_x)
                z_exp = self.norm_dist_exp(sample, torch.zeros(bs, sample.shape[1]).to(device), torch.ones(bs, sample.shape[1]).to(device))
                x_exps.append(x_exp)
                z_exps.append(z_exp)
        
        x_exps_tensor = torch.cat(x_exps, dim=1).to(device)
        z_exps_tensor = torch.cat(z_exps, dim=1).to(device)
        print(x_exps_tensor.shape, z_exps_tensor.shape)
        x_exps_max = torch.max(x_exps_tensor, dim=1)[0]
        z_exps_max = torch.max(z_exps_tensor, dim=1)[0]
        print(x_exps_max.shape, z_exps_max.shape)
        
        pq_sum_tensor = torch.zeros(bs).to(device)
        
        with torch.no_grad():
            for inx, sample in enumerate(z):
                mu_x = means_x[inx]
                var_x = vars_x[inx]
                #q_z_x = self.norm_dist_exp(sample, mu_z, var_z)
                p_x_z, diff_x = self.norm_dist(x, mu_x, var_x, x_exps_max)
                p_z, diff_z = self.norm_dist(sample, torch.zeros(bs, sample.shape[1]).to(device), torch.ones(bs, sample.shape[1]).to(device), z_exps_max)
                diff = diff_x + diff_z
                pq_sum = p_x_z*p_z
                print(pq_sum.shape, diff.shape)
                print(diff)
                big_pq = torch.zeros_like(pq_sum).to(device)
                for i in range(bs):
                    if diff[i] >= -10:
                        big_pq[i] = pq_sum[i]
                pq_sum_tensor += big_pq

            #pq_sum_tensor = torch.cat(pq_sum, dim=1).to(device)
            #pq_sum_tensor = torch.squeeze(pq_sum_tensor)
            
            C = torch.ones(bs).to(device)
            C.new_full((bs,), (-(x.shape[1])/2)*math.log(2*math.pi))
            print(pq_sum_tensor)
            
            #return -(C + torch.log((1/K)*torch.sum(pq_sum_tensor, dim=1)))
            return torch.sum(-(C + x_exps_max + z_exps_max + torch.log((1/K)*pq_sum_tensor)))
    
    def sample_loss(self, x, mu_z, var_z):
        """
        k = mu_z.shape[1]
        bs = x.shape[0]
        Epsilon = torch.zeros(bs, k, k).to(device)
        for i in range(var_z.shape[0]):
            Epsilon[i, :] = torch.diag(var_z[i, :])
        
        #Epsilon = torch.diag(var_z)
        print(Epsilon.shape)
        dist_z = MultivariateNormal(mu_z, Epsilon)
        z = dist_z.sample_n(bs)
        """
        z = self.reparameterize(mu_z, var_z)
        max_x = torch.max(x, dim=1)[0]
        bs = x.shape[0]
        
        with torch.no_grad():
            mu_x, var_x = self.decode(z)
            q_z_x = self.norm_dist_exp(z, mu_z, var_z)
            p_x_z = self.norm_dist_exp(x, mu_x, var_x)
            p_z = self.norm_dist_exp(z, torch.zeros(bs, z.shape[1]).to(device), torch.ones(bs, z.shape[1]).to(device))
            
            pq = (p_x_z*p_z)/q_z_x
            
            C = torch.ones(bs).to(device)
            C.new_full((bs,), (-(x.shape[1])/2)*math.log(2*math.pi))
            
            return torch.sum(-(C + max_x + torch.log(pq)))
            #return torch.sum(-(C + max_x + torch.log(pq)))
        

        
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
  

    def forward(self, args, x):
        mu, logvar = self.encode(x.view(-1, 784))
        #z = self.reparameterize(mu, logvar)
        z = self.unscented(mu, logvar)
        for sample in z:
            recon_x = self.decode(sample)
        return recon_x, mu, logvar, z


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


def train(args, epoch, istrain=True):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        mu, logvar = model.encode(data.view(-1, 784))
        z = model.unscented(mu, logvar)
        for sample in z:
            #recon_batch = model.decode(sample)
            #recon_batch, mu, logvar = model(args, data)
            loss = model.UT_sample_loss(recon_batch, z, mu, logvar)
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
    UT_test_loss = torch.zeros(args.batch_size)
    test_loss = torch.zeros(args.batch_size)
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            mu, logvar = model.encode(data.view(-1, 784))
            z = model.unscented(mu, logvar)
            #recon_batch, mu, logvar = model(args, data)
            UT_test_loss += model.UT_sample_loss(data.view(-1, 784), z, mu, logvar)
            test_loss += model.sample_loss(data.view(-1, 784), mu, logvar)
            #if i == 0:
               # n = min(data.size(0), 8)
                #comparison = torch.cat([data[:n],
                                      #recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                #save_image(comparison.cpu(),
                         #'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    
    UT_test_loss /= len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss with reparameterization trick: {:.4f}'.format(test_loss))
    print('====> Test set loss with UT: {:.4f}'.format(UT_test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        #train(args, epoch)
        test(args, epoch)
        """
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
        """
