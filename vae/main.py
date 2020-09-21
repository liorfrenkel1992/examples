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
        self.fc4 = nn.Linear(400, 784)
        #self.fc41 = nn.Linear(400, 784)
        #self.fc42 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) 
        return mu + eps*std

    def decode(self, z, istrain=True):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
        #return self.fc41(h3), self.fc42(h3)
      
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
        var = torch.exp(logvar)
        var_diag = torch.zeros(bs, N, N).to(device)
        for i in range(var.shape[0]):
            var_diag[i, :] = torch.diag(var[i, :])
        varsqrt = torch.sqrt(N*var_diag)
        x_sigma = []
        
        for i in range(N):
            x_sigma.append(mu + varsqrt[:, i])

        for i in range(N):
            x_sigma.append(mu - varsqrt[:, i])

        return x_sigma
    
    def batch_diag(self, x, var):
        k = x.shape[1]
        bs = x.shape[0]
        Sigma = torch.zeros(bs, k, k).to(device)
        for i in range(var.shape[0]):
            Sigma[i, :] = torch.diag(var[i, :])
        
        return Sigma
    
    def norm_dist_exp(self, x, mu, var):
        Sigma = self.batch_diag(x, var)
        exp1 = torch.squeeze((-1/2)*torch.sum(torch.log(var), dim=1))
        exp2 = torch.squeeze(-(1/2)*torch.bmm(torch.bmm(torch.transpose((x - mu).unsqueeze(-1), 1, 2), torch.inverse(Sigma)), (x - mu).unsqueeze(-1)))
        return exp1 + exp2
    
    def norm_dist(self, x, mu, var, max_x):
        exp_norm = self.norm_dist_exp(x, mu, var)
        #Sigma = self.batch_det(x, var)
        #sqrt_det = torch.sqrt(torch.det(Sigma))
        diff = exp_norm - max_x
        
        return torch.exp(diff), diff
    
    def UT_sample_loss(self, x, z, mu_z, logvar_z):
        K = len(z)       
        bs = x.shape[0]
        var_z = torch.exp(logvar_z)
        x_exps = []
        z1_exps = []
        z2_exps = []
        means_x = []
        vars_x = []
        for sample in z:
            mu_x, logvar_x = self.decode(sample)
            var_x = torch.exp(logvar_x)
            means_x.append(mu_x)
            vars_x.append(var_x)
            x_exp = self.norm_dist_exp(x, mu_x, var_x)
            z1_exp = self.norm_dist_exp(sample, torch.zeros(bs, sample.shape[1]).to(device), torch.ones(bs, sample.shape[1]).to(device))
            z2_exp = self.norm_dist_exp(sample, mu_z, var_z)
            x_exps.append(x_exp.unsqueeze(-1))
            z1_exps.append(z1_exp.unsqueeze(-1))
            z2_exps.append(z2_exp.unsqueeze(-1))
        
        x_exps_tensor = torch.cat(x_exps, dim=1).to(device)
        z1_exps_tensor = torch.cat(z1_exps, dim=1).to(device)
        z2_exps_tensor = torch.cat(z2_exps, dim=1).to(device)
        x_exps_max = torch.max(x_exps_tensor, dim=1)[0]
        z1_exps_max = torch.max(z1_exps_tensor, dim=1)[0]
        z2_exps_max = torch.max(z2_exps_tensor, dim=1)[0]

        pq_sum_tensor = torch.zeros(bs).to(device)
        
        for inx, sample in enumerate(z):
            mu_x = means_x[inx]
            var_x = vars_x[inx]
            #q_z_x = self.norm_dist_exp(sample, mu_z, var_z)
            p_x_z, diff_x = self.norm_dist(x, mu_x, var_x, x_exps_max)
            p_z, diff_z1 = self.norm_dist(sample, torch.zeros(bs, sample.shape[1]).to(device), torch.ones(bs, sample.shape[1]).to(device), z1_exps_max)
            q_z_x, diff_z2 = self.norm_dist(sample, mu_z, var_z, z2_exps_max)
            diff = diff_x + diff_z1 - diff_z2
            pq_sum = (p_x_z*p_z)/q_z_x
            big_pq = torch.zeros_like(pq_sum).to(device)
            for i in range(bs):
                if diff[i] >= -10:
                    big_pq[i] = pq_sum[i]
            pq_sum_tensor += big_pq
            
        #C = torch.ones(bs).to(device)
        #C.new_full((bs,), (-(x.shape[1])/2)*math.log(2*math.pi))
        C = (-x.shape[1]/2)*math.log(2*math.pi)
        #D = (1/2)*(torch.sum(logvar_z, dim=1) + logvar_z.shape[1])
        
        return -(C + x_exps_max + z1_exps_max - z2_exps_max + torch.log((1/K)*pq_sum_tensor))
        #return C + D + x_exps_max + z_exps_max + torch.log((1/K)*pq_sum_tensor)
    
    def sample_loss(self, x, mu_z, logvar_z, num_samples):
        z = []
        bs = x.shape[0]
        var_z = torch.exp(logvar_z)
        Sigma = self.batch_diag(mu_z, var_z)
        
        dist_z = MultivariateNormal(mu_z, Sigma)
        for i in range(num_samples):
            z.append(dist_z.sample())
        
        K = len(z)       
        x_exps = []
        z1_exps = []
        z2_exps = []
        means_x = []
        vars_x = []
        
        for sample in z:
            mu_x, logvar_x = self.decode(sample)
            var_x = torch.exp(logvar_x)
            means_x.append(mu_x)
            vars_x.append(var_x)
            x_exp = self.norm_dist_exp(x, mu_x, var_x)
            z1_exp = self.norm_dist_exp(sample, torch.zeros(bs, sample.shape[1]).to(device), torch.ones(bs, sample.shape[1]).to(device))
            z2_exp = self.norm_dist_exp(sample, mu_z, var_z)
            x_exps.append(x_exp.unsqueeze(-1))
            z1_exps.append(z1_exp.unsqueeze(-1))
            z2_exps.append(z2_exp.unsqueeze(-1))
        
        x_exps_tensor = torch.cat(x_exps, dim=1).to(device)
        z1_exps_tensor = torch.cat(z1_exps, dim=1).to(device)
        z2_exps_tensor = torch.cat(z2_exps, dim=1).to(device)
        x_exps_max = torch.max(x_exps_tensor, dim=1)[0]
        z1_exps_max = torch.max(z1_exps_tensor, dim=1)[0]
        z2_exps_max = torch.max(z2_exps_tensor, dim=1)[0]
                       
        pq_sum_tensor = torch.zeros(bs).to(device)
        
        for inx, sample in enumerate(z):
            mu_x = means_x[inx]
            var_x = vars_x[inx]
            p_x_z, diff_x = self.norm_dist(x, mu_x, var_x, x_exps_max)
            p_z, diff_z1 = self.norm_dist(sample, torch.zeros(bs, sample.shape[1]).to(device), torch.ones(bs, sample.shape[1]).to(device), z1_exps_max)
            q_z_x, diff_z2 = self.norm_dist(sample, mu_z, var_z, z2_exps_max)
            diff = diff_x + diff_z1 - diff_z2
            pq_sum = (p_x_z*p_z)/q_z_x
            big_pq = torch.zeros_like(pq_sum).to(device)
            for i in range(bs):
                if diff[i] >= -10:
                    big_pq[i] = pq_sum[i]
            pq_sum_tensor += big_pq
            
        #C = torch.ones(bs).to(device)
        #C.new_full((bs,), (-(x.shape[1])/2)*math.log(2*math.pi))
        C = (-x.shape[1]/2)*math.log(2*math.pi)
            
        return -(C + x_exps_max + z1_exps_max - z2_exps_max + torch.log((1/K)*pq_sum_tensor))
        

        
    def unscented_mu_cov(self, x_sigma):
        #Approximate mean, covariance from 2N sigma points transformed through
        #an arbitrary non-linear transformation.
        #Returns a flattened 1d array for x.
        N = len(x_sigma)
        x_sigma = [torch.unsqueeze(point, 1) for point in x_sigma]
        pts = torch.cat(x_sigma, dim=1).to(device)
        
        x_mu = torch.mean(pts, dim=1, keepdim=True)
        diff = pts - x_mu
        x_cov = torch.dot(torch.transpose(diff[0], 1, 2), diff) / N
        return x_mu, x_cov
  

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        #z = self.reparameterize(mu, logvar)
        #z = self.unscented(mu, logvar)
        
        bs = x.shape[0]
        var = torch.exp(logvar)
        Sigma = self.batch_diag(mu, var)
        
        dist_z = MultivariateNormal(mu, Sigma)
        z = dist_z.sample()
        
        recon_x = self.decode(z)
        return recon_x, mu, logvar


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
    bs = args.batch_size
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        #mu, logvar = model.encode(data.view(-1, 784))
        #z = model.unscented(mu, logvar)
        #loss = (1/bs)*torch.sum(model.sample_loss(data.view(-1, 784), mu, logvar, 2*mu.shape[1]))
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
    #UT_test_loss = torch.zeros(args.batch_size).to(device)
    #test_loss = torch.zeros(args.batch_size).to(device)
    bs = args.batch_size
    true_loss = 0
    UT_loss = 0
    reg_loss = 0
    true_test_loss = 0
    UT_test_loss = 0
    reg_test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            print(data)
            #recon_batch, mu, logvar = model(data)
            mu, logvar = model.encode(data.view(-1, 784))
            z1 = model.unscented(mu, logvar)
            
            z2 = []
            var = torch.exp(logvar)
            Sigma = model.batch_diag(mu, var)
            dist_z = MultivariateNormal(mu, Sigma)
            for j in range(2*mu.shape[1]):
                z2.append(dist_z.sample())
            
            for inx1, sample1 in enumerate(z1):
                recon_batch1 = model.decode(sample1)
                UT_test_loss += loss_function(recon_batch1, data, mu, logvar).item()
            UT_test_loss /= len(z1)
            UT_loss += UT_test_loss
            print('UT loss: ', UT_test_loss/args.batch_size)
            UT_test_loss = 0
            for inx2, sample2 in enumerate(z2):
                recon_batch2 = model.decode(sample2)
                reg_test_loss += loss_function(recon_batch2, data, mu, logvar).item()
            reg_test_loss /= len(z2)
            reg_loss += reg_test_loss
            print('regular sampling loss: ', reg_test_loss/args.batch_size)
            reg_test_loss = 0
            
            z3 = []
            for j in range(10000):
                z3.append(dist_z.sample())
            for inx3, sample3 in enumerate(z3):
                recon_batch3 = model.decode(sample3)
                true_test_loss += loss_function(recon_batch3, data, mu, logvar).item()
            true_test_loss /= len(z3)
            true_loss += true_test_loss
            print('true sampling loss: ', true_test_loss/args.batch_size)
            true_test_loss = 0
            
            """
            UT_test_loss = (1/bs)*torch.sum(model.UT_sample_loss(data.view(-1, 784), z1, mu, logvar)).item()
            UT_loss += UT_test_loss
            print('UT score: ', UT_test_loss)
            UT_test_loss = 0
            reg_test_loss = (1/bs)*torch.sum(model.sample_loss(data.view(-1, 784), mu, logvar, 2*mu.shape[1])).item()
            reg_loss += reg_test_loss
            print('regular sampling score: ', reg_test_loss)
            reg_test_loss = 0
            true_test_loss = (1/bs)*torch.sum(model.sample_loss(data.view(-1, 784), mu, logvar, 10000)).item()
            true_loss += true_test_loss
            print('true sampling score: ', true_test_loss)
            true_test_loss = 0
            """
            #if i == 0:
               # n = min(data.size(0), 8)
                #comparison = torch.cat([data[:n],
                                      #recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                #save_image(comparison.cpu(),
                         #'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    UT_loss /= len(test_loader.dataset)
    reg_loss /= len(test_loader.dataset)
    true_loss /= len(test_loader.dataset)
    print('====> Test set loss with regular sampling: {:.4f}'.format(reg_loss))
    print('====> Test set loss with UT: {:.4f}'.format(UT_loss))
    print('====> True test set loss: {:.4f}'.format(true_loss))

if __name__ == "__main__":
    #for epoch in range(1, args.epochs + 1):
        #train(args, epoch)
    PATH = '/data/vae/results_regular.pth'
    #torch.save(model.state_dict(), PATH)
    model.load_state_dict(torch.load(PATH))
    test(args, 10)
    """
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
    """
