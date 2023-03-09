import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from skimage.io import imread, imsave

class VolumeRenderer(nn.Module):
    def __init__(self, sr=1.0, res=512, projection='perspective'):
        self.sr = sr
        self.res = res
        self.camera = torch.FloatTensor([3.0,3.0,3.0])
        self.center = torch.FloatTensor([0.0,0.0,0.0])
        self.up = torch.FloatTensor([0,1,0])
        self.projection = projection

    def get_rays(self):
        camera_direction = (self.center - self.camera) / torch.linalg.norm(self.center - self.camera)
        if self.projection == 'perspective':
            screen = self.camera + camera_direction
        else:
            screen = self.camera
        screen_right = torch.cross(camera_direction, self.up)
        screen_right = screen_right / torch.linalg.norm(screen_right)
        screen_up = torch.cross(screen_right, camera_direction)
        screen_up = screen_up / torch.linalg.norm(screen_up)
        xs = torch.linspace(-0.5, 0.5, steps=self.res)
        ys = torch.linspace(0.5, -0.5, steps=self.res)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        self.save_image('/mnt/g/vis_data/x.png',x+0.5)
        self.save_image('/mnt/g/vis_data/y.png',y+0.5)
        print((x.unsqueeze(-1)*screen_right).size())
        ray_targets = screen.expand(1,1,3).repeat(self.res,self.res,1) + x.unsqueeze(-1)*screen_right+y.unsqueeze(-1)*screen_up
        if self.projection == 'perspective':
            ray_origins = torch.FloatTensor(self.camera).expand(1,1,3).repeat(self.res,self.res,1)
            ray_directions = (ray_targets - ray_origins) / torch.linalg.norm((ray_targets - ray_origins),dim=-1,keepdim=True)
        else:
            ray_origins = ray_targets
            ray_directions = camera_direction.expand(1,1,3).repeat(self.res,self.res,1)


        return ray_origins, ray_directions
    
    def render(self, data):
        ray_origins, ray_directions = self.get_rays()
        ray_origins = ray_origins.cuda()
        ray_directions = ray_directions.cuda()

        inv_ray_directions = 1.0 / ray_directions
        dx = torch.abs(inv_ray_directions[:,:,0]/128.0)
        dy = torch.abs(inv_ray_directions[:,:,1]/128.0)
        dz = torch.abs(inv_ray_directions[:,:,2]/128.0)
        dt = torch.cat((dx.unsqueeze(-1),dy.unsqueeze(-1),dz.unsqueeze(-1)),dim=-1)
        dt, _ = torch.min(dt,dim=-1,keepdim=True)
        print(dt.min(),dt.max())

        rgbs = torch.zeros((self.res,self.res,1)).cuda()
        alpha = torch.zeros((self.res,self.res,1)).cuda()

        s = torch.zeros((self.res,self.res,1)).cuda()
        while s.max() < 10:
            sample_coordinates = ray_origins + ray_directions * s
            values = self.sample_from_3dgrid(data, sample_coordinates)
            if rgbs.sum() != 0 and values.max() == 0:
                break
            density = -(alpha-1.0) * values
            rgbs += values * density * (values > 0.1)
            alpha += density * (values > 0.1)
            alpha = alpha.clip(0,1)
            s += dt

        return rgbs

    def save_image(self, path, img):
        img = img.cpu().numpy()
        img = img * 255
        img = img.astype(np.uint8)
        imsave(path, img)
        

    def sample_from_3dgrid(self, grid, coordinates):
        """
        Expects coordinates in shape (batch_size, num_points_per_batch, 3)
        Expects grid in shape (1, channels, H, W, D)
        (Also works if grid has batch size)
        Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
        """
        batch_size, n_coords, n_dims = coordinates.shape
        sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                        coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                        mode='bilinear', padding_mode='zeros', align_corners=False)
        N, C, H, W, D = sampled_features.shape
        sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
        return sampled_features


def transferFunction(x):
    r = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
    g = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  1.0*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
    b = 0.1*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  1.0*np.exp( -(x - -3.0)**2/0.5 )
    a = 0.6*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) + 0.01*np.exp( -(x - -3.0)**2/0.5 )

    return r,g,b,a

# data = np.fromfile('/mnt/g/vis_data/vorts/global_normalized/vorts-0001.iw', dtype = '<f')
# data = data.reshape(128,128,128)

# img = np.zeros((128,128,3))


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [batch_size, nb_bins, 3]
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)  # Pixel values
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)

data = np.fromfile('/mnt/g/vis_data/vorts/global_normalized/vorts-0001.iw', dtype = '<f')
# data = np.fromfile('/mnt/g/vis_data/global_normalized/tornado-0001.iw', dtype = '<f')
data = data.reshape(128,128,128)
data = torch.FloatTensor(data).unsqueeze(0).unsqueeze(0)
data = data.cuda()

renderer = VolumeRenderer()
img = renderer.render(data)
renderer.save_image('/mnt/g/vis_data/test.png',img)
exit()

res = 512

eye = np.array([1.5,1.5,1.5])
# eye = np.array([0.5,0.5,2.0])
center = np.array([0.5,0.5,0.5])
screen = (center - eye) / np.linalg.norm(center - eye) + eye
print(screen)
dir = (center-eye) / np.linalg.norm(center - eye)
screen_right = np.cross(dir, np.array([0,1,0]))
screen_right = screen_right / np.linalg.norm(screen_right)
screen_up = np.cross(screen_right, dir)
screen_up = screen_up / np.linalg.norm(screen_up)

screen_right = torch.FloatTensor(screen_right)
screen_up = torch.FloatTensor(screen_up)
screen = torch.FloatTensor(screen)
ray_origins = torch.FloatTensor(eye).expand(1,1,3).repeat(res,res,1)
# print(ray_origins)
xs = torch.linspace(-1.5, 1.5, steps=res)
ys = torch.linspace(1.5, -1.5, steps=res)
x, y = torch.meshgrid(xs, ys, indexing='xy')
print(x.size())
z = torch.ones((res,res)) * 2.5
grid = torch.cat((x.unsqueeze(-1),y.unsqueeze(-1),z.unsqueeze(-1)),dim=-1)
ray_targets = screen.expand(1,1,3).repeat(res,res,1) + x.unsqueeze(-1)*screen_right+y.unsqueeze(-1)*screen_up
ray_directions = (ray_targets - ray_origins) / torch.linalg.norm((ray_targets - ray_origins),dim=-1,keepdim=True)
# ray_directions = torch.FloatTensor((0,0,-1.0)).expand(1,1,3).repeat(512,512,1)
rgbs = torch.zeros((res,res,1)).cuda()
alpha = torch.zeros((res,res,1)).cuda()

data = data.cuda()
ray_origins = ray_origins.cuda()
ray_directions = ray_directions.cuda()

inv_ray_directions = 1.0 / ray_directions
dx = torch.abs(inv_ray_directions[:,:,0]/128.0)
dy = torch.abs(inv_ray_directions[:,:,1]/128.0)
dz = torch.abs(inv_ray_directions[:,:,2]/128.0)
dt = torch.cat((dx.unsqueeze(-1),dy.unsqueeze(-1),dz.unsqueeze(-1)),dim=-1)
dt, _ = torch.min(dt,dim=-1,keepdim=True)
print(dt.min(),dt.max())

s = torch.zeros((res,res,1)).cuda()
while s.max() < 5:
    sample_coordinates = ray_origins + ray_directions * s
    values = sample_from_3dgrid(data, sample_coordinates)
    if rgbs.sum() != 0 and values.max() == 0:
        break
    density = -(alpha-1.0) * values
    rgbs += values * density * (values > 0.3)
    alpha += density * (values > 0.3)
    alpha = alpha.clip(0,1)
    s += dt

print(alpha.mean())

img = rgbs.cpu().numpy()
img = img * 255
img = img.astype(np.uint8)
imsave('/mnt/g/vis_data/test.png',img)