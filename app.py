import argparse
import torch
import yaml
import sys, os

sys.path.append('model/networkbending')


from torchvision import utils
from .model import Generator
from tqdm import tqdm
from util import *

import datetime

MODEL_PATH ='model/networkbending/'

#For GANSpace
from sklearn.decomposition import PCA



class API():
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--size', type=int, default=1024)
        parser.add_argument('--sample', type=int, default=1)
        parser.add_argument('--pics', type=int, default=1)
        parser.add_argument('--truncation', type=float, default=0.5)
        parser.add_argument('--truncation_mean', type=int, default=4096)
        parser.add_argument('--ckpt', type=str, default=MODEL_PATH+"models/stylegan2-ffhq-config-f.pt")
        parser.add_argument('--channel_multiplier', type=int, default=2)
        parser.add_argument('--config', type=str, default=MODEL_PATH+"configs/empty_transform_config.yaml")
        parser.add_argument('--load_latent', type=str, default="") 
        parser.add_argument('--clusters', type=str, default=MODEL_PATH+"configs/example_cluster_dict.yaml")

        self.args = parser.parse_args(args=[])


device = 'cpu'


class NetworkBending():
    def __init__(self):
        print("loading network bending app")
        
        api = API()
        self.args = api.args
        
        self.args.latent = 512
        self.args.n_mlp = 8
        
        g_ema = Generator(
            self.args.size, self.args.latent, self.args.n_mlp, channel_multiplier=self.args.channel_multiplier
        ).to(device)
        new_state_dict = g_ema.state_dict()
        
        #model load
        ext_state_dict  = torch.load(self.args.ckpt)['g_ema']
        
        new_state_dict.update(ext_state_dict)
        g_ema.load_state_dict(new_state_dict)
        g_ema.eval()
        g_ema.to(device)
        
        
        #load network parameter
        yaml_config = {}
        with open(self.args.config, 'r') as stream:
            try:
                yaml_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        cluster_config = {}
        if self.args.clusters != "":
            with open(self.args.clusters, 'r') as stream:
                try:
                    cluster_config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
        
        
        # checkpoint = torch.load(args.ckpt)
        mean_latent = g_ema.mean_latent(self.args.truncation_mean)
        
        layer_channel_dims = create_layer_channel_dim_dict(self.args.channel_multiplier)
        
        #generate test
        # self.generate(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims)
        
        self.yaml_config = yaml_config
        self.cluster_config = cluster_config
        self.layer_channel_dims = layer_channel_dims
        
        self.sample_z = torch.randn(self.args.sample, self.args.latent, device=device)
        self.g_ema = g_ema
        self.mean_latent = mean_latent
        
        # Call this if there is no ganspace data
        # GEN_SAMPLES = 200
        # self.ganSpace(GEN_SAMPLES)


    
    def random_sampling(self):
          args = self.args
          g_ema = self.g_ema
          mean_latent = self.mean_latent
          
          yaml_config = self.yaml_config
          cluster_config = self.cluster_config
          layer_channel_dims = self.layer_channel_dims
          current_time = datetime.datetime.now().strftime('%m%d%H%M%S')  

          
          with torch.no_grad():
                g_ema.eval()
                t_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)
                for i in tqdm(range(args.pics)):
                    sample_z = torch.randn(args.sample, args.latent, device=device)
                    print(sample_z.shape)
                    print(device)

                    sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent, transform_dict_list=t_dict_list)

                    if not os.path.exists('sample'):
                        os.makedirs('sample')

                    path =  f'server_data/{current_time}.png'
                    utils.save_image(
                        sample,
                        path,
                        nrow=1,
                        normalize=True,
                        range=(-1, 1))
        
                    return path
                        
    
    def w_sampling(self,sample_w):
          args = self.args
          g_ema = self.g_ema
          mean_latent = self.mean_latent
          
          yaml_config = self.yaml_config
          cluster_config = self.cluster_config
          layer_channel_dims = self.layer_channel_dims
          current_time = datetime.datetime.now().strftime('%m%d%H%M%S')  

          
          with torch.no_grad():
                g_ema.eval()
                t_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)
                for i in tqdm(range(args.pics)):
                    sample_w = sample_w.unsqueeze(0)
                    sample, _ = g_ema([sample_w], truncation=args.truncation, truncation_latent=mean_latent, transform_dict_list=t_dict_list)

                    if not os.path.exists('sample'):
                        os.makedirs('sample')

                    path =  f'server_data/ganspace/{current_time}.png'
                    utils.save_image(
                        sample,
                        path,
                        nrow=1,
                        normalize=True,
                        range=(-1, 1))
                    
                    return path
        
                
    #private function
    def ganSpace(self, GEN_SAMPLES):
        ''''
        Making GANSpace data
        '''
        args = self.args
        g_ema = self.g_ema
        mean_latent = self.mean_latent
        
        
        sample_z = torch.randn(GEN_SAMPLES, args.latent, device=device)
        
        sample_w = g_ema.get_latent(sample_z)
        
        # self.w_sampling(sample_w[0])
        
        torch.save(sample_w,'./server_data/ganspace/ganspace_w_data.pt')    
        
        args = self.args
        g_ema = self.g_ema
        mean_latent = self.mean_latent
        
        yaml_config = self.yaml_config
        cluster_config = self.cluster_config
        layer_channel_dims = self.layer_channel_dims
        
        with torch.no_grad():
            g_ema.eval()
            t_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)
            for i in tqdm(range(sample_w.size(0))):
                sample_input = sample_w[i].unsqueeze(0)
                sample, _ = g_ema([sample_input], truncation=args.truncation, truncation_latent=mean_latent, transform_dict_list=t_dict_list)

                if not os.path.exists('sample'):
                    os.makedirs('sample')

                path =  f'server_data/ganspace/img/{i}.png'
                utils.save_image(
                    sample,
                    path,
                    nrow=1,
                    normalize=True,
                    range=(-1, 1))
        
        
        
        sample_w =sample_w.detach().cpu().numpy()
        pca = PCA(n_components=2)
        print('train start')
        pca.fit(sample_w)
        projection_data = pca.transform(sample_w)
        torch.save(projection_data,'./server_data/ganspace/ganspace_2d_pos.pt')
        print('train end')
        pcomponents = torch.tensor(pca.components_) #ToDO: need to consider
    
    
    
 
    
    
    def genAPI(self,dilate):
        
        # print(self.yaml_config['transforms'][3]['params'])
        self.yaml_config['transforms'][3]['params'] = [dilate]
        
        t_dict_list = create_transforms_dict_list(self.yaml_config, self.cluster_config, self.layer_channel_dims)
        sample, _ = self.g_ema([self.sample_z], truncation=0.5, truncation_latent=self.mean_latent, transform_dict_list=t_dict_list)
        current_time = datetime.datetime.now().strftime('%m%d%H%M%S')  

        if not os.path.exists('sample'):
                    os.makedirs('sample')

        path =  f'server_data/{current_time}.png'
        utils.save_image(
            sample,
            path,
            nrow=1,
            normalize=True,
            range=(-1, 1))
        
        return path


    
    def generate(self,args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims):
        with torch.no_grad():
            g_ema.eval()
            t_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)
            for i in tqdm(range(args.pics)):
                sample_z = torch.randn(args.sample, args.latent, device=device)
                print(sample_z.size())
                sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent, transform_dict_list=t_dict_list)

                if not os.path.exists('sample'):
                    os.makedirs('sample')

                utils.save_image(
                    sample,
                    f'server_data/{str(i).zfill(6)}.png',
                    nrow=1,
                    normalize=True,
                    range=(-1, 1))
        
        
    
    def generate_from_latent(selfargs, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims, latent, noises):
        with torch.no_grad():
            g_ema.eval()
            slice_latent = latent[0,:]
            slce_latent = slice_latent.unsqueeze(0)
            print(slice_latent.size())
            for i in tqdm(range(args.pics)):
                t_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)
                sample, _ = g_ema([slce_latent], input_is_latent=True, noise=noises, transform_dict_list=t_dict_list)

                if not os.path.exists('sample'):
                    os.makedirs('sample')

                utils.save_image(
                    sample,
                    f'sample/{str(i).zfill(6)}.png',
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )


# if __name__ == '__main__':
#     test = NetworkBending()