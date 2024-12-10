from scipy.stats import pearsonr,binom,linregress
import numpy as np
import argparse, os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.models import inception_v3
from math import log10, sqrt, exp     
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import pairwise_distances
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from PIL import Image
from einops import rearrange
from ldm.util import instantiate_from_config
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Resize
import shutil
from taming.modules.losses.lpips import LPIPS
import scipy as sp

def pairwise_corr_all(ground_truth, predictions):
    r = np.corrcoef(ground_truth, predictions)
    r = r[:len(ground_truth), len(ground_truth):]  
    congruents = np.diag(r)
    success = r < congruents
    success_cnt = np.sum(success, 0)
    perf = np.mean(success_cnt) / (len(ground_truth)-1)
    p = 1 - binom.cdf(perf*len(ground_truth)*(len(ground_truth)-1), len(ground_truth)*(len(ground_truth)-1), 0.5)
    
    return perf, p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        nargs="?",
        help="'Specify the mode (train, validation, test)",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default='C:/Users/Asus/Desktop/latent-diffusion/ldm-eval/',
        help="dir to write results to",
    )

    parser.add_argument(
        "--configs",
        type=str,
        nargs="?",
        default='C:/Users/Asus/Desktop/latent-diffusion/conf/news/fmri_to_img_vqmodel7.yaml',
        help="configs",
    )

    parser.add_argument(
        "--weights",
        type=str,
        nargs="?",
        default='f16_1024',
        help="configs",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="?",
        #default= "/media/handeg/Expansion/latent-diffusion/ldm-checkpoints/2023-07-14T15-43-17_act_config/checkpoints/epoch=000753.ckpt",
        default= "C:/Users/Asus/Desktop/latent-diffusion/conf/checkpoints/2024-08-04T16-37-42_fmri_to_img_vqmodel7/checkpoints/last.ckpt",
        help="checkpoint",
        )
    
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        nargs="?",
        help="batchsize",
    )
    ######################################################################################################################
    parser.add_argument(
        "--dataroot",
        type=str,
        default='D:/bold5000/derivatives/fmriprep/',
        nargs="?",
        help="dir containing BOLD5000 (`which including folders named subject name: BOLD5000/ - CSI1,CSI2,CSI3,CSI4`)",
    )
    parser.add_argument(
        "--stimroot",
        type=str,
        default='C:/Users/Asus/Desktop/latent-diffusion/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/',
        nargs="?",
        help="dir containing stimulus images (`which including folders named stimulus dataset name: Presented_Stimuli/ - ImageNet, COCO, Scenes`)",
    )
    parser.add_argument(
        "--subject",
        type=list,
        default=["CSI1"],
        nargs="?",
        help="Specify the subject names",
    )
    parser.add_argument(
        "--img_dim",
        type=int,
        default=256,
        nargs="?",
        help="Specify the image dimension (optional)",
    )
    parser.add_argument(
        "--scene_stim",
        type=str,
        default= r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Stimuli_Presentation_Lists',
        nargs="?",
    )
    parser.add_argument(
        "--single_sub",
        type=str,
        default= True,
        nargs="?",
        help = "If number of subject is 1 please assign True this flag"
    )
    parser.add_argument(
        "--subdata",
        type=list,
        #default=["ImageNet","COCO", "Scenes"],
        default=["ImageNet"],
        nargs="?",
        help="Specify the subdata(ImageNet, COCO, Scenes)",
    )
    parser.add_argument(
        "--stimcheckpoint",
        type=str,
        nargs="?",
        default= "C:/Users/Asus/Desktop/latent-diffusion/conf/checkpoints/2024-07-31T22-50-56_img_to_img_vqmodel7/checkpoints/last.ckpt",
        help="stimcheckpoint",
    )
    parser.add_argument(
        "--volume_data",
        default=False,
    )
    parser.add_argument(
        "--stimconfigs",
        type=str,
        nargs="?",
        default='C:/Users/Asus/Desktop/latent-diffusion/conf/news/img_to_img_vqmodel7.yaml',
        help="stimconfigs",
    )
            
    opt = parser.parse_args()
    number = 8

    if opt.volume_data: 
      from ldm.data.volactivity import activityTrain, activityValidation, activityTest
    else:
      from ldm.data.activity import activityTrain, activityValidation, activityTest
      
    if opt.mode == "train":
        dataset = activityTrain(one_sub = opt.single_sub, ids = opt.subject, size=opt.img_dim)
    elif opt.mode == "validation":
        dataset = activityValidation(one_sub = opt.single_sub, ids = opt.subject, subDataset=opt.subdata, size=opt.img_dim)
    elif opt.mode == "test":
        dataset = activityTest(one_sub = opt.single_sub, ids = opt.subject, subDataset=opt.subdata, size=opt.img_dim)
    else:
        raise NotImplementedError
    
    dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=False)

    config = OmegaConf.load(opt.configs)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.checkpoint, map_location="cpu")["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    def imagenet_model_encode(x):
        config = OmegaConf.load(opt.stimconfigs)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(opt.stimcheckpoint,
                                         map_location="cpu")["state_dict"], strict=False)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        model.eval()
        z,_,[_,_,indices] = model.encode(x)
        print(f'VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}')
        return z
    
    eval_folder = opt.checkpoint.split("/")[-3]
    
    outdir = os.path.join(opt.outdir,eval_folder)

    if os.path.exists(outdir):
        shutil.rmtree(outdir)
        print(f"Folder '{outdir}' deleted successfully.")
    else:
        print(f"Folder '{outdir}' does not exist.")
    
    os.makedirs(outdir, exist_ok=True)

    results_df = []
    
    perceptual_loss = LPIPS().eval()
    
    class SwAVModel(nn.Module):
        def __init__(self, pretrained=True):
            super(SwAVModel, self).__init__()
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Identity()  
    
        def forward(self, x):
            return self.backbone(x)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    swaw_model = SwAVModel().cuda()
    swaw_model.eval()
    
    def preprocess_image(image):
        image = transform(image).unsqueeze(0).cuda()
        return image
    
    def extract_features(model, image):
        with torch.no_grad():
            features = model(image)
        return features.cpu().numpy().flatten()
    
    def calculate_swav_metric(original_image, reconstructed_image):        
        original_image_tensor = preprocess_image(original_image)
        original_features = extract_features(swaw_model, original_image_tensor)
        
        reconstructed_image_tensor = preprocess_image(reconstructed_image)
        reconstructed_features = extract_features(swaw_model, reconstructed_image_tensor)
        distance = pairwise_distances(original_features.reshape(1, -1), reconstructed_features.reshape(1, -1), metric='euclidean')
        swav_scores = distance[0][0]
        return swav_scores
    
    def pearson_correlation(img1, img2):
        """Compute the Pearson correlation coefficient between two normalized images.
        
        Args:
            img1 (np.ndarray): The first image array. Shape should be (H, W, C) or (N, H, W, C).
            img2 (np.ndarray): The second image array. Shape should be (H, W, C) or (N, H, W, C).
        
        Returns:
            float: The Pearson correlation coefficient.
        """
        assert img1.shape == img2.shape, "Input images must have the same dimensions"
        
        if img1.ndim == 4: 
            img1 = img1.reshape(img1.shape[0], -1)
            img2 = img2.reshape(img2.shape[0], -1)
        elif img1.ndim == 3:
            img1 = img1.flatten()
            img2 = img2.flatten()
        else:
            raise ValueError("Input images must be of shape (H, W, C) or (N, H, W, C)")
        
        mean1 = np.mean(img1, axis=-1, keepdims=True)
        mean2 = np.mean(img2, axis=-1,keepdims=True)
        diff1 = img1 - mean1[:, np.newaxis]
        diff2 = img2 - mean2[:, np.newaxis]
        numerator = np.sum(diff1 * diff2, axis=-1)
        denominator = np.sqrt(np.sum(diff1 ** 2, axis=-1) * np.sum(diff2 ** 2, axis=-1))
        correlation = numerator / denominator
        
        return np.mean(correlation)
    
    def pixcorr(image1,image2):
        image1_flat = image1.flatten()
        image2_flat = image2.flatten()
        mean1 = np.mean(image1_flat)
        mean2 = np.mean(image2_flat)
        
        covariance = np.mean((image1_flat - mean1) * (image2_flat - mean2))
        
        std1 = np.std(image1_flat)
        std2 = np.std(image2_flat)
        
        pixel_correlation_value = covariance / (std1 * std2)
        return np.abs(pixel_correlation_value)
        
    
    def PSNR(original, predicted, max_pixel = 1.0): 
        mse = np.mean((original - predicted) ** 2) 
        if(mse == 0):  
            return 100
        return 20 * log10(max_pixel / sqrt(mse))

    
    def get_unique_name(outdir, filename):
        file,extension = os.path.splitext(filename)
        base = file.split('/')[-1]  
        version = 1
        unique_name = filename
        
        while  os.path.exists(unique_name):
            unique_name = "{}_v{}{}".format(base, version, extension)
            version += 1
        return unique_name
    
    def chunk_list(lst, chunk_size):
        chunks = []
        for i in range(0, len(lst), chunk_size):
            chunks.append(lst[i:i + chunk_size])
        return chunks
    
    def ssim(image1, image2, K = [0.01, 0.03], L = 1, window_size = 11):
        image1 = rearrange(torch.Tensor(image1).unsqueeze(0), 'b h w c -> b c h w')
        image2 = rearrange(torch.Tensor(image2).unsqueeze(0), 'b h w c -> b c h w') 
        
        _, channel1, _, _ = image1.shape
        _, channel2, _, _ = image2.shape
        channel = min(channel1, channel2)
    
        sigma = 1.5   
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        _1D_window = (gauss/gauss.sum()).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
                
        C1 = K[0]**2;
        C2 = K[1]**2;
        
        mu1 = F.conv2d(image1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(image2, window, padding = window_size//2, groups = channel)
    
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
    
        sigma1_sq = F.conv2d(image1*image1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(image2*image2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(image1*image2, window, padding = window_size//2, groups = channel) - mu1_mu2
    
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
        
    perceptual_loss = LPIPS().eval()
    
    def split_psnr_values_to_excel(input_csv_path, output_excel_path):
        df = pd.read_csv(input_csv_path, header=None)
        new_data = []
        
        for index, row in df.iterrows():
            new_row = []
            for cell in row:
                if isinstance(cell, str):
                    values = cell.split(',')
                    for value in values:
                        try:
                            new_row.append(float(value))
                        except ValueError:
                            continue
                else:
                    new_row.append(cell)
            new_data.append(new_row)

        new_df = pd.DataFrame(new_data)
        cols = list(df.columns)
        new_df.index =df.iloc[:, 0].tolist()[1:] #df.columns
        new_df.columns = df.iloc[0].tolist()[1:] #df.index
        new_df.to_excel(output_excel_path, index=True)
    
    def reconstruct_with_vqgan(x,model):
        z,_,[_,_,indices] = model.encode(x)
        print(f'VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}')
        xrec = model.decode(z)
        return xrec
    
    def reconstruct_with_vqgan_encode(x,model):
        z,_,[_,_,indices] = model.encode(x)
        print(f'VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}')
        return z
    
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
        print(f"Folder '{outdir}' deleted successfully.")
    else:
        print(f"Folder '{outdir}' does not exist.")
        
    os.makedirs(outdir, exist_ok=True)
    
    t = 0
    origs = []
    preds = []
    stims = []
    
    with torch.no_grad():
        stim_images_dict = {}
        predicted_images_dict = {}
        
        psnr_dict = {}
        ssim_dict = {}
        pcc_dict = {}
        pixcorr_dict = {}
        percep_dict = {}
        inception_dict = {}
        swav_dict = {}
        pcc_valz_dict = {}

        for i, data in enumerate(dataloader):
            stim_image,init_image = model.get_input(data, "activity")
            stim_image = stim_image.to(device)

            if opt.volume_data:
                stim_image2 = rearrange(stim_image, 'b c s h w -> b s h w c')
            else:
                stim_image2 = rearrange(stim_image, 'b c h w -> b h w c')
            stim = (torch.clamp((stim_image2 + 1.0) / 2.0, min=0.0, max=1.0).cpu().detach().numpy()[0] * 255).astype(np.uint8)
            stim_image2 = stim_image2.cpu().detach().numpy()
            stims.append(stim_image2[0])
    
            init_image = init_image.to(device)
            
            psnr_values = []
            ssim_values = []
            pcc_values = []
            percep_values = []
            pixcorr_values = []
            incep_values = []
            swav_values = []
            pcc_val_z = []
            
            psnr_df = pd.DataFrame()
            ssim_df = pd.DataFrame()
            pcc_df = pd.DataFrame()
            pixcorr_df = pd.DataFrame()
            percep_df = pd.DataFrame()
            incep_df = pd.DataFrame()
            swav_df = pd.DataFrame()

            x_samples_ddim_list = []  
                        
            file_name = data['c_name'][0].split('/')[-1]
            print(file_name)
            orig_file = str(t) + '_real_' +file_name
            pred_file = str(t) + '_pred_' + file_name
        
            
            init_image2 = rearrange(init_image, 'b c h w -> b h w c')
            
            orig = init_image2[0].cpu().detach().numpy()
            orig = orig - np.min(orig)
            orig = orig / np.max(orig)
            
            grid = np.copy(orig)
            origs.append(orig)
            
            orig_z = imagenet_model_encode(init_image)
            orig_z = orig_z.cpu().detach().numpy()
                        
            for i in range(number):
                predicted_image = reconstruct_with_vqgan(stim_image,model)
                
                pred_z = reconstruct_with_vqgan_encode(stim_image, model)
                pred_z = pred_z.cpu().detach().numpy()
    
                predicted_image2 = rearrange(predicted_image, 'b c h w -> b h w c')
                x_samples_ddim_list.append(predicted_image2.cpu().detach().numpy())
                
                pred = predicted_image2.cpu().detach().numpy()[0]
                pred = pred - np.min(pred)
                pred = pred / np.max(pred)
                preds.append(pred)
                
                grid = np.concatenate((grid, pred), axis=1)

                
                psnr_values.append(PSNR(orig,pred))
                ssim_values.append(ssim(orig,pred))
                pcc_values.append(np.abs(pearson_correlation(orig,pred)))
                pixcorr_values.append(pixcorr(orig,pred))
                percep_values.append(perceptual_loss(rearrange(torch.Tensor(orig).unsqueeze(0), 'b h w c -> b c h w'),rearrange(torch.Tensor(pred).unsqueeze(0), 'b h w c -> b c h w')).item())
                incep_values.append(criterion.calculate_inception_score(rearrange(torch.Tensor(pred).unsqueeze(0), 'b h w c -> b c h w')[0]))
                swav_values.append(calculate_swav_metric(orig, pred))
                pcc_val_z.append(np.abs(pearson_correlation(orig_z,pred_z)))

                
            Image.fromarray(np.uint8(grid*255)).save(os.path.join(outdir, pred_file))
            
            ############# LATENT SPACE DISTRIBUTIONS ################
            # z_original = imagenet_model_encode(init_image)
            # z_predicted = reconstruct_with_vqgan_encode(init_image, model)
            
            # # Latent vektörleri numpy array'e çevirme
            # z_original_np = z_original.cpu().detach().numpy().reshape(-1, z_original.shape[-1])
            # z_predicted_np = z_predicted.cpu().detach().numpy().reshape(-1, z_predicted.shape[-1])
            
            # Latent vektörlerin dağılımını görselleştirme
            # visualize_latents(z_original_np, z_predicted_np, method='pca',orig_file=orig_file,pred_file=pred_file)
            # visualize_latents(z_original_np, z_predicted_np, method='tsne',orig_file=orig_file,pred_file=pred_file)
            # visualize_latents(z_original_np, z_predicted_np, method='umap',orig_file=orig_file,pred_file=pred_file)
            ############# LATENT SPACE DISTRIBUTIONS ################

            
            psnr_dict[file_name] = psnr_values
            ssim_dict[file_name] = ssim_values
            pcc_dict[file_name] = pcc_values
            pixcorr_dict[file_name] = pixcorr_values
            percep_dict[file_name] = percep_values
            swav_dict[file_name] = swav_values
            pcc_valz_dict[file_name] = pcc_val_z


            csv_files = os.path.join(outdir, "csv_files")
            if not os.path.exists(csv_files):
                os.mkdir(csv_files)
            psnr_df = pd.concat([psnr_df,pd.DataFrame.from_dict(psnr_dict, orient='index', columns=[f'PSNR_{i+1}' for i in range(number)])])
            psnr_df.to_csv(os.path.join(csv_files,'psnr.csv'), index=True)

            ssim_df = pd.concat([ssim_df,pd.DataFrame.from_dict(ssim_dict, orient='index', columns=[f'SSIM_{i+1}' for i in range(number)])])
            ssim_df.to_csv(os.path.join(csv_files,'ssim.csv'), index=True)

            pcc_df = pd.concat([pcc_df,pd.DataFrame.from_dict(pcc_dict, orient='index', columns=[f'PCC_{i+1}' for i in range(number)])])
            pcc_df.to_csv(os.path.join(csv_files,'pcc.csv'), index=True)

            pixcorr_df =  pd.concat([pixcorr_df,pd.DataFrame.from_dict(pixcorr_dict, orient='index', columns=[f'PIXCORR_{i+1}' for i in range(number)])])
            pixcorr_df.to_csv(os.path.join(csv_files,'pixcorr.csv'), index=True)

            percep_df =  pd.concat([percep_df,pd.DataFrame.from_dict(percep_dict, orient='index', columns=[f'PERCEP_{i+1}' for i in range(number)])])
            percep_df.to_csv(os.path.join(csv_files,'percep.csv'), index=True)

            incep_df =  pd.concat([incep_df,pd.DataFrame.from_dict(inception_dict, orient='index', columns=[f'INCEP_{i+1}' for i in range(5)])])
            incep_df.to_csv(os.path.join(csv_files,'incep.csv'), index=True)
            
            swav_df =  pd.concat([swav_df,pd.DataFrame.from_dict(swav_dict, orient='index', columns=[f'SWAV_{i+1}' for i in range(number)])])
            swav_df.to_csv(os.path.join(csv_files,'swav.csv'), index=True)

            t+=1
            
            merge = False
            if merge:
                merged_df = pd.concat([psnr_df, ssim_df, pcc_df, percep_df, pixcorr_df,swav_df, ], axis=1) #incep_df
                
            preds2 = chunk_list(preds, number)
            
            psnr_matrix = np.zeros([len(origs),len(origs)])
            ssim_matrix = np.zeros([len(origs),len(origs)])
            pcc_matrix = np.zeros([len(origs),len(origs)])
            pixcorr_matrix = np.zeros([len(origs),len(origs)])
            percep_matrix = np.zeros([len(origs),len(origs)])
            swav_matrix = np.zeros([len(origs),len(origs)])
            incep_matrix = np.zeros([len(origs),len(origs)])
            for i, orig in enumerate(origs):
                for j, pred_list in enumerate(preds2):
                    psnr_values = [PSNR(orig, pred) for pred in pred_list]
                    psnr_matrix[i, j] = np.mean(psnr_values)
                    
                    ssim_values = [ssim(orig, pred) for pred in pred_list]
                    ssim_matrix[i, j] = np.mean(ssim_values)
                    
                    pcc_values = [pearson_correlation(orig, pred) for pred in pred_list]
                    pcc_matrix[i, j] = np.mean(pcc_values)
                    
                    pixcorr_values = [pixcorr(orig, pred) for pred in pred_list]
                    pixcorr_matrix[i, j] = np.mean(pixcorr_values)
                    
                    percep_values = [perceptual_loss(rearrange(torch.Tensor(orig).unsqueeze(0), 'b h w c -> b c h w'),rearrange(torch.Tensor(pred).unsqueeze(0), 'b h w c -> b c h w')).item() for pred in pred_list]
                    percep_matrix[i, j] = np.mean(percep_values)
                    
                    incep_values = [criterion(pred, orig) for pred in pred_list]
                    incep_matrix[i, j] = np.mean(incep_values)
                    
                    swav_values = [calculate_swav_metric(orig, pred) for pred in pred_list]
                    swav_matrix[i, j] = np.mean(swav_values)
    
    def plot_matrix(matrix, title, xlabel, ylabel, colorbar=True, save_path=None):
        plt.figure(figsize=(10, 10))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.imshow(matrix, cmap='viridis')
        
        if colorbar:
            plt.colorbar()
    
        plt.grid(which='both', color='black', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-0.5, matrix.shape[1], 1), np.arange(0, matrix.shape[1] + 1, 1), rotation=90)
        plt.yticks(np.arange(-0.5, matrix.shape[0], 1), np.arange(0, matrix.shape[0] + 1, 1))
        plt.gca().set_xticks(np.arange(-.5, matrix.shape[1], 1), minor=True)
        plt.gca().set_yticks(np.arange(-.5, matrix.shape[0], 1), minor=True)
        plt.grid(which='minor', color='black', linestyle='-', linewidth=1)
        
        for i in range(min(matrix.shape)):
            plt.gca().add_patch(Rectangle((i-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', linewidth=2.5))
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
    
    metric_matrix = os.path.join(outdir, "metric_matrix")
    if not os.path.exists(metric_matrix):
        os.mkdir(metric_matrix)
    plot_matrix(psnr_matrix, "PSNR Matrix", "Presented Images", "Reconstructed Images", save_path=os.path.join(metric_matrix, 'psnr_matrix.png'))
    plot_matrix(ssim_matrix, "SSIM Matrix", "Presented Images", "Reconstructed Images", save_path=os.path.join(metric_matrix, 'ssim_matrix.png'))
    plot_matrix(pcc_matrix, "PCC Matrix", "Presented Images", "Reconstructed Images", save_path=os.path.join(metric_matrix, 'pcc_matrix.png'))
    plot_matrix(pixcorr_matrix, "PixCorr Matrix", "Presented Images", "Reconstructed Images", save_path=os.path.join(metric_matrix, 'pixcorr_matrix.png'))
    plot_matrix(percep_matrix, "Perceptual Loss Matrix", "Presented Images", "Reconstructed Images", save_path=os.path.join(metric_matrix, 'perceptual_loss_matrix.png'))
    plot_matrix(incep_matrix, "Inception Loss Matrix", "Presented Images", "Reconstructed Images", save_path=os.path.join(metric_matrix, 'inception_loss_matrix.png'))
    plot_matrix(swav_matrix, "SwAV Loss Matrix", "Presented Images", "Reconstructed Images", save_path=os.path.join(metric_matrix, 'swav_loss_matrix.png'))


    def save_matrices_to_excel(psnr_matrix, ssim_matrix, pcc_matrix, pixcorr_matrix, percep_matrix,swav_matrix, file_path):
        with pd.ExcelWriter(file_path) as writer:
            pd.DataFrame(psnr_matrix).to_excel(writer, sheet_name='PSNR')
            pd.DataFrame(ssim_matrix).to_excel(writer, sheet_name='SSIM')
            pd.DataFrame(pcc_matrix).to_excel(writer, sheet_name='PCC')
            pd.DataFrame(pixcorr_matrix).to_excel(writer, sheet_name='PixCorr')
            pd.DataFrame(percep_matrix).to_excel(writer, sheet_name='Perceptual_Loss')
            pd.DataFrame(swav_matrix).to_excel(writer, sheet_name='SwAV_Loss')
            pd.DataFrame(incep_matrix).to_excel(writer, sheet_name='Incep_Loss')
    
    file_path = os.path.join(outdir,'matrices.xlsx')
    save_matrices_to_excel(psnr_matrix, ssim_matrix, pcc_matrix, pixcorr_matrix, percep_matrix, swav_matrix, file_path) #incep_matrix
        
    def split_pix_values_to_excel(input_csv_path, output_excel_path):
        df = pd.read_csv(input_csv_path, header=None)
        new_data = []
        
        for index, row in df.iterrows():
            new_row = []
            for cell in row:
                if isinstance(cell, str):
                    values = cell.split(',')
                    for value in values:
                        try:
                            new_row.append(float(value))
                        except ValueError:
                            continue
                else:
                    new_row.append(cell)
            new_data.append(new_row)
        new_data = new_data[1:]
        new_df = pd.DataFrame(new_data)
        cols = list(df.columns)
        new_df.index =df.iloc[:, 0].tolist()[1:] #df.columns
        new_df.columns = df.iloc[0].tolist()[1:] #df.index
        new_df.to_excel(output_excel_path, index=True)
        
    split_pix_values_to_excel(os.path.join(csv_files, 'pixcorr.csv'), os.path.join(outdir, 'pixcorr.xlsx'))
    split_pix_values_to_excel(os.path.join(csv_files, 'psnr.csv'), os.path.join(outdir, 'psnr.xlsx'))
    split_pix_values_to_excel(os.path.join(csv_files, 'ssim.csv'), os.path.join(outdir, 'ssim.xlsx'))
    split_pix_values_to_excel(os.path.join(csv_files, 'pcc.csv'), os.path.join(outdir, 'pcc.xlsx'))
    split_pix_values_to_excel(os.path.join(csv_files, 'percep.csv'), os.path.join(outdir, 'percep.xlsx'))
    split_pix_values_to_excel(os.path.join(csv_files, 'swav.csv'), os.path.join(outdir, 'swav.xlsx'))
    
    pcc_valz_avg = []
    for key in pcc_valz_dict.keys():
        pcc_valz_avg.append(np.mean(pcc_valz_dict[key][0:5]))
    print("Overall 14 averages of PCC 1 avg is 5 img avg" , np.mean(pcc_valz_avg))
    print("Overall 14 std of PCC 1 avg is 5 img avg" , np.std(pcc_valz_avg))
    

    pcc_valz_avg = []
    pcc_valz_sem = []
    
    for key in pcc_valz_dict.keys():
        # 5 rec sample avg
        avg = np.mean(pcc_valz_dict[key][0:5])
        pcc_valz_avg.append(avg)
        
        # 5 rec samples std
        std_dev = np.std(pcc_valz_dict[key][0:5], ddof=1)
        
        # SEM calc
        sem = std_dev / np.sqrt(5)
        pcc_valz_sem.append(sem)
    
    # Genel 14 ortalamanın ortalamasını hesapla
    overall_avg = np.mean(pcc_valz_avg)
    print("Overall 14 averages of PCC 1 avg is 5 img avg:", overall_avg)
    
    # Genel SEM hesapla (14 orijinal görüntünün ortalama SEM'i)
    overall_sem = np.mean(pcc_valz_sem)
    print("Overall SEM of PCC 1 avg is 5 img avg:", overall_sem)

    #önce samplelar kendi içinde ortalama sonra ortalamaların ortalaması 
    pixcorr_avg = []
    for key in pixcorr_dict.keys():
        pixcorr_avg.append(np.mean(pixcorr_dict[key]))
    print(f"Overall 14 averages of pixcorr 1 avg is {number} img avg" , np.mean(pixcorr_avg))
    print(f"Overall 14 std of pixcorr 1 avg is {number} img avg" , np.std(pixcorr_avg))
    
    # tümümünün ortalaması 
    pixcorr_avg = []
    for key in pixcorr_dict.keys():
        pixcorr_avg.append(pixcorr_dict[key])
    print(f"Overall 14 averages of pixcorr 1 avg is {number} img avg" , np.mean(pixcorr_avg))
    
    ssim_avg = []
    for key in ssim_dict.keys():
        ssim_avg.append(np.mean(ssim_dict[key]))
    print(f"Overall 14 averages of pixcorr 1 avg is {number} img avg" , np.mean(ssim_avg))
    print(f"Overall 14 std of pixcorr 1 avg is {number} img avg" , np.std(ssim_avg))

                ###Inception Feautures###
    feat_list = []
    def fn(module, inputs, outputs):
        feat_list.append(outputs.cpu().detach().numpy())
        
    inception_v3 = models.inception_v3(pretrained=True)
    layer="avgpool"
    if layer== 'avgpool':
        inception_v3.avgpool.register_forward_hook(fn) 
    elif layer == 'lastconv':
        inception_v3.Mixed_7c.register_forward_hook(fn)
    inception_v3.eval()
    
    def load_image(image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image)
        return image
    
    def extract_inception_features(model, image):
        with torch.no_grad():
            image = image.unsqueeze(0) 
            features = model(image)
        return features
    
    def cosine_similarity(tensor1, tensor2):
        return F.cosine_similarity(tensor1, tensor2).item()

    def two_way_comparison(original_image, reconstructed_image):
        original_features = inception_v3(original_image.unsqueeze(0))
        reconstructed_features = inception_v3(reconstructed_image.unsqueeze(0))
        return original_features, reconstructed_features

    incep_gt = []
    incep_recs = []
    chunk_preds = chunk_list(preds, number)  
    for original_image, reconstructed_images in zip(origs, chunk_preds):
        original_image = load_image(original_image)
        for reconstructed_image in reconstructed_images:
            reconstructed_image = load_image(reconstructed_image)
            original_features,reconstructed_features = two_way_comparison(original_image, reconstructed_image)
            incep_gt.append(original_features.detach().numpy())
            incep_recs.append(reconstructed_features.detach().numpy())

    
    incep_gt = np.array(incep_gt).reshape(len(incep_gt),-1)
    incep_recs = np.array(incep_recs).reshape(len(incep_recs),-1)

    incep_perf,incep_p = pairwise_corr_all(incep_gt, incep_recs)

    feat_list = []
    def fn(module, inputs, outputs):
        feat_list.append(outputs.cpu().detach().numpy())

    
    alexnet2 = models.alexnet(pretrained=True)
    alexnet5 = models.alexnet(pretrained=True)

    alexnet2.features[4].register_forward_hook(fn)
    alexnet5.features[11].register_forward_hook(fn)
    alexnet2.eval()
    alexnet5.eval()


    def load_image(image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
        image = transform(image).unsqueeze(0) 
        return image
    
    def two_way_comparison(original_image, reconstructed_image, alexnet2, alexnet5):
        original_features_2 = alexnet2(original_image).cpu().detach().numpy()
        reconstructed_features_2 = alexnet2(reconstructed_image).cpu().detach().numpy()
        
        original_features_5 = alexnet5(original_image).cpu().detach().numpy()
        reconstructed_features_5 = alexnet5(reconstructed_image).cpu().detach().numpy()
        return original_features_2,reconstructed_features_2,original_features_5,reconstructed_features_5

    
    gt2 = []
    recs2 = []
    gt5 = []
    recs5 = []
    
    chunk_preds = chunk_list(preds, number)  
    for original_image, reconstructed_images in zip(origs, chunk_preds):
        original_image = load_image(original_image)
        for reconstructed_image in reconstructed_images:
            reconstructed_image = load_image(reconstructed_image)
            original_features_2,reconstructed_features_2,original_features_5,reconstructed_features_5 = two_way_comparison(original_image, reconstructed_image, alexnet2, alexnet5)
            gt2.append(original_features_2)
            recs2.append(original_features_2)
            
            gt5.append(original_features_5)
            recs5.append(original_features_5)
            
    #### chuck list acc ########
    chucks_gt2 = chunk_list(gt2,number)
    chucks_gt2 = [np.abs(np.array(i).mean()) for i in chucks_gt2]
    chucks_recs2 = chunk_list(recs2,number)
    

    chucks_gt5 = chunk_list(gt5,number)
    chucks_gt5 = [np.abs(np.array(i).mean()) for i in chucks_gt5]
    chucks_recs5 = chunk_list(recs5,number)
    chucks_recs5 = [np.abs(np.array(i).mean()) for i in chucks_recs5]
    
    
    
    gt2 = np.array(gt2).reshape(len(gt2),-1)
    gt5 = np.array(gt5).reshape(len(gt5),-1)
    recs2 = np.array(recs2).reshape(len(recs2),-1)
    recs5 = np.array(recs5).reshape(len(recs5),-1)

  perf2,p2 = pairwise_corr_all(gt2, recs2)
    perf5,p5 = pairwise_corr_all(gt5, recs5)
            
######################################################################################
            ###SwAV and efficientnet###
    
    def two_way_comparison(original_image, reconstructed_image, swav, effdet):
        original_features_2 = swav(original_image).cpu().detach().numpy()
        reconstructed_features_2 = swav(reconstructed_image).cpu().detach().numpy()
        
        original_features_5 = effdet(original_image).cpu().detach().numpy()
        reconstructed_features_5 = effdet(reconstructed_image).cpu().detach().numpy()
        return original_features_2,reconstructed_features_2,original_features_5,reconstructed_features_5
        
    efficientnet = models.efficientnet_b1(weights=True)
    efficientnet.avgpool.register_forward_hook(fn) 
    
    swav = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav.avgpool.register_forward_hook(fn) 
    
    swav.eval()
    efficientnet.eval()
    
    swav_gt = []
    swav_recs = []
    effnet_gt = []
    effnet_recs = []
    
    chunk_preds = chunk_list(preds, number)
    for original_image, reconstructed_images in zip(origs, chunk_preds):
        original_image = load_image(original_image)
        for reconstructed_image in reconstructed_images:
            reconstructed_image = load_image(reconstructed_image)
            original_swavfeatures,reconstructed_swavfeatures, original_effnetfeatures,reconstructed_effnetfeatures = two_way_comparison(original_image, reconstructed_image,swav,efficientnet)

            swav_gt.append(original_swavfeatures)
            swav_recs.append(reconstructed_swavfeatures)

            effnet_gt.append(original_effnetfeatures)
            effnet_recs.append(reconstructed_effnetfeatures)

    
    swav_gt = np.array(swav_gt).reshape(len(swav_gt),-1)
    swav_recs = np.array(swav_recs).reshape(len(swav_recs),-1)

    swav_perf,swav_p = pairwise_corr_all(swav_gt, swav_recs)
    ### distance ####

    distance_fn = sp.spatial.distance.correlation
    swav_dist = np.array([distance_fn(i,j) for i,j in zip (swav_gt,swav_recs)]).mean()
    #################
    effnet_gt = np.array(effnet_gt).reshape(len(effnet_gt),-1)
    effnet_recs = np.array(effnet_recs).reshape(len(effnet_recs),-1)

    effnet_perf,effnet_p = pairwise_corr_all(effnet_gt, effnet_recs)
    
    ### distance ####
    effnet_dist = np.array([distance_fn(i,j) for i,j in zip (effnet_gt,effnet_recs)]).mean()
    #################
    
    def max_val_metric(matrix, which_metric):
        max_val = np.max(matrix)
        max_loc = np.unravel_index(np.argmax(matrix), matrix.shape)
        print(f"Maximum {which_metric} value: {max_val} at location: {max_loc}")
        return max_val, max_loc
    
    max_psnr_value, max_psnr_loc = max_val_metric(psnr_matrix, "PSNR")
    max_ssim_value, max_ssim_loc = max_val_metric(ssim_matrix, "SSIM")
    max_pcc_value, max_pcc_loc = max_val_metric(pcc_matrix, "PCC")
    max_pixcorr_value, max_pixcorr_loc = max_val_metric(pixcorr_matrix, "PIXCORR")
    max_percep_value, max_percep_loc = max_val_metric(percep_matrix, "PERCEP")
    
    def overall_avg(matrix,which_metric):
        matrix_avg = 0
        number_vals = matrix.shape[0] * matrix.shape[1]
        for i in range((matrix.shape[0])):
            for j in range(matrix.shape[1]):
                matrix_avg +=matrix[i,j]
        matrix_avg = matrix_avg / number_vals
        return matrix_avg
        
    def diagnonal_avg(matrix, which_metric):
        len_matrix = matrix.shape[0]
        diag_vals = []
        for i in range(min(matrix.shape)):
            diag_vals.append(matrix[i, i])
        diag_avg = diag_vals / len_matrix
        print("{which_metric} diagonal avg: " ,diag_avg)
        print("{which_metric} diagonal std: ",np.std(diag_avg))
        return diag_vals, diag_avg, np.std(diag_avg)
    
