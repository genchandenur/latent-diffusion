import os
import cv2
import gzip
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import zoom

def resize_image(image,dim):
    image = cv2.resize(image,(dim,dim))
    return image

def minmax_scalingg(image): # [0, 1]
    image = image / np.max(image)
    return image

def minmax_scaling(image, min_val = None, max_val = None):
    image = (image - min_val) / (max_val - min_val) 
    return image

def subject_parents(parent_dir, sub_id):
    parents = os.listdir(parent_dir)
    return [os.path.join(parent_dir, sub) for sub in parents if sub[4:] in sub_id]

def convert_rgb(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

def flatten_relative_paths(data):
    keys, paths = [], []
    for key, value in data.items():
        for sublist in value:
            paths.append(sublist)
            keys.append(key)
    return keys, paths

def build_directory_structure(parent_dir, sub_id):
    df = {}
    data = subject_parents(parent_dir, sub_id)

    for path in data:
        func_paths = [
            os.path.join(path, sub_path, "func")
            for sub_path in os.listdir(path)
            if os.path.isdir(os.path.join(path, sub_path))
        ]
        df[path] = [func for func in func_paths if os.path.exists(func)]
    
    return df

def filter_bold_files(directory_structure):
    filtered_files = {}
    for key, func_paths in directory_structure.items():
        filtered_files[key] = []
        for func_path in func_paths:
            bold_files = [
                os.path.join(func_path, f) 
                for f in os.listdir(func_path)
                if 'bold.nii.gz' in f and 'run-' in f
            ]
            filtered_files[key].extend(bold_files)

    return filtered_files

def read_stim_files(directory,txt_files):
    txt_file = [os.path.join(directory,txt) for txt in txt_files]
    sub_paths = [[os.listdir(txt)] for txt in txt_file]
    
    merged_paths = []
    
    for txt, sub_list in zip(txt_file, sub_paths):
        for sub in sub_list[0]:
            merged_paths.append(os.path.join(txt, sub))
    
    for path in merged_paths:
        if "info" in path:
            merged_paths.remove(path)
        
    nested_list = []
    
    for path in merged_paths:
        sub_files = [f for f in os.listdir(path) if f.endswith(".txt")]
        nested_list.append([path, sub_files])
    
    
    for sublist in nested_list:
        joined_paths = [os.path.join(sublist[0], subpath) for subpath in sublist[1]]
        sublist[1] = joined_paths
        
    column1 = [item[0] for item in nested_list]
    column2 = [item[1] for item in nested_list]
    return column2

def stim_folder(scene_stim):
    dataset = {}
    data_dic = os.listdir(scene_stim)
    for data in data_dic:
        dataset[data] = os.listdir(os.path.join(scene_stim,data))
    return dataset

def chunk_list(lst, chunk_size):
    chunks = []
    for i in range(0, len(lst), chunk_size):
        chunks.append(lst[i:i + chunk_size])
    return chunks

class MriActivityBase(Dataset):
    def __init__(self, data_root,stim_root,scene_stim, sub_id,single_sub, img_dim, subDataset, axes):
        self.data_root = data_root
        self.stim_root = stim_root
        self.scene_stim = scene_stim
        self.sub_id = sub_id
        self.single_sub = single_sub
        self.img_dim = img_dim
        self.subDataset = subDataset
        self.axes = axes
        self.data = {}
        

        if self.single_sub:
            if isinstance(self.sub_id, str):
                txt_files = [self.sub_id]
            else:
                txt_files = list(self.sub_id)
        else:
            txt_files = list(self.sub_id)   
                    
        column2 = read_stim_files(self.stim_root,txt_files)
        dataset = stim_folder(self.scene_stim)
        
        stimulus_list = list()
        
        for file_path in column2:
            for path in file_path:
                with open(path, 'r') as file:
                    file_contents = [file.read()]
                    file_contents = file_contents[0].split('\n')
                for file in file_contents:
                    if file == "":
                        file_contents.remove(file)
                
                for item in range(len(file_contents)):
                    if file_contents[item][0:4] == 'rep_':
                        file_contents[item] = file_contents[item][4:]
                    for key in dataset.keys():
                        if file_contents[item] in dataset[key]:
                            file_contents[item] = scene_stim + "/" + key + "/" + file_contents[item]
                            stimulus_list.append(file_contents[item])
        
        self.data["file_path_"] = stimulus_list
        
        self.data["relative_file_path_"] = build_directory_structure(self.data_root, self.sub_id)
        self.data["relative_file_path_"] = filter_bold_files(self.data["relative_file_path_"])
        self.data["relative_file_path_"] = [item for sublist in self.data["relative_file_path_"].values() for item in sublist]
        self.data["relative_file_path_"] = {path: list(range(194)) for path in self.data["relative_file_path_"]}
        for run in self.data["relative_file_path_"].keys():
            self.data["relative_file_path_"][run] = self.data["relative_file_path_"][run][3:188]
        for run in self.data["relative_file_path_"].keys():
            self.data["relative_file_path_"][run] = [self.data["relative_file_path_"][run][x:x+5] for x in range(0,len(self.data["relative_file_path_"][run]),5)]
        self.data["relative_file_path_"], self.data["index"] = flatten_relative_paths(self.data["relative_file_path_"])
        
        if self.subDataset is not None:
            self.subDataset = [item.lower() for item in self.subDataset]
    
            idx = [i for i, v in enumerate(self.data["file_path_"]) if all(sub.lower() not in v.lower() for sub in self.subDataset)]
    
            for ids in sorted(idx, reverse=True):
                del self.data["relative_file_path_"][ids]
                del self.data["file_path_"][ids] 
                del self.data["index"][ids] 
                
        self.data["relative_file_path_"], self.data["file_path_"],self.data["index"] = self.getNamefor(self)
        self._length = len(self.data["file_path_"])
        
    def __len__(self):
        return self._length
    
    def getNamefor(self, data):
        test_file_paths = []
        test_relative_file_paths = []
        test_indices = []
        test_vol_indexes = []
        for idx, (file_path, relative_file_path,index) in enumerate(zip(self.data["file_path_"], self.data["relative_file_path_"], self.data["index"])):
            if "ses-15" in relative_file_path and "run-10" in relative_file_path:
                    test_file_paths.append(file_path)
                    test_relative_file_paths.append(relative_file_path)
                    test_indices.append(idx)
                    test_vol_indexes.append(index)

        self.data["file_path_"] = [path for idx, path in enumerate(self.data["file_path_"]) if idx not in test_indices]
        self.data["relative_file_path_"] = [path for idx, path in enumerate(self.data["relative_file_path_"]) if idx not in test_indices]
        self.data["index"] = [path for idx, path in enumerate(self.data["index"]) if idx not in test_vol_indexes]


        train_size = int(len(self.data["relative_file_path_"])*0.8)

        if "Train" in type(self).__name__:
            self.data["relative_file_path_"] = self.data["relative_file_path_"][0:train_size]
            self.data["file_path_"] = self.data["file_path_"][0:train_size]
            self.data["index"] = self.data["index"][0:train_size]
            return self.data["relative_file_path_"], self.data["file_path_"], self.data["index"]

        elif "Valid" in type(self).__name__:
            self.data["relative_file_path_"] = self.data["relative_file_path_"][train_size:]
            self.data["file_path_"] = self.data["file_path_"][train_size:]
            self.data["index"] = self.data["index"][train_size:]
            return self.data["relative_file_path_"], self.data["file_path_"], self.data["index"]

        else:
            return test_relative_file_paths, test_file_paths, test_vol_indexes
        
    def __getitem__(self, i):
        hemo_vols = []
        vol_idx = self.data["relative_file_path_"][i]
        idx = self.data["index"][i]
        
        with gzip.open(self.data["relative_file_path_"][i], 'rb') as f:
            nii_vol = nib.Nifti1Image.from_bytes(f.read())
            nii_data = nii_vol.get_fdata()
        for x in idx:
            resampled_data = zoom(nii_data[:,:,:,x], (72/106, 88/106, 72/69))
            resampled_data = minmax_scalingg(resampled_data)            
            hemo_vols.append(resampled_data)
        
        path_label = self.data["file_path_"][i] 
        stimulus = cv2.imread(path_label)
        stimulus = convert_rgb(stimulus)
        stimulus = resize_image(stimulus,self.img_dim)
        stimulus = (stimulus / 127.5 - 1.0).astype(np.float32)
        vol = np.array(hemo_vols).transpose(1,2,3,0)
        if self.axes == "coronal":
            vol = [vol[i, :, :, :] for i in range(vol.shape[0])]
        elif self.axes == "sagittal":
            vol = [vol[:, i, :, :] for i in range(vol.shape[1])]
        elif self.axes == "axial":
            vol = [vol[:, :, i, :] for i in range(vol.shape[2])]
        else:
            pass
            
        example = {
            "image" : stimulus, #[ 256, 256, 3 ]
            "activity" : vol,
            "relative_file_path_" : self.data["relative_file_path_"][i],
            "file_path_" : path_label,
            "c_name" : self.data["file_path_"][i]

            }
        
        return example

class activityTrain(MriActivityBase):
    def __init__(self, size = None, ids = None, one_sub = None, axes = None, **kwargs):
        super().__init__(data_root = "C:/Users/Asus/Desktop/snapshots/1.1.1/files/derivatives/fmriprep", stim_root = r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Stimuli_Presentation_Lists',
                         scene_stim = r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Scene_Stimuli\Presented_Stimuli',
                         sub_id = ids, single_sub= one_sub, img_dim=size, subDataset = ["ImageNet"], axes = axes)

class activityValidation(MriActivityBase):
    def __init__(self, size = None, ids = None, one_sub = None, axes = None, **kwargs):
        super().__init__(data_root = "C:/Users/Asus/Desktop/snapshots/1.1.1/files/derivatives/fmriprep", stim_root = r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Stimuli_Presentation_Lists',
                         scene_stim = r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Scene_Stimuli\Presented_Stimuli',
                         sub_id = ids, single_sub= one_sub, img_dim=size, subDataset = ["ImageNet"], axes = axes)
class activityTest(MriActivityBase):
    def __init__(self, size = None, ids = None, one_sub = None, axes = None, **kwargs):
        super().__init__(data_root = "C:/Users/Asus/Desktop/snapshots/1.1.1/files/derivatives/fmriprep", stim_root = r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Stimuli_Presentation_Lists',
                         scene_stim = r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Scene_Stimuli\Presented_Stimuli',
                         sub_id = ids, single_sub= one_sub, img_dim=size, subDataset = ["ImageNet"], axes = axes)
        
