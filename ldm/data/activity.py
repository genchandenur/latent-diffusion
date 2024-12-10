"""
# fmri activity subject paths 
#     data_root = 'D:/bold5000/derivatives/fmriprep, 
# Stimulus Presentation lists paths
#     stim_root = r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Stimuli_Presentation_Lists',
# Stimulus images paths
#     scene_stim = r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Scene_Stimuli\Presented_Stimuli',
# Which subject used for LDM model ('CSI1', 'CSI2', 'CSI3', 'CSI4') You can use one or multiple subject
#     sub_id = ids
# If you use single subject please assign this flag True if not assign False
#     single_sub= one_sub
# Resize image in terms of this flag below
#     img_dim=size
# If you want to use one or multiple domain of image assign image domain
#     subDataset = 'ImageNet'

BOLD5000 dataset folder path looks like

    ├── CSI1
    │  ├── BOLD5000_CSI1_sess1
    │    ├──Behavioral_Data
    │    ├──BOLD_Raw
    │      ├──01_BOLD_CSI1_Sess-13_Run-1
    │         ├── concat_file
    │          ├── conc.03-0001-000001.npy
    │          ├── conc.03-0001-000002.npy
    │          ├── ...
    │        ├── ...
    │      ├──02_BOLD_CSI1_Sess-13_Run-2
    │      ├──03_BOLD_CSI1_Sess-13_Run-3
    │      ├── ...
    │    ├──physio
    │    ├──DICOM_log_171214112502.txt
    │  ├── BOLD5000_CSI1_sess2
    │   ├── ...

    ├── CSI2
    │   ├── ...
    
    Label file name changed Scene to Scenes
    Presented Stimulus files looks like
    ├── COCO
    │   ├── COCO_train2014_000000000036.jpg
    │   ├── COCO_train2014_000000000584.jpg
    │   ├── ...
    ├── ImageNet
    │   ├── n01440764_10110.JPEG
    │   ├── n01440764_13744.JPEG
    │   ├── ...
    ├── Scenes
    │   ├── ...

"""

import os
import glob
import cv2
import numpy as np
import itertools
from torch.utils.data import Dataset

def resize_image(image,dim):
    image = cv2.resize(image,(dim,dim))
    return image

def convert_rgb(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

def filter_list(directory,list_for):
    directory = list(filter(lambda item: list_for in item, directory))
    return directory

def flat_paths(directory):
    directory = list(itertools.chain(*directory))
    return directory

def normalize_activity(image, min_val = None, max_val = None): # [-1, 1]
    image = 2 * ((image - min_val) / max_val - min_val) - 1
    return image

def minmax_scaling(image, min_val = None, max_val = None): # [0, 1]
    image = (image - min_val) / (max_val - min_val) 
    return image

def minmax_scalingg(image):
    image = image / np.max(image)
    return image

def read_stim_files(directory, txt_files):
    txt_file_paths = [os.path.join(directory, txt) for txt in txt_files]
    nested_list = []

    for txt_path in txt_file_paths:
        sub_dirs = [os.listdir(txt_path)]
        for sub_dir in sub_dirs[0]:
            full_path = os.path.join(txt_path, sub_dir)
            files = [f for f in os.listdir(full_path) if f.endswith(".txt")]
            nested_list.append([full_path, files])

    for sublist in nested_list:
        sublist[1] = [os.path.join(sublist[0], subpath) for subpath in sublist[1]]

    column2 = [item[1] for item in nested_list]
    return column2

def stim_folder(scene_stim):
    dataset = {}
    for data in os.listdir(scene_stim):
        dataset[data] = os.listdir(os.path.join(scene_stim,data))
    return dataset

def subject_parents(parent_dir, txt_files):
    participants = os.listdir(parent_dir)
    subject_dirs = [os.path.join(parent_dir, sub) for sub in participants if sub[4:] in txt_files]
    ses_paths = []

    for sub_dir in subject_dirs:
        for folder in os.listdir(sub_dir):
            if "ses" in folder:
                ses_paths.append(os.path.join(sub_dir, folder))

    ses_paths = [
        os.path.join(ses, "interpolation") for ses in sorted(ses_paths)
        if not (
            ("ses-16" in ses and any(sub in ses for sub in ["CSI1", "CSI2", "CSI3"])) or
            ("ses-10" in ses and "CSI4" in ses))
    ]
    
    joined_paths = []
    for ses_path in ses_paths:
        for folder in os.listdir(ses_path):
            if os.path.isdir(os.path.join(ses_path, folder)):
                joined_paths.append(os.path.join(ses_path, folder))

    correspond_path = [
        run[x:x + 5] for run in [ses[1][3:188] for ses in joined_paths]
        for x in range(0, len(run), 5)
    ]
    return correspond_path


class MriActivityBase(Dataset):
    def __init__(self, data_root,stim_root,scene_stim, sub_id,single_sub, img_dim, subDataset):
        self.data_root = data_root
        self.stim_root = stim_root
        self.scene_stim = scene_stim
        self.sub_id = sub_id
        self.single_sub = single_sub
        self.img_dim = img_dim
        self.subDataset = subDataset
        self.data = {}
        
        if not self.single_sub:
            assert len(sub_id) > 1
        txt_files = os.listdir(self.stim_root)

        if self.single_sub:
            if isinstance(self.sub_id, str):
                txt_files = [self.sub_id]
            else:
                txt_files = list(self.sub_id)
                
        column2 = read_stim_files(self.stim_root, txt_files)
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
                            
        self.data["image_files"] = ["ses-15","run-10"]
        
        self.data["file_path_"] = stimulus_list
        self.data["relative_file_path_"] = subject_parents(self.data_root, self.sub_id)
        
        if self.subDataset is not None:
            self.subDataset = [item.lower() for item in self.subDataset]
    
            idx = [i for i, v in enumerate(self.data["file_path_"]) if all(sub.lower() not in v.lower() for sub in self.subDataset)]
    
            for ids in sorted(idx, reverse=True):
                del self.data["relative_file_path_"][ids]
                del self.data["file_path_"][ids]   

        self.data["relative_file_path_"], self.data["file_path_"] = self.getNamefor(self)
        self._length = len(self.data["relative_file_path_"])
        
    def __len__(self):
        return self._length
    
    def getNamefor(self, data):
        test_file_paths = []
        test_relative_file_paths = []
        test_indices = []
        for idx, (file_path, relative_file_path) in enumerate(zip(self.data["file_path_"], self.data["relative_file_path_"])):
            single_path = relative_file_path[0]
            if "ses-15" in single_path and "run-10" in single_path:
                    test_file_paths.append(file_path)
                    test_relative_file_paths.append(relative_file_path)
                    test_indices.append(idx)
                    

        self.data["file_path_"] = [path for idx, path in enumerate(self.data["file_path_"]) if idx not in test_indices]
        self.data["relative_file_path_"] = [path for idx, path in enumerate(self.data["relative_file_path_"]) if idx not in test_indices]


        train_size = int(len(self.data["relative_file_path_"])*0.8)

        if "Train" in type(self).__name__:
            self.data["relative_file_path_"] = self.data["relative_file_path_"][0:train_size]
            self.data["file_path_"] = self.data["file_path_"][0:train_size]
            return self.data["relative_file_path_"], self.data["file_path_"]

        elif "Valid" in type(self).__name__:
            self.data["relative_file_path_"] = self.data["relative_file_path_"][train_size:]
            self.data["file_path_"] = self.data["file_path_"][train_size:]
            return self.data["relative_file_path_"], self.data["file_path_"]

        else:
            return test_relative_file_paths, test_file_paths
            
            
    def __getitem__(self, i):    
        image = {key : resize_image(minmax_scaling(np.load(key,allow_pickle=True),min_val=np.min(np.load(key,allow_pickle=True)),max_val=np.max(np.load(key,allow_pickle=True))),self.img_dim) for key in self.data["relative_file_path_"][i]}
        path_label = self.data["file_path_"][i] 
        stimulus = cv2.imread(path_label)
        stimulus = convert_rgb(stimulus)
        stimulus = resize_image(stimulus,self.img_dim)
        stimulus = (stimulus / 127.5 - 1.0).astype(np.float32)
        

        example = {
            "image" : stimulus,
            "activity" : np.array(list(image.values())).transpose(1,2,0),
            "relative_file_path_" : self.data["relative_file_path_"][i],
            "c_name" : self.data["file_path_"][i],
            "image_files" : self.data["image_files"],
            }
        
        return example
                               

class activityTrain(MriActivityBase):
    def __init__(self, size = None, ids = None, one_sub = None, **kwargs):
        super().__init__(data_root = "D:/bold5000/derivatives/fmriprep", stim_root = r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Stimuli_Presentation_Lists',
                         scene_stim = r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Scene_Stimuli\Presented_Stimuli',
                         sub_id = ids, single_sub= one_sub, img_dim=size,subDataset = ["ImageNet"]) # ["ImageNet","Scene","COCO"]


class activityValidation(MriActivityBase):
    def __init__(self, size = None, ids = None, one_sub = None, **kwargs):
        super().__init__(data_root = "D:/bold5000/derivatives/fmriprep", stim_root = r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Stimuli_Presentation_Lists',
                         scene_stim = r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Scene_Stimuli\Presented_Stimuli',
                         sub_id = ids, single_sub= one_sub, img_dim=size,subDataset = ["ImageNet"]) # ["ImageNet","Scene","COCO"]

class activityTest(MriActivityBase):
    def __init__(self, size = None, ids = None, one_sub = None, **kwargs):
        super().__init__(data_root = "D:/bold5000/derivatives/fmriprep", stim_root = r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Stimuli_Presentation_Lists',
                         scene_stim = r'C:\Users\Asus\Desktop\latent-diffusion\BOLD5000_Stimuli\Scene_Stimuli\Presented_Stimuli',
                         sub_id = ids, single_sub= one_sub, img_dim=size,subDataset = ["ImageNet"]) # ["ImageNet","Scene","COCO"]
