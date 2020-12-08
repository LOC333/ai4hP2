"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        return med_reshape(volume, (volume.shape[0], 64, 64))
    

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array.  verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.

        mask3d = np.zeros(volume.shape)
        for idx in range(volume.shape[0]):
            image = volume[idx, :, :]
            image = image.astype(np.single)/np.max(image)
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            prediction = self.model(image_tensor.to(self.device))

            mask3d[idx,:,:] = torch.argmax(np.squeeze(prediction.cpu().detach()), dim=0)
       
        return mask3d
