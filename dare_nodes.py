from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import folder_paths
import os

import comfy.controlnet
import comfy.sd
import comfy.utils
import comfy.model_management
import torch.nn.functional as F

from safetensors.torch import save_file


def _L2Normalize(v, eps=1e-12):
    return v/(torch.norm(v) + eps)

def spectral_norm(W, u=None, Num_iter=100):
    '''
    Spectral Norm of a Matrix is its maximum singular value.
    This function employs the Power iteration procedure to
    compute the maximum singular value.
    ---------------------
    :param W: Input(weight) matrix - autograd.variable
    :param u: Some initial random vector - FloatTensor
    :param Num_iter: Number of Power Iterations
    :return: Spectral Norm of W, orthogonal vector _u
    '''
    if not Num_iter >= 1:
        raise ValueError("Power iteration must be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0,1).cuda()
    W = W.to(u.device).type(torch.FloatTensor)
    # Power iteration
    wdata = W.data.to(u.device)
    for _ in range(Num_iter):
        wdata = wdata.view(u.shape[-1],-1)
        v = _L2Normalize(torch.matmul(u, wdata))
        u = _L2Normalize(torch.matmul(v, torch.transpose(wdata,0, 1)))
    sigma = torch.sum(F.linear(u, torch.transpose(wdata, 0,1)) * v)
    return sigma, u

def merge_tensors(tensor1, tensor2, p):
    # Calculate the delta of the weights
    delta = tensor2 - tensor1
    # Generate the mask m^t from Bernoulli distribution
    m = torch.bernoulli(torch.full(delta.shape, p)).to(tensor1.dtype)
    # Apply the mask to the delta to get δ̃^t
    delta_tilde = m * delta
    # Scale the masked delta by the dropout rate to get δ̂^t
    delta_hat = delta_tilde / (1 - p)
    return delta_hat

def merge_weights(f1, f2, p, lambda_val, lipschitz_regularize, strength):
    merged_tensors = {}
    keys2 = set(f2.keys())
    if f1 is None:
        keys1 = set(f2.keys())
    else:
        keys1 = set(f1.keys())

    common_keys = keys1.intersection(keys2)
    lips = []

    for key in common_keys:
        tensor2 = f2[key]
        if f1 is None:
            tensor1 = torch.zeros_like(tensor2)
        else:
            tensor1 = f1[key]
        tensor1, tensor2 = resize_tensors(tensor1, tensor2)
        diff = strength * lambda_val * merge_tensors(tensor1, tensor2, p)
        merged_tensors[key] = tensor1 + diff
        if(len(merged_tensors[key].shape) != 0):
            if lipschitz_regularize > 0:
                sn = spectral_norm(merged_tensors[key])[0].cpu()
                lips.append(sn)
        #print("merging", key)

    if lipschitz_regularize > 0:
        sn = max(lips)
        print("Regularizing lipschitz ", sn, "to", lipschitz_regularize)
        for key in common_keys:
            merged_tensors[key] /= sn/lipschitz_regularize

    return merged_tensors

def resize_tensors(tensor1, tensor2):
    if len(tensor1.shape) not in [1, 2]:
        return tensor1, tensor2

    # Pad along the last dimension (width)
    if tensor1.shape[-1] < tensor2.shape[-1]:
        padding_size = tensor2.shape[-1] - tensor1.shape[-1]
        tensor1 = F.pad(tensor1, (0, padding_size, 0, 0))
    elif tensor2.shape[-1] < tensor1.shape[-1]:
        padding_size = tensor1.shape[-1] - tensor2.shape[-1]
        tensor2 = F.pad(tensor2, (0, padding_size, 0, 0))

    # Pad along the first dimension (height)
    if tensor1.shape[0] < tensor2.shape[0]:
        padding_size = tensor2.shape[0] - tensor1.shape[0]
        tensor1 = F.pad(tensor1, (0, 0, 0, padding_size))
    elif tensor2.shape[0] < tensor1.shape[0]:
        padding_size = tensor1.shape[0] - tensor2.shape[0]
        tensor2 = F.pad(tensor2, (0, 0, 0, padding_size))

    return tensor1, tensor2


class DARE_Merge_LoraStack:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                            "lora_stack": ("LORA_STACK", ),
                            "lambda_val": ("FLOAT", {"default": 1.5, "min": -4.0, "step": 0.1, "max": 4.0}),
                            "p": ("FLOAT", {"default": 0.13, "min": 0.01, "step": 0.01, "max": 1.0}),
                            "lipschitz_regularizer": ("FLOAT", {"default": 5.0, "min": -1, "step": 0.1, "max": 10000.0}),
                            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                            }
        }

    RETURN_TYPES = ["LoRA"]
    RETURN_NAMES = ["LoRA"]
    FUNCTION = "apply_lora_stack"
    CATEGORY = "Comfyroll/IO"

    def apply_lora_stack(self, lora_stack, lambda_val, p, lipschitz_regularizer, seed):
        # lipschitz -> rename spectral_norm
        # regularizer list with off/spectral_norm

        # Initialise the list
        lora_params = list()
        torch.manual_seed(seed)
        # Extend lora_params with lora-stack items 
        if lora_stack:
            lora_params.extend(lora_stack)
        else:
            return [None]

        # Initialise the model and clip
        lora_name, strength_model, strength_clip = lora_params[0]
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora0 = comfy.utils.load_torch_file(lora_path, safe_load=True)
        weights = merge_weights(None, lora0, p, lambda_val, -1, strength_model)

        # Loop through the list
        for i in range(len(lora_params)-1):
            lora_name, strength_model, strength_clip = lora_params[i+1]
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            if(i==len(lora_params)-2):
                weights = merge_weights(weights, lora, p, lambda_val, lipschitz_regularizer, strength_model)
            else:
                weights = merge_weights(weights, lora, p, lambda_val, -1, strength_model)

        return [weights]


class ApplyLoRA:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "LoRA": ("LoRA",),
                            "model": ("MODEL",),
                            "clip": ("CLIP", ),
                            "lora_model_wt": ("FLOAT", {"default": 1.0, "min": 0.01, "step": 0.01, "max": 10.0}),
                            "lora_clip_wt": ("FLOAT", {"default": 1.0, "min": 0.01, "step": 0.01, "max": 10.0})
                              }
        }
    RETURN_TYPES = ["MODEL", "CLIP"]
    RETURN_NAMES = ["model", "clip"]
    FUNCTION = "apply_lora_stack"

    CATEGORY = "advanced/model_merging"

    def apply_lora_stack(self, model, clip, LoRA, lora_model_wt, lora_clip_wt):
        return comfy.sd.load_lora_for_models(model, clip, LoRA, lora_model_wt, lora_clip_wt)

class SaveLoRA:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "LoRA": ("LoRA",),
                              "filename_prefix": ("STRING", {"default": "LoRAs/ComfyUI"}),},
                              } 
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "advanced/model_merging"

    def save(self, LoRA, filename_prefix):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        save_file(LoRA, output_checkpoint)

        return {}


NODE_CLASS_MAPPINGS = {
    "DARE Merge LoRA Stack": DARE_Merge_LoraStack,
    "Apply LoRA": ApplyLoRA,
    "Save LoRA": SaveLoRA,
}
