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

def spectral_norm(W, u=None, Num_iter=10):
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

def apply_dare(delta, p):
    # Generate the mask m^t from Bernoulli distribution
    m = torch.bernoulli(torch.full(delta.shape, p)).to(delta.dtype)
    # Apply the mask to the delta to get δ̃^t
    delta_tilde = m * delta
    # Scale the masked delta by the dropout rate to get δ̂^t
    delta_hat = delta_tilde / (1 - p)
    return delta_hat

def apply_spectral_norm(lora_weights, scale):
    lips = []
    for key in lora_weights.keys():
        if "alpha" in key:
            continue
        name = key.split(".")[0]
        sn = spectral_norm(lora_weights[key])[0].cpu()
        lips.append(sn)

    sn = max(lips)
    #print("Regularizing lipschitz ", sn, "to", scale)
    for key in lora_weights.keys():
        if("alpha" not in key):
            lora_weights[key] *= scale / sn

    return lora_weights

def merge_weights(lora, p, lambda_val, scale, strength, count_merged):
    merged_tensors = {}
    keys = set(lora.keys())

    for key in keys:
        diff = strength * lambda_val * apply_dare(lora[key], p)
        merged_tensors[key] = diff
        name = key.split(".")[0]

    return merged_tensors

class DARE_Merge_LoraStack:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                            "lora_stack": ("LORA_STACK", ),
                            "lambda_val": ("FLOAT", {"default": 1.5, "min": -4.0, "step": 0.1, "max": 4.0}),
                            "p": ("FLOAT", {"default": 0.13, "min": 0.01, "step": 0.01, "max": 1.0}),
                            "scale": ("FLOAT", {"default": 0.2, "min": -1, "step": 0.001, "max": 10000.0}),
                            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                            }
        }

    RETURN_TYPES = ["LoRA"]
    RETURN_NAMES = ["LoRA"]
    FUNCTION = "apply_lora_stack"
    CATEGORY = "Comfyroll/IO"

    def apply_lora_stack(self, lora_stack, lambda_val, p, scale, seed):
        # Initialise the list
        lora_list = list()
        torch.manual_seed(seed)
        # Extend lora_list with lora-stack items 
        if lora_stack:
            lora_list.extend(lora_stack)
        else:
            return [None]

        # Initialise the model and clip
        lora_name, strength_model, strength_clip = lora_list[0]
        lora_path = folder_paths.get_full_path("loras", lora_name)

        lora0 = comfy.utils.load_torch_file(lora_path, safe_load=True)
        weights = {}
        for key in lora0.keys():
            weights[key]=torch.zeros_like(lora0[key])

        # Loop through the list
        for i in range(len(lora_list)):
            lora_name, strength_model, strength_clip = lora_list[i]
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

            lora_weights = merge_weights(lora, p, lambda_val, 1, strength_model, len(lora_list))
            for key in weights.keys():
                weights[key] += lora_weights[key]

        if scale > 0:
            weights = apply_spectral_norm(weights, scale)
        for key in weights.keys():
            if("alpha" in key):
                weights[key]=torch.ones_like(weights[key])

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
