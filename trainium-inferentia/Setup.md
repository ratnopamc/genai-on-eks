# Deep Learning On Inf2 Using the  Deep Learning AMI

To find the DLAMI execute the below command. Change the env `AWS_REGION` accordingly where Inferentia2 instances are available.

```
export AWS_REGION=us-west-2
export OS_VERSION="Amazon Linux 2"
DLAMIINF2=$(aws ec2 describe-images --region $AWS_REGION --owners amazon \
--filters 'Name=name,Values=Deep Learning AMI ($OS_VERSION) Version ??.?' 'Name=state,Values=available' \
--query 'reverse(sort_by(Images, &CreationDate))[:1].ImageId' --output text)
```

Follow the prompts on the console and use the value of the DLAMIINF2 to launch an Inf2 EC2 of your choice. 

Make sure to allocate 128 GB of storage on inf2.xlarge instance type. 

Create a kyepair file and download it locally.

Once the EC2 instance is in `Running` state, login to the instance

Locate the .pem file, `cd` into the directory containing the file and make it executable first.

```
chmod 400 <Key-pair-file>.pem
ssh -L localhost:8888:localhost:8888 -i "<Key-pair-file>.pem" ubuntu@<DNS Name of the instance>
```

Once logged in, install jupyter if not already installed using snap.

```
# On Ubutun OS, install using snap
sudo snap install jupyter

# On Amazon Linux 2, below commands have been tested with "ami-0b9d0d380d6d4c7e2".

# install dependencies including neuron, diffusers, transformers and accelerators

source /opt/aws_neuron_venv_pytorch/bin/activate
pip install diffusers==0.16.1
! pip install -U transformers
! pip install -U accelerate

mkdir ~/stable_diffusion
cd ~/stable_diffusion
wget https://ud-workshop.s3.amazonaws.com/sd2_compile_dir_512.tar.gz
tar -zxf sd2_compile_dir_512.tar.gz
rm -f sd2_compile_dir_512.tar.gz
wget https://ud-workshop.s3.amazonaws.com/hf_pretrained_sd2_512_inference.ipynb

```

## SetUp Jupyter Notebook

In the terminal, type `jupyter notebook` as below. If you're using `Ubuntu`, try with `jupyter lab` to launch the Notebook.

```
pip install ipykernel==6.23.1
python3.8 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
jupyter notebook
```


Copy the URL that says `Copy and paste one of these URLs`.
Now you can access the Jupyter Notebook Server from your local browser.


When  you open the notebook make sure to change the kernel to `Python (torch-neuronx)`.

## Pre-compile the diffusion model

We downloaded the pre-compiled file directory `sd2_compile_dir_512`.

First, load the enviornment and parameter settings.

```
import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

import torch
import torch.nn as nn
import torch_neuronx
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import time
import copy
from IPython.display import clear_output

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.cross_attention import CrossAttention

clear_output(wait=False)

```

Next, we need to define the model on Neuron SDk.

```
class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple

class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        sample = self.unetwrap(sample, timestep.float().expand((sample.shape[0],)), encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]
    

# Optimized attention
def get_attention_scores(self, query, key, attn_mask):       
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    # Check for square matmuls
    if(query.size() == key.size()):
        attention_scores = custom_badbmm(
            key,
            query.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=1).permute(0,2,1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = custom_badbmm(
            query,
            key.transpose(-1, -2)
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.to(dtype)
        
    return attention_probs

def custom_badbmm(a, b):
    bmm = torch.bmm(a, b)
    scaled = bmm * 0.125
    return scaled

```

## Reload the model with different prompts.

```
# --- Load all compiled models ---
COMPILER_WORKDIR_ROOT = 'sd2_compile_dir_512'
model_id = "stabilityai/stable-diffusion-2-1-base"
text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Load the compiled UNet onto two neuron cores.
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0,1]
pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

# Load other compiled models onto a single neuron core.
pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
pipe.vae.decoder = torch.jit.load(decoder_filename)
pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)

# Run pipeline
prompt = ["a photo of an astronaut riding a horse on mars",
          "sonic on the moon",
          "elvis playing guitar while eating a hotdog",
          "saved by the bell",
          "engineers eating lunch at the opera",
          "panda eating bamboo on a plane",
          "A digital illustration of a steampunk flying machine in the sky with cogs and mechanisms, 4k, detailed, trending in artstation, fantasy vivid colors",
          "kids playing soccer at the FIFA World Cup"
         ]

plt.title("Image")
plt.xlabel("X pixel scaling")
plt.ylabel("Y pixels scaling")

total_time = 0
for x in prompt:
    start_time = time.time()
    image = pipe(x).images[0]
    total_time = total_time + (time.time()-start_time)
    image.save("image.png")
    image = mpimg.imread("image.png")
    #clear_output(wait=True)
    plt.imshow(image)
    plt.show()
print("Average time: ", np.round((total_time/len(prompt)), 2), "seconds")

```

## Try with different prompts

```
user_input = ""
print("Enter Prompt, type exit to quit")
while user_input != "exit": 
    total_time = 0
    user_input = input("What prompt would you like to give?  ")
    if user_input == "exit":
        break
    start_time = time.time()
    image = pipe(user_input).images[0]
    total_time = total_time + (time.time()-start_time)
    image.save("image.png")

    plt.title("Image")
    plt.xlabel("X pixel scaling")
    plt.ylabel("Y pixels scaling")

    image = mpimg.imread("image.png")
    plt.imshow(image)
    plt.show()
    print("time: ", np.round(total_time, 2), "seconds")

```

The latest generated picture is saved the local directory with name `image.png`.

