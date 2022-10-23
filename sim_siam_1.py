

## Same as LIGHTLY -- But FAIR -- Now deprecated-- https://github.com/facebookresearch/vissl
## Same as LIGHTLY -- But FAIR -- Now deprecated-- FAIR Self-Supervision Benchmark is deprecated. 
#Please see VISSL, a ground-up rewrite of benchmark in PyTorch.

## TODO -- Book -- Weak Supervision -- https://www.oreilly.com/library/view/practical-weak-supervision/9781492077053/
## TODO -- Book -- Weak Supervision -- https://github.com/practicalweaksupervisionbook/companion


#Source #TODO --LIGHTLY_at_a_GLANCE 
## TODO -- Alternative Source -LIGHTLY_at_a_GLANCE- - https://github.com/lightly-ai/lightly/blob/master/docs/source/getting_started/lightly_at_a_glance.rst
## TODO -- Alternative Source -LIGHTLY_at_a_GLANCE__JUPYTER_NOTEBOOK -- https://colab.research.google.com/drive/1ubepXnpANiWOSmq80e-mqAxjLx53m-zu?usp=sharing#scrollTo=p_GoGjxWPw1w
# TODO -- MAIN CONCEPTS https://github.com/lightly-ai/lightly/blob/master/docs/source/getting_started/main_concepts.rst
## TODO -- https://github.com/lightly-ai/lightly/blob/master/lightly/models/simsiam.py
## TODO -- dataLoader_MemoryOptimize --- each element in a list has its own refcount, while a numpy array only has a single refcount. 
## TODO -- dataLoader_MemoryOptimize >> https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
## TODO -- MLP Head -- also seen in the MAIN.py from the GIT -- https://github.com/IgorSusmelj/simsiam-cifar10
"""
        # create a simsiam model based on ResNet
        self.resnet_simsiam = \
            lightly.models.SimSiam(self.backbone, num_ftrs=512)# TODO -- Original Code , num_mlp_layers=2) 
"""


import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch.__version__:", torch.__version__)
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)


import math
import torch.nn as nn #https://pytorch.org/tutorials/beginner/nn_tutorial.html
import torchvision
import numpy as np
import lightly
from lightly.models.modules.heads import SimSiamPredictionHead # TODO # Lightly_original_code\models\modules\heads.py
from lightly.models.modules.heads import SimSiamProjectionHead

# The default configuration with a batch size and input resolution of 256
# requires 16GB of GPU memory.

num_workers = 8 ## Multi-process Data Loading >> https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
batch_size = 128
seed = 1
epochs = 50
input_size = 256
# dimension of the embeddings
num_ftrs = 512
# dimension of the output of the prediction and projection heads
out_dim = proj_hidden_dim = 512 #TODO -SimSiamProjectionHead == See Layers Structure in OWN Terminal LOGS FIle -- (projection_head): SimSiamProjectionHead( (layers): Sequential(
# the prediction head uses a bottleneck architecture
pred_hidden_dim = 128
# seed torch and numpy
torch.manual_seed(0)
np.random.seed(0)

# set the path to the dataset
path_to_data = '/home/dhankar/temp/04_22/Images/not_planes'

# Setup data augmentations and loaders
# ------------------------------------
# Since we're working on satellite images, it makes sense to use horizontal and
# vertical flips as well as random rotation transformations. 
# We apply weak color 
# jitter to ---learn an invariance---- of the model with respect to slight changes in
# the color of the water.
#

# define the augmentations for self-supervised learning
# TODO # Lightly_original_code\collate.py

collate_fn = lightly.data.ImageCollateFunction(
    input_size=input_size,
    # require invariance to flips and rotations
    hf_prob=0.5,
    vf_prob=0.5,
    rr_prob=0.5,
    # satellite images are all taken from the same height
    # so we use only slight random cropping
    min_scale=0.5,
    # use a weak color jitter for invariance w.r.t small color changes
    cj_prob=0.2,
    cj_bright=0.1,
    cj_contrast=0.1,
    cj_hue=0.1,
    cj_sat=0.1,
)

# create a lightly dataset for training, since the augmentations are handled
# by the collate function, there is no need to apply additional ones here
dataset_train_simsiam = lightly.data.LightlyDataset(input_dir=path_to_data) 
## Provides a uniform data interface for the embedding models. >> https://docs.lightly.ai/lightly.data.html
print("---type(dataset_train_simsiam)-----",type(dataset_train_simsiam)) ##- <class 'lightly.data.dataset.LightlyDataset'>
sample, target, fname = dataset_train_simsiam[0]
print("----fname--dataset_train_simsiam---",fname) ## 10378780_15.tiff
print("----sample--dataset_train_simsiam---",sample) #
print("----target--dataset_train_simsiam---",target) #



# create a dataloader for training
print("----batch_size---Original Code -- 128 --",batch_size) # 128 
rohit_train_batch_size = 6 ## 52 TIFF FIles here -- TIFF KAGGLE DATA 
dataloader_train_simsiam = torch.utils.data.DataLoader(dataset_train_simsiam,
    batch_size=rohit_train_batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True, ## WHY -- TRUE 
    num_workers=num_workers
)

# create a torchvision transformation for embedding the dataset after training
# here, we resize the images to match the input size during training and apply
# a normalization of the color channel based on statistics from imagenet
test_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((input_size, input_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=lightly.data.collate.imagenet_normalize['mean'],
                    std=lightly.data.collate.imagenet_normalize['std'],
                )
            ])

# create a lightly dataset for embedding
dataset_test = lightly.data.LightlyDataset(input_dir=path_to_data,transform=test_transforms)


# create a dataloader for embedding
rohit_test_batch_size = 6 
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=rohit_test_batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# Create the SimSiam model
# ------------------------
# Create a ResNet backbone and remove the classification head

class SimSiam(nn.Module):
    """ # TODO -- https://github.com/lightly-ai/lightly/blob/master/lightly/models/simsiam.py
        #SimSiamProjectionHead == See Layers Structure in OWN LOGS FIle -- (projection_head): SimSiamProjectionHead( (layers): Sequential(
    """
    def __init__(
        self, backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(
            num_ftrs, proj_hidden_dim, out_dim
        )
        self.prediction_head = SimSiamPredictionHead(
            out_dim, pred_hidden_dim, out_dim
        )

    def forward(self, x):
        # get representations
        f = self.backbone(x).flatten(start_dim=1)
        # get projections
        z_projection_head = self.projection_head(f)
        # get predictions
        p_prediction_head = self.prediction_head(z_projection_head)
        print("--[INFO_forward]--Type(p--from--prediction_head-",type(p_prediction_head))
        # stop gradient
        z_projection_head_detached = z_projection_head.detach()
        print("--[INFO_forward]---Type(z_projection_head_detached---",type(z_projection_head_detached))

        return z_projection_head_detached, p_prediction_head


# we use a pretrained resnet for this tutorial to speed
# up training time but you can also train one from scratch
resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1]) #
#remove the classification head
## get all layers except linear layer
## TODO -- Above same as -- resnet = torch.nn.Sequential(*list(resnet.children())[:-1]) 
#Source #TODO --LIGHTLY_at_a_GLANCE 

model = SimSiam(backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim)
print("----Model_SimSiam---model---\n",model)

# SimSiam uses a symmetric negative cosine similarity loss and does therefore
# not require any negative samples. We build a criterion and an optimizer.
criterion = lightly.loss.NegativeCosineSimilarity()

## TODO == https://discuss.pytorch.org/t/about-cosine-similarity-how-to-choose-the-loss-function-and-the-network-i-have-two-plans/95245/4


# scale the learning rate 
lr = 0.05 * batch_size / 256
# use SGD with momentum and weight decay
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=5e-4
)

#TRAIN 

# Train SimSiam
# --------------------
# 
# To train the SimSiam model, you can use a classic PyTorch training loop:
# For every epoch, iterate over all batches in the training data, extract
# the two transforms of every image, pass them through the model, and calculate
# the loss. Then, simply update the weights with the optimizer. Don't forget to
# reset the gradients!
#
# Since SimSiam doesn't require negative samples, it is a good idea to check 
# whether the outputs of the model have collapsed into a single direction. For
# this we can simply check the standard deviation of the L2 normalized output
# vectors. If it is close to one divided by the square root of the output 
# dimension, everything is fine (you can read
# up on this idea `here <https://arxiv.org/abs/2011.10566>`_).

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

avg_loss = 0.
avg_output_std = 0.
for e in range(epochs):
    print("----type(dataloader_train_simsiam----",type(dataloader_train_simsiam))
    #<class 'torch.utils.data.dataloader.DataLoader'>

    for (x0, x1), _,_ in dataloader_train_simsiam:
        print("----type(dataloader_train_simsiam--aa--",type(dataloader_train_simsiam))        
        # print("--var_1--",var_1)
        # print("--var_2--",var_2)
        # move images to the gpu
        print("-Train SimSiam-type---x0--",type(x0))
        x0 = x0.to(device)
        x1 = x1.to(device)

        # run the model on both transforms of the images
        # we get projections (z0 and z1) and
        # predictions (p0 and p1) as output

        z0, p0 = model(x0)
        z1, p1 = model(x1)
        print("-Train SimSiam-type---p1--",type(p1))
        print("-Train SimSiam-type---p1--",p1)

        # apply the symmetric negative cosine similarity
        # and run backpropagation
        loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
        print("-Train SimSiam-type-loss-",type(loss))
        loss.backward() ## TODO -- Pytorch Code details 
        optimizer.step() # TODO -- Pytorch Code details 
        optimizer.zero_grad() # TODO -- Pytorch Code details 

        # calculate the per-dimension standard deviation of the outputs
        # we can use this later to check whether the embeddings are collapsing
        output = p0.detach()
        output = torch.nn.functional.normalize(output, dim=1)
        
        output_std = torch.std(output, 0)
        output_std = output_std.mean()

        # use moving averages to track the loss and standard deviation
        w = 0.9
        avg_loss = w * avg_loss + (1 - w) * loss.item()
        avg_output_std = w * avg_output_std + (1 - w) * output_std.item()

    # the level of collapse is large if the standard deviation of the l2
    # normalized output is much smaller than 1 / sqrt(dim)
    collapse_level = max(0., 1 - math.sqrt(out_dim) * avg_output_std)
    # print intermediate results
    print(f'[Epoch {e:3d}] '
        f'Loss = {avg_loss:.2f} | '
        f'Collapse Level: {collapse_level:.2f} / 1.00')


## GET_EMBEDDINGS
# To embed the images in the dataset we simply iterate over the test dataloader
# and feed the images to the model backbone. Make sure to disable gradients for
# this part.

embeddings = []
filenames = []

# disable gradients for faster calculations
model.eval()
with torch.no_grad():
    for i, (x, _, fnames) in enumerate(dataloader_test):
        print("--filenmes---for--Embeddings",fnames)
        # move the images to the gpu
        x = x.to(device)
        # embed the images with the pre-trained backbone
        y = model.backbone(x).flatten(start_dim=1)
        print("---type-embeddings--y--aaa-",type(y))

        # store the embeddings and filenames in lists
        embeddings.append(y)
        filenames = filenames + list(fnames)

# concatenate the embeddings and convert to numpy
#print("---type-embeddings---bbb-1",type(embeddings)) ## LIST 
print("---LIST---embeddings---bbb-1",embeddings) ## LIST 
embeddings = torch.cat(embeddings, dim=0) ##concatenate >> https://discuss.pytorch.org/t/how-to-concatenate-list-of-pytorch-tensors/1350/3
#print("---type-embeddings---bbb-2",type(embeddings)) ##<class 'torch.Tensor'>
embeddings = embeddings.cpu().numpy()
print("---type-embeddings---bbb-3",type(embeddings))


# Scatter Plot and Nearest Neighbors
# ----------------------------------
# Now that we have the embeddings, we can visualize the data with a scatter plot.
# Further down, we also check out the nearest neighbors of a few example images.
#
# As a first step, we make a few additional imports. 

# for plotting
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.offsetbox as osb
from matplotlib import rcParams as rcp
# for resizing images to thumbnails
import torchvision.transforms.functional as functional
# for clustering and 2d representations
from sklearn import random_projection #
# https://scikit-learn.org/stable/modules/random_projection.html

# Transform embeddings using UMAP and rescale them to fit in the  [0, 1] square.

# for scatter plot we transform EMBEDDINGS (NOT-images) to a two-dimensional vector space using a random Gaussian projection
## TODO - Transformer_Random_Projection -- # https://scikit-learn.org/stable/modules/random_projection.html

projection = random_projection.GaussianRandomProjection(n_components=2) 
embeddings_2d = projection.fit_transform(embeddings)
#print("---type-embeddings_2d---GaussianRandomProjection--1--",type(embeddings_2d)) #<class 'numpy.ndarray'>
print("----embeddings_2d---GaussianRandomProjection--1--\n",embeddings_2d.ndim,embeddings_2d.shape) #<class 'numpy.ndarray'>
print("----embeddings_2d---GaussianRandomProjection--1--\n",embeddings_2d) 

# Normalize the embeddings to fit in the [0, 1] square
M = np.max(embeddings_2d, axis=0) # Maxima along the first axis
m = np.min(embeddings_2d, axis=0)  # Minima along the first axis 
embeddings_2d = (embeddings_2d - m) / (M - m)
#print("---type-embeddings_2d---GaussianRandomProjection--2--",type(embeddings_2d)) #<class 'numpy.ndarray'>
print("----embeddings_2d---GaussianRandomProjection--2--\n",embeddings_2d.ndim,embeddings_2d.shape) 
#<class 'numpy.ndarray'>
print("----embeddings_2d---GaussianRandomProjection--2--\n",embeddings_2d) 


def get_scatter_plot_with_thumbnails():
    """Creates a scatter plot with image overlays.
    """
    # initialize empty figure and add subplot
    fig = plt.figure()
    fig.suptitle('Scattr_Plot-Test_CODE')
    ax = fig.add_subplot(1, 1, 1)
    # shuffle images and find out which images to show
    shown_images_idx = []
    shown_images = np.array([[1., 1.]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    print("--get_scatter--type(iterator------",type(iterator)) ## LIST
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        print("-----embeddings_2d[i]----",embeddings_2d[i])
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        print("-----embeddings_2d-->>dist---",dist)
        if np.min(dist) < 2e-3:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)

    # plot image overlays
    print("--All Images to Plot ---shown_images_idx--",shown_images_idx)
    for idx in shown_images_idx:
        
        thumbnail_size = int(rcp['figure.figsize'][0] * 2.)
        path = os.path.join(path_to_data, filenames[idx])
        print("-Local_path__Images to Plot ---",path)
        img = Image.open(path)
        img = functional.resize(img, thumbnail_size)
        img = np.array(img)
        img_box = osb.AnnotationBbox(
            osb.OffsetImage(img, cmap=plt.cm.gray_r),
            embeddings_2d[idx],
            pad=0.2,
        )
        ax.add_artist(img_box)

    # set aspect ratio
    ratio = 1. / ax.get_data_ratio()
    ax.set_aspect(ratio, adjustable='box')
    plot_name = "test_code_1"
    plt.savefig('plots_dir/_'+str(plot_name)+"_.pdf", bbox_inches='tight')


# get a scatter plot with thumbnail overlays
get_scatter_plot_with_thumbnails()
print("--test_code_1-ScatterPlot--Saved-")



# Next, we plot example images and their nearest neighbors (calculated from the
# embeddings generated above). This is a very simple approach to find more images
# of a certain type where a few examples are already available. 
# 
# For example,
# when a subset of the data is already labelled and one class of images is clearly
# underrepresented, one can easily query more images of this class from the 
# unlabelled dataset.
#
# Let's get to work! The plots are shown below.

# example_images = [
#     'S2B_MSIL1C_20200526T101559_N0209_R065_T31TGE/tile_00154.png', # water 1
#     'S2B_MSIL1C_20200526T101559_N0209_R065_T32SLJ/tile_00527.png', # water 2
#     'S2B_MSIL1C_20200526T101559_N0209_R065_T32TNL/tile_00556.png', # land
#     'S2B_MSIL1C_20200526T101559_N0209_R065_T31SGD/tile_01731.png', # clouds 1
#     'S2B_MSIL1C_20200526T101559_N0209_R065_T32SMG/tile_00238.png', # clouds 2
# ]



example_images = [
    'image_0009.jpg', 
    'image_0026.jpg', 
    'image_0083.jpg', 
    'image_0103.jpg', 
    'image_0240.jpg', 
]


def get_image_as_np_array(filename: str):
    """Loads the image with filename and returns it as a numpy array.

    """
    img = Image.open(filename)
    return np.asarray(img)


def get_image_as_np_array_with_frame(filename: str, w: int = 5):
    """Returns an image as a numpy array with a black frame of width w.

    """
    img = get_image_as_np_array(filename)
    ny, nx, _ = img.shape
    # create an empty image with padding for the frame
    framed_img = np.zeros((w + ny + w, w + nx + w, 3))
    framed_img = framed_img.astype(np.uint8)
    # put the original image in the middle of the new one
    framed_img[w:-w, w:-w] = img
    return framed_img


def plot_nearest_neighbors_3x3(example_image: str, i: int):
    """Plots the example image and its eight nearest neighbors.

    """
    n_subplots = 6
    # initialize empty figure
    fig = plt.figure()
    #fig.suptitle(f"Nearest Neighbor Plot {i + 1}")
    #
    image_index = i
    example_idx = filenames.index(example_image)
    #print("--example_idx----",example_idx) #
    # INDEX of Images in own defined List above 
    # get distances to the cluster center
    print("--plot_nn--nearest_neighbors--embeddings[example_idx]----",embeddings[example_idx])
    distances = embeddings - embeddings[example_idx]
    print("-plot_nn--nearest_neighbors--distances----",distances)

    distances = np.power(distances,2).sum(-1).squeeze()
    # sort indices by distance to the center
    nearest_neighbors = np.argsort(distances)[:n_subplots]
    nearest_neighbors_all = np.argsort(distances)

    print("--plot_nn--nearest_neighbors--SORTED_DISTANCES-",nearest_neighbors)
    print("--plot_nn--nearest_neighbors_all----SORTED_DISTANCES-",nearest_neighbors_all)
    
    # show images
    for plot_offset, plot_idx in enumerate(nearest_neighbors):
        ax = fig.add_subplot(3, 3, plot_offset + 1)
        # get the corresponding filename
        fname = os.path.join(path_to_data, filenames[plot_idx])
        print("--plot_nn--Example Image--filenames--",fname)
        print("-plot_nn--nearest_neighbors---plot_offset-",plot_offset)
        
        file_name = fname.rsplit("/",1)[1] 
        print("--plot_nn--Example Image--filenames-- -",file_name)
        example_image_name = file_name.rsplit("_",1)[0] 
        print("--plot_nn--Example Image--filenames---",example_image_name)

        # plot_name = "nn_rohit_test"
        str_name = str(image_index) + "_k_N_Neighbours"
        #ax.set_title(str_name)

        # if plot_offset == 0: 
        #     print("--plot_nn--Example Image--filenames--",fname)
        #     file_name = fname.rsplit("/",1)[1] 
        #     print("--plot_nn--Example Image--filenames----",file_name)
        #     example_image_name = file_name.rsplit("_",1)[0] 
        #     print("--plot_nn--Example Image--filenames---",example_image_name)
        # else:
        #     pass

        if plot_offset == 0:
            print("----image_index---aa-",image_index)
            ax.set_title(f"Example Image")
            plt.imshow(get_image_as_np_array_with_frame(fname))
            plt.axis("off")
            #plt.savefig('plots_dir/_example_'+str(example_image_name)+"_.pdf", bbox_inches='tight')
        else:
            plt.imshow(get_image_as_np_array(fname))
            print("----image_index--bbb--",image_index)

            plt.axis("off")
            ax.set_title(str(plot_offset))
            # plot_name = example_image_name
            plt.savefig('plots_dir/_nn_2_'+str(str_name)+"_.pdf", bbox_inches='tight')
        # let's disable the axis
        #plt.axis("off")

# show example images for each cluster
for i, example_image in enumerate(example_images):
    plot_nearest_neighbors_3x3(example_image, i)
