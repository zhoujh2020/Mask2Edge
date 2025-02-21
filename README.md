# Mask2Edge
Mask2Edge: Masking Dependencies and Dynamically Capturing Pixel Differences in Edge Detection

## Abstract
Edge detection plays an important role in computer vision tasks. Deep learning-based edge detectors commonly rely on encoding the long and short-term dependencies of pixel values to mine contextual information. They strongly focus on all positions in the image, ignoring the potential issue of over-encoding. Furthermore, most of these models have not attempted to leverage the inherent properties of edges. In this paper, we introduce a query-based edge detector named Mask2Edge, which is capable of masking dependencies and dynamically capturing pixel differences. Specifically, we first devise a masking strategy based on the sparsity of edges to alleviate the over-encoding issue. We propose a Region-guided Masked Attention, which adapts to edge detection and is capable of constraining cross-attention with appropriate masking intensity to extract relatively complete local features. Subsequently, we design a structure to capture the pixel differences that can help identify edges. We introduce dynamic convolutions into edge detection and refine the application scope and generation method of attention weights to effectively perceive changes in pixel gradients. Extensive experiments demonstrate the superiority of Mask2Edge compared with state-of-the-art methods.

# Preparing Data
We rely on the links provided in the RCF Repository (many thanks!). The augmented BSDS500, PASCAL VOC, and NYUD datasets can be downloaded using:
    wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
    wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
    wget http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz

BIPED Dataset can be downloaded with:

    https://drive.google.com/drive/folders/1lZuvJxL4dvhVGgiITmZsjUJPBBrFI_bM

# Results
The Mask2Edge results for BSDS in a single-scale setting will be available soon.

# Tools
The evaluation program for ODS OIS is available at the following link:

    https://github.com/pdollar/edges
The PR curve tool is provided here:

    https://github.com/MCG-NKU/plot-edge-pr-curves

# train
python main.py -- mode train

# test
python main.py -- mode test

# Acknowledgement
This code heavily relies on [CATS](https://github.com/WHUHLX/CATS) and [Mask2Former](https://bowenc0221.github.io/mask2former/). Many thanks to them for their excellent work.  

