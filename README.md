# Mask2Edge
Mask2Edge: Masking Dependencies and Dynamically Capturing Pixel Differences in Edge Detection

## Abstract
Edge detection plays an important role in computer vision tasks. Deep learning-based edge detectors commonly rely on encoding the long and short-term dependencies of pixel values to mine contextual information. They strongly focus on all positions in the image, ignoring the potential issue of over-encoding. Furthermore, most of these models have not attempted to leverage the inherent properties of edges. In this paper, we introduce a query-based edge detector named Mask2Edge, which is capable of masking dependencies and dynamically capturing pixel differences. Specifically, we first devise a masking strategy based on the sparsity of edges to alleviate the over-encoding issue. We propose a Region-guided Masked Attention, which adapts to edge detection and is capable of constraining cross-attention with appropriate masking intensity to extract relatively complete local features. Subsequently, we design a structure to capture the pixel differences that can help identify edges. We introduce dynamic convolutions into edge detection and refine the application scope and generation method of attention weights to effectively perceive changes in pixel gradients. Extensive experiments demonstrate the superiority of Mask2Edge compared with state-of-the-art methods.

# Preparing Data
We use the links in RCF Repository (really thanks for that).
The augmented BSDS500, PASCAL VOC, and NYUD datasets can be downloaded with:

    wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
    wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
    wget http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz

BIPED Dataset is can be downloaded with:

    https://drive.google.com/drive/folders/1lZuvJxL4dvhVGgiITmZsjUJPBBrFI_bM

# Results
Mask2Edge Results for BSDS under a single-scale setting can be found soon.

# Tools
The evaluation program of ODS OIS is here:

    https://github.com/pdollar/edges
The PR curve tool is here:

    https://github.com/MCG-NKU/plot-edge-pr-curves

# Start
python main.py -- mode train


# Acknowledgement & Citation
The code is highly based on [CATS](https://github.com/WHUHLX/CATS) and [
Mask2Former](https://bowenc0221.github.io/mask2former/). Many thanks for their great work.  

