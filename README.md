# MuS: Handwritten Text Generation with Multimodal Representation for Accurate Character Structure

This repository contains the official PyTorch implementation for the paper **"Handwritten Text Generation with Multimodal Representation for Accurate Character Structure"**.

Our model, **MuS** (**Mu**ltimodal **S**tructural Generator), introduces a novel framework that leverages both symbolic (text) and visual (rendered image) representations to significantly improve the structural accuracy of generated handwritten text.

![Model Architecture Dark](img/fig2.png?raw=true#gh-dark-mode-only)
![Model Architecture Light](img/fig2.png?raw=true#gh-light-mode-only)
*Overview of the proposed MuS architecture.*

## Key Contributions
- **Multimodal Content Encoder**: Fuses features from both text and its rendered image to capture fine-grained character structures.
- **Text Structure Loss ($L_T$)**: A novel loss function that supervises content accuracy and structural integrity from an image-level perspective.
- **State-of-the-Art Performance**: Achieves leading results on the IAM and CVL datasets in terms of generation quality (FID, GS, HWD) and content accuracy ($\Delta$CER, $\Delta$WER).

![Qualitative Results](img/fig1.png?raw=true)
*Our method (Ours) better preserves character proportions and alignment compared to SOTA methods, especially for challenging words like "way".*

## Installation

1.  **Clone the repository and create a Conda environment:**
    ```console
    # We recommend Python 3.9+
    conda create --name mus python=3.9
    conda activate mus
    ```

2.  **Install PyTorch and other dependencies:**
    ```console
    # Please adjust the CUDA version to match your system
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    
    # Clone this repository
    git clone [https://github.com/your-username/MuS.git](https://github.com/your-username/MuS.git) && cd MuS

    # Install required packages
    pip install -r requirements.txt
    ```

3.  **Download pre-trained models and dataset files:**
    
    From [This folder](https://drive.google.com/drive/folders/13rJhjl7VsyiXlPTBvnp1EKkKEhckLalr?usp=sharing), please download the necessary files (e.g., pre-trained checkpoints, dataset pickles) and place them into the `files/` folder respectively.
    


## Training

To train the MuS model from scratch on the IAM dataset, run the following command:

```console
python train.py 
```

Useful arguments:
```console
python train.py
        --dataset DATASET          # Dataset to use (e.g., IAM, CVL). Default: IAM.
        --data_path PATH           # Path to the pre-processed dataset pickle file.
        --checkpoints_dir PATH     # Directory to save model checkpoints. Default: ./checkpoints
        --resume                   # Resume training from the last checkpoint with the same name.
        --wandb                    # Use Weights & Biases for logging.
```

## Generation / Inference

### Generate Samples for Evaluation

To generate all samples for FID/KID/HWD evaluation (replicating the IAM test set), you can use the following script:

```console
python generate_fakes.py --checkpoint [checkpoint] --generate_type [fid, ocr, iv_s, iv_u, oov_s, oov_u]
```

### Generate Custom Text

To generate a specific text with a given style, provide a folder containing sample images from the desired writer.

```console
python generate_fakes_single.py --style-folder [pqth-stylefolder] \
                    --checkpoint [checkpoint] \
                    --output "results" \
                    --text "May the force be with you."
```

Below is a qualitative comparison showing the output of our model against other methods for the text *"May the force be with you."* and *"Next year is a new beginning."*

![Generation Examples](img/fig4.png?raw=true)

## Acknowledgements
This work is partially inspired by the code and methodologies from previous outstanding research in HTG, including [HWT](https://github.com/ankanbhunia/Handwriting-Transformers) and [VATr](https://github.com/aimagelab/VATr). We thank the authors for making their work publicly available.
