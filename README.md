# DNAQuantification
Repository containing files needed for DNA content quantification on TEM images

# Installation instructions
To set up the DNA content quantification workflow, you need to perform the following steps:

1. Clone the *DNAQuantification* repository. To do so, either use the green button **_Code_** in the upper right corner of this website or [git](https://git-scm.com/downloads) bash with the following command: 
```
git clone https://github.com/roessler-f/DNAQuantification
```

2. Download and install the image processing software **Fiji** to run the macro script *contrastEnhancementAndScale.ijm*. You can download the software and find installation instructions using this [link](https://imagej.net/software/fiji/).

3. Download and install the Python distribution **Anaconda** to run the script *PixelClassifier.py*. You can install Anaconda using this [link](https://www.anaconda.com/products/individual).

4. After you downloaded and installed Anaconda, open the Anaconda prompt for Windows or the terminal window for macOS and Linux and create a new conda environment using the following command:
```
conda create -n DNAquantification python=3.7
```

5. Then go to the downloaded *DNAQuantification* repository, activate the conda environment and install all necessary dependencies using the following commands: 
```
cd path\to\DNAquantification_repository
conda activate DNAquantification
pip install -r python_requirements.txt
```

6. Download and install **MATLAB** to run the script *measureDNALength.m*. You can get MATLAB using this [link](https://se.mathworks.com/products/get-matlab.html). Additionally to the basic version of MATLAB, you need the **[Image Processing Toolbox](https://se.mathworks.com/help/images/getting-started-with-image-processing-toolbox.html)** to run the script. 

# Setup before use
How to run the DNA content quantification workflow is described in the following book chapter: ...

Before running the Python and the MATLAB script, you need to perform the following steps:
- In the Python script **_PixelClassifier.py_**, you need to define (1) an **input folder** containing the contrast-normalized and rescaled images, (2) an **output folder** where the segmented images will be saved to, (3) the **path to the checkpoint files** of the trained convolutional neural network (CNN) provided in this repository and (4) the **pixel size of the rescaled images**. Define these 4 variables in the *PixelClassifier.py* file on line 34-40 (you can open the file using a simple text editor for example):
```
# Define path to input folder containing contrast-normalized and rescaled images:
input_path = ...
# Define path to output folder (segmented images will be saved there):
output_path = ...
# Define path to checkpoint file of CNN (needs to look like this: path + 'trained_PixelClassifier.ckpt'):
save_path = ...
# Define size of rescaled images in pixel (!!! Always one pixel bigger than actual size !!!):
whole_image_size = ...							# e.g. whole_image_size = 1025
```

- In the MATLAB script **_measureDNALength.m_**, you need to define an **input folder** containing the segmented images. Define this path on line 9 of the script: 
```
% TO DO: Define path to folder containing segmented images
inputpath = ...;
```

👉 **ATTENTION:** All scripts expect the input folders to contain the images acquired in one tile set to be saved in a subfolder (e.g. “…/input folder/tile set 1/”, “…/input folder/tile set 2/” etc.). So when defining an input folder, make sure to use the correct directory (= not a single tile set folder).   
