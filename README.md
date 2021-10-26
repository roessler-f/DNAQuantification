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

# Setup before first use
How to run the DNA content quantification workflow is described in the following book chapter: ...
Before running the Python and the MATLAB script, you need to perform the following steps:
1. ...
