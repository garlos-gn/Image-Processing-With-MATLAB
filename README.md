# Image Processing With MATLAB

This repository is a mix of 2 different projects: editing images in the space and frequency domain and segmenting images in order to obtain the amount of coins in the photo.

1. In the "Image preprocessing" folder, we find just one script, which consists of using MATLAB functions in order to:
- Crop images
- Improving visualization
- Applying different LPF in order to smooth the image
- Applying HPF to improve the details of the image
- Joining information from different canals
- Characterization and noise removal
- Pattern detection

2. In the "Image segmentation" folder, there's two different MATLAB scripts:
- In the "segmentation.m" script different segmentation techniques are studied, and finally it is chosen one single implementation. The idea behind the project is to use mathematycal morphollogy (open(s) and close(s)) and binarizing images by using an umbral. At the end of the script there is just one code that classifies the amount of coins and computes the amount of cash in the image.
- The idea behind "segmentation_W.m" is to apply this script to a different scenario, in which the background is white instead of black. Since open and close are dual operations, they can be substituted and (at least theoretically) we should get the same result.
