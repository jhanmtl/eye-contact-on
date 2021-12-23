This is the project repo for the eye-contact correction project inspired by [Isikdogan et al.](https://arxiv.org/pdf/1906.05378.pdf) 

The final pytorch model is to be converted to the ONNX format for realtimeclient-side inference.

The main challenge of the project will be finding suitable data that accurately captures gaze directions. The plan to overcome this is to use synthetic data from [Unity Eyes](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/) and convert them to more realistic representations with Image-to-Image translation. In particular, the generative architectures and lossese in the paper [Contrastive Learning for Unpaired
Image-to-Image Translation](https://arxiv.org/pdf/2007.15651.pdf) will be implemented as an exercise in my ability to understand and apply modern machine learning research. 
