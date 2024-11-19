# This will be my approach to a Face Swapping Algorithm
It not yet decided which technique will be used for the swapping itself, 

Previous recherche showed the following:
- There is an akin seesaw between identity replacement and attribute replacement
- SOTA Methods are: MegaFS, FaceSwap, Blendface
- Diffusion based Methods redefine this Problem as a conditional inpainting task (FaceX)
- Transformer models are (ReliableSwap, TransFS, StyleGan2)
- VAE's commonly use KLDivergence, MSE, CrossEntropy, LPIPS and perceptual loss
- Used Models could be VQ-VAE, CVAE-GAN, Transformer, Cnn's 

The most common approach is using a GAN that includes VAE to encode and process different facial regions seperately, allowing to blend faces seamlessly.