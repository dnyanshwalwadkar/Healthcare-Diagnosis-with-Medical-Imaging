# Healthcare-Diagnosis-with-Medical-Imaging
@dnyaneshwalwadkar / d.walwadkar@se21.qmul.ac.uk

## What are AI-powered medical imaging applications? 
Aim of medical imaging is to capture abnormalities using image processing and machine learning techniques. Application areas can be divided into sub-branches such as the diagnosis of various diseases and medical operation planning. The top applications of AI-powered medical imaging are:

### Revealing cardiovascular abnormalities
According to an article published by Frontiers in Cardiovascular Medicine Journal in 2019, the integration of AI into cardiac imaging will accelerate the process of the image analysis which is a repetitive task that can be automated, therefore healthcare professionals engaged in this work can focus on more important tasks.

### Prediction of Alzheimer’s disease
The Radiologic Society of America suggests that advances in AI can lead to predicting Alzheimer’s disease years before it occurs by the identification of metabolic brain changes.

### Cancer detection
In early 2020, the Google health team announced that they developed an AI-based imaging system that outperformed medical professionals in detecting breast cancer.

### Treatment revaluation
This is mostly used for cancer patients undergoing treatment to check if the treatment is working effectively and diminishing the size of the tumor.

### Surgical Planning
Medical imaging also allows for the segmentation of the image related to the surgical area so that the algorithm can do the planning for healthcare professionals automatically. Surgical planning with the help of medical imaging can saves time in surgeries.

### How mature are medical imaging applications?
AI medical imaging applications give great results in research projects and pilots like this one for breast cancer or pancreatic cancer. However, its use in the field is not common yet. This is because the FDA approval process can take years and the applications that were previously approved did not provide significant benefits. A computer-aided cancer detection software, approved by the FDA in 1998, was reported to cost more than $ 400M and did not show any significant improvement in a study.

However, future of AI-powered medical imaging is bright and new advancements happen continuously. For instance, in November 2020, the Technical University of Munich has developed a machine learning model for detecting diabetes induced eye diseases as accurately as humans. 

## How can AI-powered medical imaging technologies be used during COVID-19 outbreak? 
Medical imaging is one of the AI-powered solutions that is on an uptrend with the COVID-19 pandemic. Due to the rapid increase in the number of patients, the analysis and interpretation of patients’ chest scan results became a problem. A Chinese company, Huiying Medical has developed an AI-powered imaging diagnostic solution to detect the virus in the early stage with 96% accuracy. 
Artificial Intelligence is revolutionizing Healthcare in many
areas such as:
Disease Diagnosis with medical imaging
Surgical Robots
* Maximizing Hospital Efficiency
* AI healthcare market is expected to reach $45.2 billion USD
by 2026 from the current valuation of $4.9 billion USD.
* Deep learning has been proven to be superior in detecting
diseases from X-rays, MRI scans and CT scans which could
significantly improve the speed and accuracy of diagnosis.
 
 # In Our Project 
 

# Project Architecture

![alt text](https://github.com/dnyanshwalwadkar/Healthcare-Diagnosis-with-Medical-Imaging/blob/main/Project-Architecture.png)

In this project, I workerd on an AI/ML methods / algorithms & medical diagnosis images.
* I improved the speed and accuracy of
detecting and localizing brain tumors based on MRI scans.
* This would drastically reduce the cost of cancer diagnosis &
help in early diagnosis of tumors which would essentially be a
life saver.
* Used brain MRI scans and have developed a model that could detect and localize tumors.
* With 3929 Brain MRI scans along with
their brain tumour location.


![alt text](https://github.com/dnyanshwalwadkar/Healthcare-Diagnosis-with-Medical-Imaging/blob/main/Brain-Tumour-detected.png)

# CONVOLUTIONAL NEURAL NETWORKS
(REVIEW)
* The first CNN layers are used to extract high level general
features.
* The last couple of layers are used to perform classification (on
a specific task).
* Local respective fields scan the image first searching for
simple shapes such as edges/lines
* These edges are then picked up by the subsequent layer to
form more complex features.

![alt text](https://github.com/dnyanshwalwadkar/Healthcare-Diagnosis-with-Medical-Imaging/blob/main/CNN-Architecture.png)

# RESNET (RESIDUAL NETWORK) (REVIEW)
* As CNNs grow deeper, vanishing gradient tend to occur which
negatively impact network performance.
* Vanishing gradient problem occurs when the gradient is backpropagated
to earlier layers which results in a very small
gradient.
* Residual Neural Network includes “skip connection” feature
which enables training of 152 layers without vanishing gradient
issues.
* Resnet works by adding “identity mappings” on top of the CNN.
* ImageNet contains 11 million images and 11,000 categories.
* ImageNet is used to train ResNet deep network.

# TRANSFER LEARNING?
* Transfer learning is a machine learning technique in which a
network that has been trained to perform a specific task is
being reused (repurposed) as a starting point for another
similar task.
* Transfer learning is widely used since starting from a pretrained
models can dramatically reduce the computational
time required if training is performed from scratch.

# WHAT IS IMAGE SEGMENTATION?
* Recall when we applied CNN for image classification
problems? We had to convert the image into a vector and
possibly add a classification head at the end.
* However, in case of Unet, we convert (encode) the image
into a vector followed by up sampling (decode) it back again
into an image.
* In case of Unet, the input and output have the same size so
the size of the image is preserved.
* For classical CNNs: they are generally used when the entire
image is needed to be classified as a class label.
* For Unet: pixel level classification is performed.
* U-net formulates a loss function for every pixel in the input
image.
* Softmax function is applied to every pixel which makes the
segmentation problem works as a classification problem
where classification is performed on every pixel of the image.

# RESUNET
* ResUNet architecture combines UNet backbone architecture with residual blocks to
overcome the vanishing gradients problems present in deep architectures.
* Unet architecture is based on Fully Convolutional Networks and modified in a way
that it performs well on segmentation tasks.
* Resunet consists of three parts:
* (1) Encoder or contracting path
* (2) Bottleneck
* (3) Decoder or expansive path

![alt text](https://github.com/dnyanshwalwadkar/Healthcare-Diagnosis-with-Medical-Imaging/blob/main/RESUNET.png)

# RESUNET Architecture
## CONTRACTION PATH
(ENCODER)
* The contraction path consist of several contraction blocks, each block takes an input that passes through res-blocks
followed by 2x2 max pooling. Feature maps after each block doubles, which helps the model learn complex
features effectively.
## BOTTLENECK
* The bottleneck block, serves as a connection between contraction path and expansion path.
* The block takes the input and then passes through a res-block followed by 2x2 up-sampling
convolution layers.
## EXPANSION PATH (DECODER)
* Significant advantage of this architecture lies in expansion or decoder section. Each block takes in the
up-sampled input from the previous layer and concatenates with the corresponding output features from the
res-blocks in the contraction path. This is then again passed through the resblock followed by 2x2 up-sampling
convolution layers.
* This helps to ensure that features
learned while contracting are used
while reconstructing the image.
* Finally in the last layer of expansion path, the output from the res-block is passed through 1x1 convolution layer to
produce the desired output with the same size as the input.

# RESUNET ARCHITECTURE:
1. Encoder or contracting path consist of 4 blocks:
* First block consists of 3x3 convolution layer + Relu + Batch-
Normalization
* Remaining three blocks consist of Res-blocks followed by
Max-pooling 2x2.
2. Bottleneck:
* It is in-between the contracting and expanding path.
* It consist of Res-block followed by up sampling conv layer
2x2.
3. Expanding or Decoder path consist of 4 blocks:
* 3 blocks following bottleneck consist of Res-blocks followed
by up-sampling conv layer 2 x 2
8 Final block consist of Res-block followed by 1x1 conv layer.

![alt text](https://github.com/dnyanshwalwadkar/Healthcare-Diagnosis-with-Medical-Imaging/blob/main/Total-Output.png)

Thank You.
Hope you able to understand every point, concept & Architeture.
for more info do contact me
d.walwadkar@se21.qmul.ac.uk
