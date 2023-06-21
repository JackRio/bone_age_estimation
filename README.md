# Bone Age Estimation in Left Hand Radiographs
This project focuses on developing a deep learning model for bone age estimation using left hand radiographs. The primary goal is to build a proof of concept that showcases the capabilities of artificial intelligence (AI) in assisting radiologists with this task. The project is a collaborative effort between a radiologist from Mexico and Cardiology national institute "Ignacio Chavez".

## Dataset
The model will be trained on a publicly available dataset provided by the RSNA Pediatric Bone Age Challenge (2017). This dataset consists of a diverse range of X-ray images of pediatric patients along with their corresponding bone age labels. It serves as a comprehensive resource for training and evaluating the proposed deep learning model.

Additionally, the project aims to evaluate the model's performance on a private dataset obtained from a hospital. This evaluation will help assess the model's real-world applicability and provide insights into its effectiveness when applied to different data sources.

## Project Scope and Purpose
The primary objective of this project is to develop a deep learning model that can accurately estimate the bone age of pediatric patients. By leveraging the power of AI, the model intends to assist radiologists by automating and augmenting their bone age assessment process. This collaboration seeks to combine the expertise of a radiologist with the technical skills of an AI student to create a robust and reliable tool.

It is important to note that this project is solely for research purposes and will not be utilized in any commercial aspect. The code may contain re-implementations of existing models, but proper references will be provided to acknowledge the original work.

## Expected Deliverables
- A trained deep learning model capable of estimating bone age from X-ray images.
- A comprehensive evaluation of the model's performance on both public and private datasets.
- Documentation detailing the model architecture, training procedure, and evaluation results.
- Code repository containing the source code, dataset processing scripts, and model implementation.
- Proper attribution and references to any external code or models used in the project.
- By developing and presenting this proof of concept, the project aims to demonstrate the potential of AI in assisting radiologists and advancing medical imaging practices for bone age estimation.  


### Data Preprocessing Pipeline
1. The RSNA dataset is first passed through [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
2. The output of SAM is then filtered based on the score of the segmentation mask and the weighted quantity of voxels
3. The best mask is then post-processed with morphological closing to fill any small holes in the mask
4. The final mask is then used to remove background pixels from the X-ray scans
5. The image is then cropped to the nearest foreground pixel
6. Manual filtering is done to remove any bad images or failed images

### Training
1. The model is trained on the RSNA dataset
2. Pretrained weights from the ImageNet dataset are used to initialize the model
3. ResNet-50 is used as the baseline model
4. ....Steps incoming....
5. The final model will be fintuned and evaluated on the private dataset

### Evaluation
1. The model is evaluated using MAE loss. The MAE loss is calculated by taking the absolute difference between the predicted bone age and the actual bone age.
2. The model is evaluated on the RSNA dataset and private dataset (later)
