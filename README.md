# Customization

### Model : DreamBooth, TextualInversion, Custom-Diffusion, ProFusion   

In main.py, you can train above models.   

The original codes for DreamBooth, Textual-Inversion, and Custom-Diffusion are   
https://github.com/huggingface/diffusers/tree/main.   
Download the datasets.   
Installation is required.   
```
pip install --upgrade diffusers[torch]
```

The original code for ProFusion is   
https://github.com/drboog/ProFusion/tree/main. 
Download the datasets.   
Installation is required.   
```
git clone https://github.com/drboog/ProFusion.git
cd ProFusion
mv diffusers profusion_diffusers
```
In setup.py, you can replce setup name with the name you prefer.   
```
python setup.py install
```
Change forder ./profusion_diffusers/src/profusion_diffusers/pipelines/pipeline_utils.py to pipeline_utils.py I uploaded.

## Metrics   

* Using the CLIP to measure Image-Image similarity between:
  * Train images ang generated images
  * Placeholder w/ and placeholder w/o
  * *ProFusion requires a single input image, so evaluation is not feasible.*
  * | |RN50|RN101|RN50x4|RN50x16|RN50x64|ViT-B/32|ViT-B/16|ViT-L/14|ViT-L/14@336px|
    |:--:|:---:|:-----:|:------:|:-------:|:-------:|:--------:|:--------:|:--------:|:--------------:|
    |Custom-Diffusion|0.8548|0.8864|0.8716|0.8371|0.7731|0.8567|0.8585|0.8352|0.8194|
    |DreamBooth|0.8456|0.911|0.8888|0.8545|0.7806|0.8854|0.8987|0.8511|0.8514|
    |ProFusion|0.647|0.72575|0.6875|0.596|0.47075|0.643|0.617|0.5735|0.58425|
    |Textual-Inversion|0.6942|0.8008|0.7355|0.6104|0.5196|0.7226|0.7271|0.6753|0.6689|
    
    
       
* Using the DINO model (ViT-B/32) to measure Image-Image similarity between:
    * Train images and generated images
* Using KID to measure Image-Image similarity between:
    * Class images found through retrieval and images generated through prompts for each retrieval (2 images per prompt)
    * *ProFusion requires a single input images, so evaluation is not feasible*
 
 * Using a pretrained face recognition model to measute Image-Image similarity between:
    * Train face images and generated face images.
    * Available models :   
    ["AdaFace", "VGG-Face", "Facenet", "OpenFace", "DeepFace", "ArcFace", "SFace"]
    * The original code of AdaFace is   
    https://github.com/mk-minchul/AdaFace
    Installation of face_alignment and pretrained models.
    * All models except AdaFace utilize the deepface library.
      https://github.com/serengil/deepface
    
* Using the CLIP model to measure Image-Text similarity between:
    * Generated images and prompts used for generating those images.
    * Available models :   
      ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
