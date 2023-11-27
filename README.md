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

Custom-Diffusion is w/ fine-tune all version.   
The DreamBooth paper does not explicitly specify the particular CLIP model used, so I use separate notation.   

* Using the CLIP to measure Image-Image similarity between:
  * Train images ang generated images
  * Placeholder w/ and placeholder w/o
  * *ProFusion requires a single input image, so evaluation is not feasible.*
  * | |RN50|RN101|RN50x4|RN50x16|RN50x64|ViT-B/32|ViT-B/16|ViT-L/14|ViT-L/14@336px|DreamBooth|
    |:--:|:---:|:-----:|:------:|:------:|:------:|:-------:|:------:|:-----:|:--:|:--:|
    |Custom-Diffusion|0.8548|0.8864|0.8716|0.8371|0.7731|0.8567|0.8585|0.8352|0.8194|-|
    |Custom-Diffusion(Paper)|-|-|-|-|-|0.748|-|-|-|-|
    |DreamBooth|0.8456|0.911|0.8888|0.8545|0.7806|0.8854|0.8987|0.8511|0.8514|-|
    |DreamBooth(Paper)|-|-|-|-|-|-|-|-|-|0.812|-|
    |ProFusion|0.647|0.72575|0.6875|0.596|0.47075|0.643|0.617|0.5735|0.58425|-|
    |Textual-Inversion|0.6942|0.8008|0.7355|0.6104|0.5196|0.7226|0.7271|0.6753|0.6689|-|
    
    
       
* Using the DINO model (ViT-B/32) to measure Image-Image similarity between:
    * Train images and generated images
    * | |DINO (ViT-B/32)|
      |:-:|:-----------:|
      |Custom-Diffusion|0.7881|
      |DreamBooth|0.7951|
      |DreamBooth(Paper)|0.696|
      |ProFusion|0.5311|
      |Textual-Inversion|0.4356|
   
* Using KID to measure Image-Image similarity between:
    * Class images found through retrieval and images generated through prompts for each retrieval (2 images per prompt)
    * In Custom-Diffusion, a scale of 10^3 is used. It has been represented by scaling it down to 1.
    * *ProFusion requires a single input images, so evaluation is not feasible*
    * | |KID|
      |:-:|:-:|
      |Custom-Diffusion|0.0208|
      |Custom-Diffusion(Paper)|0.0193|
      |DreamBooth|0.0355|
      |ProFusion|-|
      |Textual-Inversion|0.0136|
   
 
 * Using a pretrained face recognition model to measute Image-Image similarity between:
    * Train face images and generated face images.
    * Available models :   
    * The original code of AdaFace is   
    https://github.com/mk-minchul/AdaFace
    Installation of face_alignment and pretrained models.
    * All models except AdaFace utilize the deepface library.
      https://github.com/serengil/deepface
    * | |VGG-Face|Facenet|Facenet512|OpenFace|DeepFace|ArcFace|SFace|AdaFace|
      |:-:|:--:|:---:|:-----:|:------:|:-------:|:-------:|:--------:|:--------:|
      |ProFusion|0.6012|0.5523|0.6132|0.6218|0.6689|0.4444|0.4033|0.5413|
      |ProFusion(Paper)|0.720|0.616|0.597|0.681|0.774|0.459|0.443|0.432|
    
* Using the CLIP model to measure Image-Text similarity between:
    * Generated images and prompts used for generating those images.
    * In Custom-Diffusion, a scale of 2.5 is used. It has been represented by scaling it down to 1.
    * | |RN50|RN101|RN50x4|RN50x16|RN50x64|ViT-B/32|ViT-B/16|ViT-L/14|ViT-L/14@336px|DreamBooth|
      |:--:|:---:|:-----:|:------:|:-------:|:-------:|:--------:|:--------:|:--------:|:--------------:|:-:|
      |Custom-Diffusion|0.2800|0.5309|0.4340|0.3295|0.2283|0.3328|0.3265|0.2801|0.2857|-|
      |Custom-Diffusion(Paper)|-|-|-|-|-|0.318|-|-|-|-|
      |DreamBooth|0.2504|0.4756|0.3999|0.2946|0.2265|0.2954|0.2922|0.2308|0.2344|-|
      |DreamBooth(Paper)|-|-|-|-|-|-|-|-|-|0.306|
      |ProFusion|0.2075|0.4420|0.3865|0.2750|0.1782|0.2728|0.2910|0.2357|0.2351|-|
      |ProFusion(Paper)|0.223|0.446|0.374|0.279|0.202|0.293|0.283|0.225|0.229|-|
      |Textual-Inversion|0.2482|0.4587|0.3858|0.2769|0.2205|0.2941|0.3020|0.2347|0.2404|-|

  
The TextualInveersion paper illustrates similarity through images without providing exact numerical values.
![image](https://github.com/seuleepy/Customization/assets/88653864/cff3a486-58e5-4108-9714-f9349291403e)
