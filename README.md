# Customization

### Model : DreamBooth, TextualInversion, CustomDiffusion, ProFusion   

In main.py, you can train above models.   

The original codes for DreamBooth, TextualInversion, and CustomDiffusion are   
https://github.com/huggingface/diffusers/tree/main.   
Installation is required.   
```
pip install --upgrade diffusers[torch]
```

The original code for ProFusion is   
https://github.com/drboog/ProFusion/tree/main.   
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
