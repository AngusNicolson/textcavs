# TextCAVs

Official implementation of:

Angus Nicolson, Yarin Gal, J. Alison Noble, TextCAVs: Debugging vision models using text, *iMIMIC Workshop @MICCAI 2024*.

A novel text-based interpretability method which uses model gradients and concept activation vectors created using text to explain models without requiring labelled concept data.

## Installation

Git large file storage [(GIT LFS)](https://github.com/git-lfs/git-lfs) is required for cloning the pretrained MIMIC-CXR model.

I used a mixture of conda and pip to install python packages and have not found 
a good way of providing these dependencies which won't require updating in the 
future. So, below is the list of python packages that need installing and 
`requirements.txt` contains the exact versions if you need them.
- pytorch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- transformers
- bitsandbytes
- sentencepiece
- accelerate
- protobuf
- open_clip_torch
- openai-clip
- nltk
- scipy

This works as of 15/08/2024:
```bash
# Create conda environment
conda create -n textcavs python=3.8
# Install pytorch (and CUDA)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
# Install other packages
pip install -r requirements.txt
# Install TextCAVs
pip install -e ./
```

## Quickstart
### ImageNet
Generate explanations for a ResNet50 for a subset of ImageNet classes using a
pretrained version of `h`:
```bash
python generate_explanations.py --exp-name imagenet_example
```
NB: if the default weights for a Torchvision ResNet50 have changed, 
the pretrained version of `h` will not be compatible and will need to be retrained 
(see "From the beginning").

### MIMIC-CXR

Generate explanations for a ResNet50 trained on the MIMIC-CXR dataset for X-ray 
classification using a pretrained `h`. 
```bash
python generate_explanations.py \
--model ./models/example_mimic_model.ckpt \
--layer avgpool \
--concept-dir ./data/text_concepts \
--concept-names mimic_rand_subset_lines \
--class-names "No Finding, Atelectasis, Cardiomegaly, Edema, Pleural Effusion" \
--exp-name mimic_example_00 \
--clip-model BiomedCLIP \
--dataset mimic-cxr \
--no-center-crop
```

## From the beginning
### ImageNet

Create the list of concepts to test
```bash
python run_llm.py
python process_llm_concepts.py
```

Download the ImageNet dataset from 
[image-net.org](https://www.image-net.org/download-images.php)
(`ILSVRC2012_devkit_t12.tar.gz`) and place it in `data/ImageNet`.

Then train the linear transformations `h` and `g`
(NB: this will take a while the first time you run it as it will need to
create all the embeddings):
```bash
python train_feature_converter.py
```

Finally, generate explanations for a subset of ImageNet classes:
```bash
python generate_explanations.py
```

If you wish to generate explanations for different classes, use the `--target-classes` 
argument. Use the same spellings as in `data/class_names_short.txt` and comma 
separate the values, e.g.:

```bash
python generate_explanations.py \
--target-classes "vulture, Komodo dragon, scorpion, American coot, king penguin, Border terrier" \
--out-suffix _animals
```


### MIMIC-CXR

Obtain permission to use and download MIMIC-CXR-JPG from [physionet.org](https://physionet.org/content/mimic-cxr-jpg/2.1.0/). Update `MIMIC_ROOT` from `textcavs/utils.py` to the location of the dataset.

Extract sentences from the clinical reports to use in training:
```bash
python extract_mimic_texts_from_reports.py
```

Then train the linear transformations `h` and `g`:
```bash
python train_feature_converter.py \
--model ./models/example_mimic_model.ckpt \
--layer avgpool \
--text-path data/text_concepts/mimic_unfiltered_lines.txt \
--exp-name mimic_example_01 \
--clip-model BiomedCLIP \
--dataset mimic-cxr \
--no-center-crop
```

Finally, generate explanations for each class:
```bash
python generate_explanations.py \
--model ./models/example_mimic_model.ckpt \
--layer avgpool \
--concept-dir ./data/text_concepts \
--concept-names mimic_rand_subset_lines \
--class-names "No Finding, Atelectasis, Cardiomegaly, Edema, Pleural Effusion" \
--exp-name mimic_example_01 \
--clip-model BiomedCLIP \
--dataset mimic-cxr \
--no-center-crop
```
