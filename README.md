# CarModelClassifier
The objetive of this project is to use Convolutional Neural Networks throught fast.ai to predict Car Model based on their pictures.


The model is currently trained using car models from Honda and Toyota brazilian catalogues.

**Requirements:**


***fastai*** version 10.0.61

**prerequisites:** numexpr, pyyaml, dataclasses, pandas, fastprogress, matplotlib, bottleneck, packaging, torch, scipy, numpy, requests, torchvision, spacy, beautifulsoup4, Pillow, nvidia-ml-py3


The model was trained in Google Colab (colab.research.google.com) using a GPU.



**Gathering Training and Validation Data:**

The Test and Validation data were gathered using google_image_search, found here: https://github.com/Joeclinton1/google-images-download.git 
The data was downloaded and cleaned based on higher losses observed during training.

Example code:
```
pip install git+https://github.com/Joeclinton1/google-images-download.git

list_honda_toyota = ['HONDA CIVIC', 'HONDA CITY', 'HONDA FIT', 'HONDA ACCORD', 'HONDA HR-V', 'HONDA WR-V', 
                     'HONDA CR-V','TOYOTA COROLLA', 'TOYOTA YARIS', 'TOYOTA PRIUS', 'TOYOTA RAV4', 'TOYOTA HILUX', 
                     'TOYOTA SW4', 'TOYOTA FIELDER', 'TOYOTA ETIOS', 'TOYOTA CAMRY']
               
               
for modelo in list_honda_toyota:
    modelo = modelo.replace(' ', '+')
    query = str(modelo)
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": query, "limit": 100, "print_urls": True}
    paths = response.download(arguments)
    
```


For Windows Anaconda environment:


```
git clone https://github.com/Joeclinton1/google-images-download.git
cd google-images-download && sudo python setup.py install
```


A list of brands and models available in Brazil is also available in the models.csv file.

