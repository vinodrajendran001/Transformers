Transformers
==============================

Transformers for sequence classification and text generation

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │   └── Generator.ipynb 
    │   └── Classifier.ipynb 
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   │── utils
    |   │   │   └── util.py
    │   │   ├── predict_classifier.py
    │   │   └── train_classifier.py
    │   │   ├── predict_generator.py
    │   │   └── train_generator.py    
    │   │   └── transformers.py   
    │   │
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------


### Steps to use the repository

Step 1: Git clone the repository

    C:\>git clone https://github.com/vinodrajendran001/Transformers.git


Step 2: Naviagate the cloned repo directory

    C:\transformers>


Step 3: Create a virtual environment and activate it

    C:\transformers> conda create -n transformer python==3.8
    (transformer)C:\transformers>

Step 4: Install all the necessary python libraries 

    (transformer)C:\transformers> pip install -r requirements.txt

#### Text Generator

Step 5: Download the wikipedia dataset http://mattmahoney.net/dc/enwik9.zip and place the zip file in ```data/processed/```

Step 6: Initiate the training

Once the training is completed, a model file will be generated in ```modes/``` folder.

    (transformer)C:\transformer> python src/models/train_generator.py

Step 7: Run the prediction

The aguments for prediction script is text. The predcition script will generate the possible text following the given input text.

    (transformer) C:\transformer>python src/models/predict_generator.py "1228X Human & Rousseau. Because many of his stories were originally published in long-forgotten magazines and journals, there are a number of [[anthology|anthologies]] by different collators each containing a different selection. His original books ha"

#### Sentiment Classifier

Step 8: Initiate the training

IMDB dataset will be downloaded automatically and placed in ```data/processed/``` during training.

Once the training is completed, a model file will be generated in ```modes/``` folder.

    (transformer)C:\transformer> python src/models/train_classifier.py

Step 9: Run the prediction

The aguments for prediction script is text. The predcition script will generate the possible text following the given input text.

    (transformer) C:\transformer>python src/models/predict_classifier.py "If you're going to watch this movie, avoid any spoilers, even spoiler free reviews. Which is why I'm not going to say anything about the movie. Not even my opinion. All I'm going to say is: The crowd applauded 3 times during the movie, and stood up to clap their hands after. This I have never witnessed in a Dutch cinema. Dutch crowds aren't usually passionate about this. I checked the row where I was sitting, and people were crying. After the movie, I was seeing people with smudged mascara. That's all I have to say about the movie."





