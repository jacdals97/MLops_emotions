# emotions

**Project description**

*A. What is the overall goal of the project?*

We (hypothetically) work at an AI analytics company where we have clients from the private and public sectors who hire us to do analytics for them.

The overall goal of the project is to create an online application where consultants at our company can do some quick and impressive NLP analytics for clients. 
In the application, the consultant will be able to upload a text string for emotion classification. The web app takes a text string, and returns an "emotion prediction" containing the predicted emotion (sadness, joy, love, anger, fear, surprise).
  

A typical use case could be an informal meeting between a consultant and a potential new client, where the consultant takes some publicly available text data relevant to the client (for example comments from reviews on Trustpilot) 
and does some quick analysis on this to impress the potential client.  


*B. What framework are you going to use and do you intend to include the framework into your project?*

We intend to use Huggingface transformers for pytorch in our project. 

*C. What data are you going to run on?*

We use a dataset consisting of 20,000 tweets in English with annotated emotions (sadness, joy, love, anger, fear, surprise). 
This data comes from the DAIR.AI group and was collected and preprocessed for the creation of this paper: https://aclanthology.org/D18-1404.pdf 
The data can be found here: 
https://huggingface.co/datasets/dair-ai/emotion/blob/main/README.md


*D. What models do you expect to use?*

We expect to use the Microsoft E5 Transformer model from Huggingface.  

## How to submit jobs to Vertex AI
We provide a file called *vertex_ai_config.yaml* which contains all the necessary arguments that can be passed to the Vertex AI platform including selecting an image, setting environment variables, accessing secrets and specifying hyperparameters for a job.
```bash
gcloud ai custom-jobs create \
    --region=europe-west2 \   
    --display-name=<run_name> \
    --config=vertex_ai_config.yaml
```

## Project structure

The directory structure of the project looks like this:

```txt

├── .dvc                 <- DVC setup to access and version data
│
├── .github              
│   └── workflows        <- Workflows that run github actions tests
│
├── app                  <- FastAPI app for emotion prediction 
│
├── config               <- Configurations for the experiments 
│
├── dockerfiles          <- dockerfiles and cloudbuild.yaml files for building images locally and in cloud   
│
├── emotions  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
├── reports              <- Final project report
│   └── figures          <- Figures for project report
│
├── tests                <- Test scripts
│
├── .gitignore           <- files to ignore 
│
├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
│  
├── README.md            <- The top-level README for developers using this project.   
│  
├── data.dvc             <- Model versioning
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
│
└── requirements_dev.txt <- The requirements file for reproducing the analysis environment
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template) and [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
