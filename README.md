# emotions

**Project description**

*A. What is the overall goal of the project?*

We (hypothetically) work at an AI analytics company where we have clients from the private and public sectors who hire us to do analytics for them.

The overall goal of the project is to create an online application where consultants at our company can do some quick and impressive NLP analytics for clients. 
In the application, the consultant will be able to upload a text string for emotion classification. The web app takes a text string, and returns an "emotion prediction" containing the predicted emotion (sadness, joy, love, anger, fear, surprise).
  

A typical use case could be an informal meeting between a consultant and a potential new client, where the consultant takes some publicly available text data relevant to the client (for example comments from reviews on Trustpilot) and does some quick analysis on this to impress the potential client.  

The web service can be accessed at: https://my-fastapi-o3qqudwy2q-ew.a.run.app

*B. What framework are you going to use and do you intend to include the framework into your project?*

We intend to use Huggingface transformers for pytorch in our project. 

*C. What data are you going to run on?*

We use a dataset consisting of 20,000 tweets in English with annotated emotions (sadness, joy, love, anger, fear, surprise). 
This data comes from the DAIR.AI group and was collected and preprocessed for the creation of this paper: https://aclanthology.org/D18-1404.pdf 
The data can be found here: 
https://huggingface.co/datasets/dair-ai/emotion/blob/main/README.md


*D. What models do you expect to use?*

We expect to use the DistilBERT base model (uncased) model from Huggingface.  

## How to submit jobs to Vertex AI
We provide a file called *vertex_ai_config.yaml* which contains all the necessary arguments that can be passed to the Vertex AI platform including selecting an image, setting environment variables, accessing secrets and specifying hyperparameters for a job.
```bash
gcloud ai custom-jobs create \
    --region=europe-west2 \   
    --display-name=<run_name> \
    --config=vertex_ai_config.yaml
```

## How to deploy service using FastAPI

We built a trigger workflow on Google Cloud Build. The trigger is manual and will, once invoked, build an image using the *dockerfiles/cloudbuild_predict.yaml*, push it to the Artifact Registry and deploy a service without traffic. If it is the first time the service is deployed, you will have to set up the Service manually.  You can use the trigger but it will fail when deploying. Afterwards you can manually configure a Cloud Run service pointing to the image in Artifact Registry and use the following settings.

Then navigate to the Cloud Run and choose the appropriate settings:
* Maximum number of instances = 10
* Allow unauthenticated invocations
* Choose container port 8000
* Set memory to 8GB and number of CPUs to 2
* Request timeout to 3600

You can also create the image locally, tag it and push it to the Artifact Registry using:
```bash
docker build -f dockerfiles/predict_model.dockerfile  -t my_fastapi .
docker tag my_fastapi gcr.io/emotions-410912/my_fastapi
docker push gcr.io/emotions-410912/my_fastapi
```
Then manage deployment manually.

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
├── pyproject.toml       <- Build system requirements 
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
│
├── requirements_dev.txt <- The requirements file for unit tests on data and model coverage
│
└── vertext_ai_config.yaml <- The configuration file used to submit jobs to Vertex AI 
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template) and [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
