# Introduction to Google AI Platform
This file contains text you can copy and paste for the examples in Cloud Academy's _Introduction to Google AI Platform_ course.  

### Introduction
[Google Cloud Platform Free Trial](https://cloud.google.com/free)  

### TensorFlow
[TensorFlow website](https://www.tensorflow.org)  
[TensorFlow installation](https://www.tensorflow.org/install/pip)  

```
python3 -V      # Check which version of Python 3 is installed
pip3 install --user --upgrade pip
pip3 install --user --upgrade virtualenv
virtualenv mlenv
source mlenv/bin/activate
pip3 install tensorflow
```

```
git clone https://github.com/cloudacademy/aiplatform-intro.git
cd aiplatform-intro/iris/trainer
python3 iris.py
```

### Training a Model with AI Platform
[Google Cloud SDK installation](https://cloud.google.com/sdk)  

```
cd ..
gcloud ai-platform local train --module-name trainer.iris --package-path trainer
```

```
PROJECT=$(gcloud config list project --format "value(core.project)")
BUCKET=gs://${PROJECT}-aiplatform  
REGION=us-central1
gsutil mb -l $REGION $BUCKET
```
```
JOB=$iris1
gcloud ai-platform jobs submit training $JOB \
    --module-name trainer.iris \
    --package-path trainer \
    --staging-bucket $BUCKET \
    --region $REGION \
    --runtime-version 2.2
```

