### Automate entire machine learning pipeline and create end to end solution for given project.

**Project** : _To build a classification methodology to determine whether a person defaults the credit card payment for the next month._

###### ** Entire project explanation provided in _**creditCardDefaulters.docx**_ file.

### Tools & Libraries:

Language : `Python3.6`

Tools : `Docker`

Cloud & Database : `GCP`, ` MongoDB atlas`

Libraries : `sklearn`, `pandas`, `numpy`, `flask`, `GCP-storage`, etc..

### Command we use to build docker image:

---------------LOCALLY------------------------------

`docker image build -t <REPOSITORY>` 

`docker images`

`docker ps `

`docker run -p 5000:5000 -d <REPOSITORY>`

`docker stop <containerID>`

`docker system prune`

----------------In GOOGLE CLOUD PLATFORM-------------------

`git clone https://github.com/Tejas2512/CreditCardDefaulters.git`

`cd CreditCardDefaulters`

`export PROJECT_ID=creditcarddefaulters`

`docker build -t gcr.io/${PROJECT_ID}/creditcard:v1 .`

`docker images`

`gcloud auth configure-docker gcr.io`

`docker push gcr.io/${PROJECT_ID}/creditcard:v1`

`gcloud compute zones list`

`gcloud config set compute/zone asia-southeast2-a`

`gcloud container clusters create creditcard  --num-nodes=2`

`kubectl create deployment creditcard  --image=gcr.io/${PROJECT_ID}/creditcard:v1`

`kubectl expose deployment creditcard --type=LoadBalancer --port 80 --target-port 8080`

`kubectl get service`

`docker image rm -f <image_id>`







