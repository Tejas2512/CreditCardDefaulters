### Automate entire machine learning pipeline and create end to end solution for given project.

**Project** : _To build a classification methodology to determine whether a person defaults the credit card payment for the next month._

###### ** Entire project explanation provided in _**creditCardDefaulters.docx**_ file.

### Tools & Libraries:

Language : `Python3.6`

Tools : `Docker`

Cloud & Database : `GCP`, ` MongoDB atlas`

Libraries : `sklearn`, `pandas`, `numpy`, `flask`, `GCP-storage`, etc..

### Command we use to build docker image:

`docker image build -t <REPOSITORY>` 

`docker images`

`docker ps `

`docker run -p 5000:5000 -d <REPOSITORY>`

`docker login dockerfitbit.azurecr.io`

`docker push dockerfitbit.azurecr.io/mlfitbit:latest`

`docker stop <containerID>`

`docker system prune`

