# DISS - Text Categorizer

Text Categorizer is a tool that implements a configurable pipeline of methods used to train models that predict the categories of textual data.

For training, it provides the ability to obtain information on the confidence of the trained models, perform cross-validation and train final models using the entire data set (to be implemented).

For prediction, it provides a server that answers queries with the classification predicted by the models.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

There are two supported manners of using this tool.
The first one is to use it natively.
The other one is to use Docker.
The first one is the recommended one because it is more stable and the following problems can occur using Docker:
- Progress bars (that are generated using tqdm) are not shown.
- When using the prediction server, CTRL+C (used to stop the service) which generates SIGINT is translated to SIGTERM, not being detected by Flask. (Flask is simply killed.)
- The order of the output can be wrong because our output is sent using a logger, but the output of other packages is not.
- The debug mode on the prediction server doesn't work because the listener of file changes doesn't detect correctly the location of the files.

### Prerequisites

TODO: prerequisites.

### Installing

Here are presented the instructions on how to install all the dependencies necessary to execute the tool.

TODO: installation steps.

## Executing

Here are presented the instructions on how to execute the tool in different environments.

* \<path-to-DISS\> is the path of folder "DISS".

* \<configuration file\> is the path of the configuration file used. ("config.ini" is provided as an example.)

* \<port\> is the port that will be used by the prediction server to listen for queries.

### Trainer

To execute natively on Windows, open a shell (an Anaconda prompt is recommended) and type the following commands:
```
cd <path-to-DISS>
conda activate text-categorizer
python text_categorizer <configuration file>
```

To execute natively on Linux, open a shell (Bash is recommended) and type the following commands:
```
cd <path-to-DISS>
source activate text-categorizer
python3 text_categorizer <configuration file>
```

To execute using Docker, open a shell and type the following commands:
```
cd <path-to-DISS>
docker-compose up -d text_categorizer-trainer
docker-compose logs -f # Shows the output. (Press CTRL+C to close.)
```
To stop the Docker container, type:
```
docker-compose down -t <seconds> text_categorizer-trainer # Please provide a large value for <seconds> so that there is enough time to save the documents that have not been preprocessed.
```

### Prediction server

To execute natively on Windows, open a shell (an Anaconda prompt is recommended) and type the following commands:
```
cd <path-to-DISS>
conda activate text-categorizer
python text_categorizer/prediction_server.py <configuration file> <port>
```

To execute natively on Linux, open a shell (Bash is recommended) and type the following commands:
```
cd <path-to-DISS>
source activate text-categorizer
python3 text_categorizer/prediction_server.py <configuration file> <port>
```

To execute using Docker, open a shell and type the following commands:
```
cd <path-to-DISS>
docker-compose up -d text_categorizer-prediction_server
docker-compose logs -f # Shows the output. (Press CTRL+C to close.)
```
To stop the Docker container, type:
```
docker-compose down text_categorizer-prediction_server
```

## Built With

* [StanfordNLP](https://stanfordnlp.github.io/stanfordnlp/) - Used for preprocessing.
* [NLTK](http://www.nltk.org/) - Used for preprocessing.
* [scikit-learn](https://scikit-learn.org/stable/) - Used to extract features and train/use the predictive models.

## Authors

* **Luís Barbosa** - [LuisVilarBarbosa](https://github.com/LuisVilarBarbosa)

## Acknowledgments

* The layout of this README was partially inspired on https://gist.github.com/PurpleBooth/109311bb0361f32d87a2.

# Development notes

The code has been tested on Windows 10, MX Linux 18.2 (Debian stable based) and Docker images for Miniconda (Debian based).
