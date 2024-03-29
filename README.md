# Text Categorizer

Text Categorizer is a tool that implements a configurable pipeline of methods used to train models that predict the categories of textual data.

For training, it provides the ability to obtain information on the confidence of the trained models and train final models using the entire data set.
The probabilities for each class for a given example of the test subset are stored in a JSON file, so that other insights can be obtained after creating the models, and several statistics are stored in an Excel file that can store the statistics of multiple executions.
(For the Excel report, take into account that the f1-score micro avg is the same as the accuracy score.)

For prediction, it provides a production server that answers queries with the classification predicted by the models.
The answer includes the probability given to each class for the given query and the weight given to each feature.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

There are two supported manners of using this tool.
The first one is to use it natively.
The other one is to use Docker.
The first one is the recommended one because it is more stable and the following problems can occur using Docker:
- Progress bars (that are generated using tqdm) are not shown immediately. They appear only after processing is complete.
- The order of the output can be wrong because our output is sent using a logger, but the output of other packages is not.

**Important:** The configuration file used for the training mode must not be changed if used for the prediction mode unless you know how the changes will be reflected in the execution of the program.

### Prerequisites

- To execute natively, a machine with Anaconda3 64-bit or Miniconda3 64-bit installed is required.
- To execute natively on Windows, it is also required to have Visual C++ Build tools installed.
- To execute natively on Linux, it is also required to have g++ and libhunspell-dev installed.
- To execute using Docker, only Docker is required and Docker Compose is recommended.

### Installing/Updating

Here are presented the instructions on how to install/update all the dependencies necessary to execute the tool in different environments.

* \<path-to-TextCategorizer\> is the path of the folder "TextCategorizer".

To install natively, open a shell (an Anaconda prompt is recommended on Windows and Bash is recommended on Linux) and type the following commands:
```
cd <path-to-TextCategorizer>
conda env create --file environment.yml
```

To update natively, open a shell (an Anaconda prompt is recommended on Windows and Bash is recommended on Linux) and type the following commands:
```
cd <path-to-TextCategorizer>
conda env create --file environment.yml --force
```

To install/update using Docker, open a shell and type the following commands:
```
cd <path-to-TextCategorizer>
docker-compose build
```

## Executing

Here are presented the instructions on how to execute the tool in different environments.

* \<path-to-TextCategorizer\> is the path of the folder "TextCategorizer".

* \<configuration file\> is the path of the configuration file used. ("config.ini" is provided as example.)

* \<port\> is the port that will be used by the prediction server to listen for queries.

### Trainer

To execute natively on Windows, open a shell (an Anaconda prompt is recommended) and type the following commands:
```
cd <path-to-TextCategorizer>
conda activate text-categorizer
python -m text_categorizer --trainer <configuration file>
```

To execute natively on Linux, open a shell (Bash is recommended) and type the following commands:
```
cd <path-to-TextCategorizer>
source activate text-categorizer
python3 -m text_categorizer --trainer <configuration file>
```

To execute using Docker, open a shell and type the following commands:
```
cd <path-to-TextCategorizer>
docker-compose up -d text_categorizer-trainer
docker-compose logs -f # Shows the output. (Press CTRL+C to close.)
```
To stop the Docker container, type:
```
docker-compose down -t <seconds> text_categorizer-trainer # Please provide a large value for <seconds> so that there is enough time to save the documents that have not been preprocessed.
```

### Prediction Server

To execute natively on Windows, open a shell (an Anaconda prompt is recommended) and type the following commands:
```
cd <path-to-TextCategorizer>
conda activate text-categorizer
python -m text_categorizer --prediction_server <configuration file> <port> # (Press CTRL+C to stop.)
```

To execute natively on Linux, open a shell (Bash is recommended) and type the following commands:
```
cd <path-to-TextCategorizer>
source activate text-categorizer
python3 -m text_categorizer --prediction_server <configuration file> <port> # (Press CTRL+C to stop.)
```

To execute using Docker, open a shell and type the following commands:
```
cd <path-to-TextCategorizer>
docker-compose up -d text_categorizer-prediction_server
docker-compose logs -f # Shows the output. (Press CTRL+C to close.)
```
To stop the Docker container, type:
```
docker-compose down text_categorizer-prediction_server
```

### REST Request

To send a REST request to the prediction server, perform the following operations:
1. Train a classifier (generating a model).
2. Start the prediction server.
3. Open a REST client or prepare corresponding code.
4. Set the method as POST.
5. Set the "Content-Type" header as "application/json".
6. Set the authentication credentials (the default is "admin" both as username and password).
7. Create a JSON body with a dictionary that contains two keys, "text" and "classifier", where the value for "text" is the text to give to the classifier and the value for "classifier" is the name of one of the trained classifiers. Examples of bodies:
    - {"text": "Example text...", "classifier": "RandomForestClassifier"}
    - {"text": "Example text...", "classifier": "LinearSVC"}

Example using cURL to send a request to the localhost:
```
curl --data '{"text": "Example text...", "classifier": "LinearSVC"}' --header 'Content-Type: application/json' --user admin:admin http://localhost:5000/
```

## Testing

Here are presented the instructions on how to execute the tests developed for the tool in different environments.

The pytest-cov plugin is used to perform the tests and generate a coverage report that can be analyzed in any browser.

It is recommended to perform the final tests using Docker because it ensures that the entire pipeline is working correctly and that no invalid cross-device renaming is occurring.

* \<path-to-TextCategorizer\> is the path of the folder "TextCategorizer".

To test natively on Windows, open a shell (an Anaconda prompt is recommended) and type the following commands:
```
cd <path-to-TextCategorizer>
conda activate text-categorizer
pytest --cov-report html --cov=text_categorizer tests/
```

To test natively on Linux, open a shell (Bash is recommended) and type the following commands:
```
cd <path-to-TextCategorizer>
source activate text-categorizer
pytest --cov-report html --cov=text_categorizer tests/
```

To test using Docker, open a shell and type the following commands:
```
cd <path-to-TextCategorizer>
docker-compose up -d text_categorizer-test
docker-compose logs -f # Shows the output. (Press CTRL+C to close.)
```

## Built With

* [MosesTokenizer](https://github.com/luismsgomes/mosestokenizer/) - Used for preprocessing.
* [NLTK](http://www.nltk.org/) - Used for preprocessing.
* [scikit-learn](https://scikit-learn.org/stable/) - Used to extract features and train/use the predictive models.

## Authors

* **Luís Barbosa** - [LuisVilarBarbosa](https://github.com/LuisVilarBarbosa)

## Acknowledgments

* The layout of this README was partially inspired on https://gist.github.com/PurpleBooth/109311bb0361f32d87a2.

# Development Notes

- The code has been tested on Windows 10, MX Linux 19 (Debian stable based) and Docker images for Miniconda (Debian based).

- Pickle is used to dump and load data to and from files. This protocol is the fastest of the tested protocols, but is considered insecure. Please take this information into consideration.

- Consulting the file "log.txt" is essential because it shows several more information than the console.

- The code is relatively generic and can be used as a basis for other experiments, but should be tuned for practical applications.

- The tool in this repository is based on the tool present in https://github.com/LuisVilarBarbosa/DISS.

- The repository present in https://github.com/LuisVilarBarbosa/TextCategorizer-experiments contains side-projects that use a minimal version of the code necessary to categorize text to test different tools that could be added to Text Categorizer.
