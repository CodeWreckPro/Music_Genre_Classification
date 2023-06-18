### MUSIC GENRE CLASSIFICATION WITH MACHINE LEARNING TECHNIQUES

The Music Genre Classification project is a machine learning-based application that aims to classify music audio files into different genres automatically. By leveraging audio signal processing techniques and machine learning algorithms, this project provides a solution for genre identification and categorization of music tracks.

## Features

- Music genre classification: The application analyzes audio features of music tracks and predicts the genre to which they belong.
- Preprocessing and feature extraction: The project includes preprocessing steps to convert audio files into suitable formats for analysis and feature extraction. Various audio features (e.g., spectral features, rhythm features, timbral features) are extracted to represent the characteristics of each music track.
- Machine learning model integration: The project employs machine learning algorithms (e.g., decision trees, random forests, neural networks) to build classification models and make genre predictions.
- Evaluation and performance metrics: The system provides evaluation metrics to assess the performance of the genre classification models, such as accuracy, precision, recall, and F1-score.
- Customization and fine-tuning: Users can experiment with different feature sets, machine learning algorithms, and model configurations to achieve optimal performance for their specific music genre classification tasks.

## Technologies Used
- Python: The core programming language used for development.
- Audio processing libraries: Used for audio file handling, preprocessing, and feature extraction (e.g., Librosa, Essentia).
- Machine learning libraries: Utilized for building and training genre classification models (e.g., scikit-learn, TensorFlow, Keras).
- Data visualization libraries: Employed for generating visualizations and performance metrics (e.g., Matplotlib, Seaborn).

#### To use this project on your local system or project you must have installed:
* Python 3.6
* Python packages:
	* IPython
	* Numpy
	* Scipy
	* Pandas
	* Scikit-learn
	* Librosa
	* Matplotlib
	* Pydub
* Jupyter Notebook (with IPython kernel)
	
**To install Python:**

_First, check if you already have it installed or not. To do that. 
Open Commmand Line(Type cmd in windows search bar or Terminal in MAC or Linux) 
Type_
~~~~
python3 --version
~~~~
_If you don't have python 3 in your computer you can use the code below_:
~~~~
sudo apt-get update
sudo apt-get install python3
~~~~

**To install packages via pip install:**
~~~~
sudo pip3 install ipython scipy numpy pandas scikit-learn librosa matplotlib jupyter pydub
~~~~
_If you haven't installed pip, you can use the codes below in your terminal_:
~~~~
sudo apt-get update
sudo apt install python3-pip
~~~~
_You should check and update your pip_:
~~~~
pip3 install --upgrade pip
~~~~

## Installation

To use this project, perform the following steps:

1. Clone this repository: `git clone https://github.com/CodeWreckPro/Music_Genre_Classification.git`
2. Navigate to the project directory: `cd Music_Genre_Classification`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage

Follow the instructions below to run the project:

1. Prepare your music dataset in a compatible format (e.g., audio files in a specific directory structure).
2. Preprocess the audio data by converting it into suitable formats for feature extraction and analysis.
3. Extract audio features from the preprocessed music files using the provided feature extraction methods.
4. Split the dataset into training and testing sets for model training and evaluation.
5. Build a genre classification model using machine learning algorithms and the extracted audio features.
6. Train the model on the training set and evaluate its performance on the testing set.
7. Make genre predictions for new music tracks using the trained model.
8. Visualize and analyze the classification results, including performance metrics and genre distribution.

Detailed instructions and code examples can be found in the project's documentation [here](docs/README.md).

## Dataset

To train and evaluate the music genre classification model, a labeled music dataset is required. The dataset should include audio files along with corresponding genre labels. Ensure that the dataset is properly formatted and compatible with the project's data preprocessing and feature extraction functions.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`.
3. Make the necessary changes and commit them: `git commit -m 'Add some feature'`.
4. Push your changes to the branch: `git push origin feature-name`.
5. Open a pull request in the original repository.


### INFORMATION ABOUT THE REPOSITORY 
* config.py file includes some properties like dataset directory, test directory and some properties for signal processing and feature extraction.
* CreateDataset.py file is used for feature extraction and creating dataset.
* ModelTrain.py file is used for creating and training a model.
* GenreRecognition.py file is for predicting the genres of test music files.
* CreateThenTrain.py file runs CreateDataset.py and ModelTrain.py sequentially. 

* Jupyter Notebook files give useful information and tutorials about signal analysis and music genre classification.

## License

This project is licensed under the [MIT License](LICENSE).

Contact
If you have any questions, suggestions, or feedback, please feel free to contact the project maintainer:

Name: RAkshith S
Email: rakshiths2001@gmail.com

Thank you for using the MUSIC GENRE CLASSIFICATION WITH MACHINE LEARNING TECHNIQUES
