### MUSIC GENRE CLASSIFICATION WITH MACHINE LEARNING TECHNIQUES

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

### INFORMATION ABOUT THE REPOSITORY 
* config.py file includes some properties like dataset directory, test directory and some properties for signal processing and feature extraction.
* CreateDataset.py file is used for feature extraction and creating dataset.
* ModelTrain.py file is used for creating and training a model.
* GenreRecognition.py file is for predicting the genres of test music files.
* CreateThenTrain.py file runs CreateDataset.py and ModelTrain.py sequentially. 

* Jupyter Notebook files give useful information and tutorials about signal analysis and music genre classification.


