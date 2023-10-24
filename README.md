## Sign-Language-Recognition-and-Intent-Detection
Sign languages (also known as signed languages)are languages that use thevisual-manual modality to convey meaning, instead of spoken words. Given a picture or videodata, recognize the sign language and detect intents.

## requirements

- pandas
- numpy
- zipfile
- matplotlib
- seaborn
- tensorflow
- visualkeras (pip install git+https://github.com/paulgavrikov/visualkeras --upgrade)
- JointBert (git clone https://github.com/monologg/JointBERT.git)

## Dataset

|       | Train  |  Test  |   Labels      |
| ----- | ------ |  ----- | ------------- |
| MNIST | 27,455 |  7,172 | 24            |
| Snips | 13,084 |  700   | 7             |
| S/A   | 8,227  |  2,056 | 7             |

- The number of labels are based on the _train_ dataset.
- Add `UNK` for labels (For intent and slot labels which are only shown in _dev_ and _test_ dataset)
- Add `PAD` for slot label

## Execution
# Sign Language Detection
SignLanguageDetectiono.ipynb file is in Models folder
1. locally download the data set folder (Archive) and unzip the folder.
2. Run all the cells in .ipynb file except 2nd one which is needed only if .ipynb file is in colab.

#Snips Intent detection 
1. SnipsIntentFromSignLanguageParallel_1.ipynb file
2. Run the whole file in colab. 
3. Run the run, run1() function in the file multiple times as many times as you wish till the desired epochs is reached.

#API
Run commad in API folder "python app.py"

#Video Classification
1. VideoClassification.ipynb file
2. just Run the file in Jupiter notebook or coalb
