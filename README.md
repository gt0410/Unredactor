# Gowtham Teja Kanneganti

## CS 5293, Spring 2019 Project 2

In this project we are tryinbg to create unredactor. Unredactor will take a redacted document and the redacted flag as input, inreturn it will give the most likely candidates to fill in redacted location. In this project we are only considered about unredacting names only.
The data that we are considering is imdb data set with many review files. These files are used to buils corpora for finding tfidf score. Few files are used to train and in these files names are redacted and written into redacted folder. These redacted files are used for testing and different classification models are built to predict the probabilies of each class. Top 5 classes i.e names similar to the test features are written at the end of text in unreddacted foleder.

## Getting Started

The following instructions will help you to run the project.

### Prerequisites

The project is done in python 3.7.2 version. Any version of python-3 will be sufficient to run the code. Also pip environment should be installed. Pyenv and pipenv can be created by using the folowong code in the project. Also a account in [github](https://github.com/) is necessary.
~~~
pyenv install python 3.7.2
pipenv --3.7.2
~~~

### Installing

After setting up the python environment and pip environment the following packages ehich are used in the code need to be installed.

~~~
pipenv install re
pipnev install numpy
pipenv install nltk
pipenv install commonregex
pipenv install gensim
pipenv install sklearn
~~~

The above packages need not be installed in the pip environment you are working but should be available to import. Also, after installing all packages from nltk need to be downloaded. This has to be done in system only once.


## Project Description

### Directory Structure

The structure of the directory of this project is as given below.

cs5293p19-project2/ \
├── COLLABORATORS \
├── LICENSE \
├── Pipfile \
├── Pipfile.lock \
├── README.md \
├── project2 \
│   ├── __init__.py \
│   └── main.py \
├── docs \
├── setup.cfg \
├── setup.py \
└── tests \
    ├── test_test1.py \
    └── test_test2.py \
    └── ...

The structure is received initially from the repository created in the git. This repository can be brought into Ubuntu by cloning that repository. This can be done by using the following code

~~~
git clone "git repository link"
~~~

After that the Pipfile and Pipfile.lock will be created when piipenv is created. All other files are created in command line.
If any changes are made in the repository then they need to be pushed into git. The status of the git can be checked using the following code.
~~~
git status
~~~

When the above command is run, it shows all the files that are modified. These files need to be added, commited and then pushed into git. The following code is followed:
~~~
git add file-name
git commit -m "Message to be displayed in git"
git push origin master
~~~
### Functions description

#### main.py

This python file contains all the functions necessary for this program.

#### Get Entity

get_entity function takes in text of each document, performs entity search to find the names of person in the document and returns the names in that text as a list. Person names can be found by using sample code given below.

~~~
for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
~~~

#### Do extraction

doextraction() gets all the files from the given glob and read each document. The text in each document is sent to get_training_features function to get all the features in that documents. Next, the text is sent to get_entity finction to get the list of names in that document. The names list returned from the ge_entity function are passed into write redacted function.

~~~
for thefile in glob.glob(glob_text):
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
~~~

This function returns the training data.

#### train_tfidf

In the train_tfidf() function we are training all the files in the train data , which I am using as a corpora to generate tfidf scores. In this function I will take the files from theh given filepath and apply tfidf_vectorizer.fit_transform().

~~~
tfidf_vectorizer = TfidfVectorizer()
x = tfidf_vectorizer.fit_transform(documents)
idf = tfidf_vectorizer.idf_
tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names(), idf))
~~~

This function returns the tfidf dictionary with the key as a word and the value for that key as the tf-df score.

#### Write Redacted Files

The write_redacted() function takes in the text for each document, the names list returned from the get_entity function and also the name of teh file. In this function the names are sorted in decreasing order of string length so that parts of other names are not replaced by small names. After these names are replaced with blocks same as the size of the names. Next these redacteed tests are written into redacted folder.

#### Train data features

get_training_features() function takes in the text from each document and the name of the file. In this function I am adding the traing features for document based on document length, rating given by the user from the file name and also the features from name like length of name, number of spaces in name, length of characters in document and the left and right words to the given name. For each name all the features are in the form of a dictionary and this will be in form of list of dictionaries.

#### Test Data Features

The get_testing_features() function takes in the gtext in each document and also the path of the text document. In this I am identifying the redacted names by using regular expressions. Using grouping in regular expressions I am able to get the block which is redactes and also the left and right names to that block. Then using the block detected I am finding the length of the word, number of spaces in the name. Here also same as the above training features we will form a dictionary and list of dictionaries for all the names.

### Running the tests

The test files test the different features of the code. This will allow us to test if the code is working as expected. There are several testing frameworks for python, for this project use the
py.test framework. For questions use the message board and see the pytest
documentation for more examples http://doc.pytest.org/en/latest/assert.html .
This tutorial give the best discussion of how to write tests
https://semaphoreci.com/community/tutorials/testing-python-applications-withpytest.

Install the pytest in your current pipfile. You can install it using the command
pipenv install pytest. To run test, you can use the command pipenv run
Alternatively, you can use the command pipenv run python setup.py test.

Test cases are written for two functions only as in my case only these two functions require text only and other functions also require the file name, etc . For the purpose of testing a string is already given and the tests are written based on this string only. The test cases are written for each function.

#### Testing names

In this test case we are testing if the function get_entity() is taking the data and extracting the names from it. After calling the function we are testing whether the number of names returned are greater than zero or not.

#### Testing redacted name list

In this test case we are testing if the function get_entity() is taking the data and extracting the redacted names from it. This redacted_names list differs from the names list as in this list if a name is a combination of 2 words first name and last name then I am getting these as two seperate names in redacted_names. This list is used to replace the names. I am doing in this manner to get the spaces in between words while replacing the string with blocks.

### Features

The main part in this program according to me is taking the features. Taking right features give give good results. The following sets of features are used: -

#### Word Features

The word features that are taken into consideration are length of the name and the number of spaces in between full name.

#### Context Features

The context features that are used are the left and right words to the names. Initially, I trained using the words as string itself and observed that the models are not performing well and predicting very few classes to all test data. To solve this problem I trained the whole data set taking it as corpora and found the tfidf dictionary with key as the word and value as the tf-df score in that dictionary. So, instead of left and right words, tfidf scores for these are taken as features. In some cases where the immediate words are identified as symbols (like '(' or "'s") tfidf score for these is given as zero.
After traing the models again using tfidf scores the models did perform well.
#### Document Features

The document features that are considered are the number of names in that document, total characters in the document (This is taken into consideration as few people may write long reviews), the rating given by the user. The rating is taken from the file name. The files are named as id_rating.txt.

### Inspirations

https://oudalab.github.io/textanalytics/projects/project2

https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/ , These videos helped me in understanding the usage of nltk package.

https://stackoverflow.com/questions/20290870/improving-the-extraction-of-human-names-with-nltk , referred to this for extracting human names.

https://likegeeks.com/nlp-tutorial-using-python-nltk/ , helped in understanding the basic nltk concepts.

https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python , helped me in using glob function to take imput files.

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html, I have refered to this link to understand what Dictvectorizer is doing.

https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/, I have refered to this link to get absic idea about the classification on text data.

https://medium.com/data-from-the-trenches/text-classification-the-first-step-toward-nlp-mastery-f5f95d525d73, helped me in understanding what features can be considered for the text classification problemss in general.

### People Contacted

Dr. Christan Grant, cgrant@ou.edu, Professor, Discussed about the evaluation part and also got the idea about Dictvectorizer when discussed in class.

Chanukya Lakamsani, chanukyalakamsani@ou.edu, Teaching Assistant, Talked to him about writing k similar names in the output.

Sai Teja Kanneganti, kannegantisaiteja@ou.edu, Co- student, Discussed about Keeping features in the form of a dictionry and using Dictvectorizer and also about writing the top k similar names for the redacted ones in the unredacted files.
