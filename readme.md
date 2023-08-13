# Resume Classifier

# Motive
  This task is given by `Interactive Cares` for interview purposes. The task is to create a backend app using `Machine Learning` that could do classification on the resume, save the `Resume` in the classified folder, and generate a `csv` with the `Resume` information.
  
# Procedure for Training the ML Model

The dataset was collected from [here](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).
In this dataset, there has `resumes` as `pdf` and a `csv` file that is made on the resumes. I used the CSV file for analysis and model training.

The CSV file contains 2484 rows which means information of 2484 categorized resumes.

## Approaches
  1. My first approach was to take just `Skill` information from  all resumes. That might reduce the size of the Dataset and less computational time.
     - When I used this approach, I found many of the `Skill` are not matched with `Qualification` and `Category`. So I reject this approach.
  2. Next approach was taking not only `Skill` information and `Qualification` data from `Resume_HTML` column.
     - This approach also made conflict with `Experience` and `Category` also many of the resume doesn't have `Qualification` data. So I had to reject the approach also
  3. Grab all of the text in a resume and preprocess the text to train an ML model.
     - Finally, I took this approach.

## Preprocess
  As I took all of the text inside of a resume. There must be anomalies presented. So I used basic NLP methods for `Text Preprocessing`
  
   - Removed `Punctuation` from text
   - Removed `Stop words` 
   - Used `Lemmatization` to remove suffixes and prefixes
   - used `Stemming` to convert a word to a root word
All of these procedures are to reduce dataset column size.

## Text to Number

  As an ML won't understand any text, I used a simple `CountVectorizer` to count how many times a word exists.

After that, the dataset size was `2484, 26847`. 2484 rows and 26847 no columns which is huge, and many columns appear only once also useless.

## Feature Selection

After performing `CountVectorize`, a numerous number of column had created. So I took the words which occurred more than 5% in the dataset. 
So after that now the column size is `1116`. These `1116` words are the features.

      Question: Why did I take more than 5% occurred words? Why not more or less?
      Ans: Simple, these words more than `5%` gave me the best result which is `60%` accurate. While using `1%` gave me `41%` and `10%` gave me `48%` using a particular ML model.

## Split Dataset
   As the dataset contains 2484 resume information which is a tiny dataset. So, I split the dataset and took `0.05%` data for testing the model, and the rest of it (2360) is for training a model.

## Data Impute
  I found the dataset is not well balanced. Which could be a reason for the biased model. So I used the `RandomOverSampler` method for balancing the dataset. So each  `Category` has 118 resumes and there are 24 `Categories`. After imputer, the dataset size is increased to `2832`.

## Model Selection
  As the dataset is small though I imputed some data into the dataset. So I chose `DecissionTree`, `RandomForest`, and `SupportVectorMachine` models for training. If the dataset had enough data or more than 10k I would use other `Gradient-based` models or `Deep Learning` models.

  | Model Name | Accuracy |
  |--------|-------|
  |SVC| 46%|
  | DecisionTreeClassification|44%|
  | RandomForestClassification |57%|

  Finally, I chose the `RandomForestClassification` for further work.

## Parameter Tuning

  I used `GridSearchCV` with `cv= 5` to find the best combination of parameters that would produce the best accuracy.

  Finally, `{'max_depth': 18, 'n_estimators': 1000, 'n_jobs': -1}` this combination for the `RandomForestClassifier` model gave `74%` accuracy which was a good step.

  
## Evalaution

After the parameter tuning step, the final evaluation report is - 

![Screenshot 2023-08-13 171520](https://github.com/AklimaRimi/interactive_cares-p1/assets/59701116/ea8767b4-679d-4863-9b75-adbde9d70bd5)

# Python API
Using the model, I made an API, as it is required. That will take a `path` as input, the `path` is where the resume is to be classified.
And a `csv` file will be produced. In the CSV tfile thew will be two columns 1st is resume name and another is the category of the resume.


# How to use the API

1. Download the file from -

![Screenshot 2023-08-13 155643](https://github.com/AklimaRimi/interactive_cares-p1/assets/59701116/9bf2622c-6b37-4dc3-a5f9-a77138a2125b)

2. Unzip the Zip file. Then cmd on folder path like
   
![Screenshot 2023-08-13 175339](https://github.com/AklimaRimi/interactive_cares-p1/assets/59701116/183b9a8e-0e61-4f00-aa28-ea7e039cd4c0)

3. Write the code on the terminal, and make sure `pyhton` is already installed in your PC.

        pip install -r requirements.txt
4. After installing all of the necessary packages. Then write the command on the terminal

       python script.py path/x/y

### Done.. You will get folders where resumes are saved and A CSV file.

# Thank You
















































