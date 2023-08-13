# Resume Classifier

# Motive
  This task is given by `Interactive Cares` for interview purposes. The task is to create a backend app using `Machine Learning` that could do classification on the resume, save the `Resume` in the classified folder, and generate a `csv` with the `Resume` information.
  
# Procedure

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

  As an ML won't understand any text so I used  

# Output 
# Evalaution
