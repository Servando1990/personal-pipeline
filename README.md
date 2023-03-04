# useful-pipeline
# FairMoney asseessment


Requirements:

1. Conda installed.


Steps:

1. Create a conda envirnoment conda env `conda env create -f environment.yml`

To see results:

Navigate to the `Exploration.ipynb` notebook to perform Exploratory Data Analysis (EDA) on the "credit.csv" file located in the "data" folder. This notebook contains reasoning and justification for the modeling approach taken such as:

    a) Dealing with sensitive variables.
    b) Droping specific variables.
    c) Dropping missing values.

The `Results.ipynb` notebook contains the class "CreditReportAnalyzer," which performs feature engineering as expected on part 1 of the assignment.

Reasoning behind features created in Part 1:

    Account rating
    We might care about how many good and bad accounts the client have:

    1. *bad_accounts*
    2. *good_accounts*
    3. *ratio_good*
    4. *ratio_bad*

    Telephone history

    If the client has changed multiple times his/her number, it could be a risk factor

    1. *change_homenumber*
    2. *change_mobile*

    Employment history

    1. *number_of_employments*: How many times th client has changed jobs
    2. *employment_sector*: It would be helpful to detect the employment sector of the client, e.g many public servants have steady incomes even for life

    Credit accounts summary
    1. *ratio_arreas*: What amount is the client behind on payments comparing to the total exposure
    2. *ratio_disnhored*: Similar to ratio arreas but check penalties payments instead of arreas
    3. *ratio_disnhored* Checks dishonoured payments

    Credit agreements summary

    1. *open_status_loans*: Checks how many open loans the client currently have
    2. *remaining_to_original_outstanding_amount_ratio_mean_open_loans*: For every open loan checks how much of the initial amount is outstanding to be paid and returns the mean
    3. *seasoning_mean_open_loans*: For every open loan checks whether it was long time that the client has been repaying the debt and returns the mean in days
    4. *overdue_ratio_mean_open_loans*: For every open loan checks how much is the client overdue comparing to the total amount and returns the mean

    Personal details summary


    1. *age*: Age of the client

The modeling results, including baselines, scores, tuned models, and plots, can be found in the `Results.ipynb`notebook.

## Future work and comments

- Future work can focus on improving the model's performance while addressing overfitting by incorporating other techniques such as early stopping in XGBoost.

- Although ROC AUC was chosen as the evaluation metric in this assignment, it's important to note that it has potential drawbacks. Other metrics such as precision, recall, or gini coefficient can be explored for this problem to provide a better evaluation.

- The JSON as it is provides useful information so i just focused on features that needed  previous transformation.

- Since the constraints and other detail specifications of the credit scoring model to be trained are not defined, it would be useful to have more information and domain context to create more useful features from the given JSON.

- As some sensitive data was spotted in the provided CSV, it is important to address the possibility of unwanted bias towards a sector of the population. Given that Fairmoney is licensed as a bank I wonder how is currently dealing with bias and fairness on its models.

- The code could be further improved by efficiency and efficacy.

- The implementation for part 2 uses two models, so some code refactoring may be necessary if the  models are needed.

- As far as the implementation of Part 1, the given JSON file had all values stored as strings, so a previous numerical conversion had to be performed. It would be useful to have a clear understanding of the data types of the JSONs to avoid unexpected errors or misinterpretation of the data.