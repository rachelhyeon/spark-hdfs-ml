# Statistics and ML using Spark and Hadoop Cluster

This repository contains two scripts that can be used on [LendingClub](https://www.lendingclub.com/) data that is stored on Apache Hadoop clusters in a parquet format. 

`monthly_report.py` can generate monthly summary statistics consisting of: 
- Total number of loans issued
- Number of 36 month and 60 month loans issued
- Total funded amount of all loans
- Total funded amount of 36 month loan
- Total funded amount of 60 month loan
- Total remaining principal to be paid for all loans
- Total remaining principal to be paid for 36 month loan
- Total remaining principal to be paid for 60 month loan
- Percentage of loans with interest rate greater than 10%
- Percentage of loans that have been fully paid by now
- Percentage of fully paid loans for grade A loans
- Percentage of fully paid loans for grade F loans
- Percentage of loans on a hardship payment plan

## Machine learning pipeline for predicting loan status

`predict_loan_status.py` produces F1 scores and accuracy for loan status prediction using decision trees, random forest, and multi-layer perceptrons and saves the metrics as a csv to the Hadoop Distributed File System (HDFS).

The features that are used for prediction are:
- `annual_inc`: The self-reported annual income provided by the borrower during registration
- `delinq_2yrs`: The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
- `purpose`: A category provided by the borrower for the loan request
- `mort_acc`: Number of mortgage accounts
- `pub_rec_bankruptcies`: Number of public record bankruptcies
- `fico_range_high`: The upper boundary range the borrower’s FICO at loan origination belongs to
- `fico_range_low`: The lower boundary range the borrower’s FICO at loan origination belongs to
- `max_bal_bc`: Maximum current balance owed on all revolving accounts
- `total_rec_late_fee`: Late fees received to date

The predictions are categorized into three categories:
- Category 1: Loans fully paid off, with no late payments
- Category 2: Loans paid back, but late
- Category 3: Loans never fully paid back
