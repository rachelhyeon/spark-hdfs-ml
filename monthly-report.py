"""
Script to generate monthly statistics for Lending Club data.
Rows are months, and column is for each summary statistics.

Summary statistics:
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
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession \
    .builder \
    .appName("Lending Club Analysis") \
    .getOrCreate()

# Use spark.read.parquet() to load the data into variable df
df = spark.read.parquet("hdfs://10.0.0.4:9000/lending-club.parquet")

# # Use df.count() to get the number of rows of data
total_loans = df.count() # 2260669

# Total number of loans issued for each month
monthly_loans_issued = df.groupBy("issue_d") \
    .count() \
    .withColumnRenamed('count', "Total loans issued")

# Number of 36 month and 60 month loans issued
monthly_36m_loans_issued = df.filter(F.col("term")=='36 months') \
    .groupBy("issue_d") \
        .count() \
        .withColumnRenamed('count', '36 month loans')

monthly_60m_loans_issued = df.filter(F.col("term")=='60 months') \
    .groupBy("issue_d") \
        .count() \
        .withColumnRenamed('count', '60 month loans')

# Total funded amount of all loans
# Total remaining principal to be paid for all loans

monthly_total = df.select(F.col("issue_d"),
                          F.col("funded_amnt"),
                          F.col("out_prncp")) \
                          .groupBy("issue_d") \
                          .sum() \
                          .withColumnRenamed('sum(funded_amnt)', 'Funded Amount') \
                          .withColumnRenamed('sum(out_prncp)', 'Remaining Principal')

# Total funded amount of 36 month loan
# Total remaining principal to be paid for 36 month loan

monthly_total_36m = df.select(F.col("issue_d"),
                              F.col("funded_amnt"),
                              F.col("out_prncp")) \
                              .filter(F.col("term")=='36 months') \
                                .groupBy("issue_d") \
                                    .sum() \
                                    .withColumnRenamed(
    "sum(funded_amnt)", "Funded Amount for 36 month loan") \
    .withColumnRenamed("sum(out_prncp)", "Remaining Principal for 36 month loan")


# Total funded amount of 60 month loan
# Total remaining principal to be paid for 60 month loan

monthly_total_60m = df.select(F.col("issue_d"),
                              F.col("funded_amnt"),
                              F.col("out_prncp")) \
                              .filter(F.col("term")=='60 months') \
                                .groupBy("issue_d") \
                                    .sum() \
                                    .withColumnRenamed(
    "sum(funded_amnt)", "Funded Amount for 60 month loan") \
    .withColumnRenamed("sum(out_prncp)", "Remaining Principal for 60 month loan")

# Percentage of loans with interest rate greater than 10%
pct_loans_int_rate_greater_than_10 = df.filter(F.col("int_rate")>10) \
.groupBy("issue_d") \
.count() \
.join(monthly_loans_issued, on="issue_d") \
.withColumn('Pct Loans with int_rate > 10', (F.col('count')/F.col("Total loans issued"))*100) \
.drop("count", "Total loans issued")

# Percentage of loans that have been fully paid by now (see loan_status)
pct_loans_fully_paid = df.filter(F.col('loan_status')=='Fully Paid') \
.groupBy("issue_d") \
.count() \
.join(monthly_loans_issued, on="issue_d") \
.withColumn('Pct of Fully Paid Loans',
            (F.col('count')/F.col("Total loans issued")*100)) \
.drop("count", "Total loans issued")

# Percentage of fully paid loans for grade A loans
grade_a_loans = df.filter(F.col('grade')=='A') \
.groupBy("issue_d") \
.count() \
    .withColumnRenamed('count', 'num_grade_a_loans')

grade_a_fully_paid = df.filter((F.col('grade')=='A') & (F.col('loan_status')=='Fully Paid')) \
.groupBy("issue_d") \
.count() \
    .withColumnRenamed('count', 'num_fully_paid_loans_grade_a')

pct_loans_fully_paid_grade_a = grade_a_fully_paid.join(grade_a_loans, on="issue_d") \
.withColumn("Pct of Fully Paid Grade A Loans",
            (F.col("num_fully_paid_loans_grade_a") / F.col("num_grade_a_loans"))*100) \
            .drop("num_grade_a_loans", "num_fully_paid_loans_grade_a")

# Percentage of fully paid loans for grade F loans
grade_f_loans = df.filter(F.col('grade')=='F') \
.groupBy("issue_d") \
.count() \
    .withColumnRenamed('count', 'num_grade_f_loans')

grade_f_fully_paid = df.filter((F.col('grade')=='F') & (F.col('loan_status')=='Fully Paid')) \
.groupBy("issue_d") \
.count() \
    .withColumnRenamed('count', 'num_fully_paid_loans_grade_f')

pct_loans_fully_paid_grade_f = grade_f_fully_paid.join(grade_f_loans, on="issue_d") \
.withColumn("Pct of Fully Paid Grade F Loans",
            (F.col("num_fully_paid_loans_grade_f") / F.col("num_grade_f_loans"))*100) \
            .drop("num_grade_f_loans", "num_fully_paid_loans_grade_f")

# Percentage of loans on a hardship payment plan
hardship_loans = df.filter(F.col("hardship_flag")=="Y") \
.groupBy("issue_d") \
.count() \
.withColumnRenamed('count', "num_hardship_loans") \
.withColumn('Pct of Hardship Loans', F.round((F.col("num_hardship_loans")/total_loans)*100, 3)) \
.drop("num_hardship_loans")

"""
All the tables to join:
- Total number of loans issued - monthly_loans_issued
- Number of 36 month and 60 month loans issued - monthly_36m_loans_issued
- Number of 60 month loans issued - monthly_60m_loans_issued
- Total funded amount/remaining principal of all loans -  monthly_total
- Total funded amount/remaining principal of 36 month loan - monthly_total_36m
- Total funded amount/remaining principal of 60 month loan - monthly_total_60m
- Percentage of loans with interest rate greater than 10% - pct_loans_int_rate_greater_than_10
- Percentage of loans that have been fully paid by now - pct_loans_fully_paid
- Percentage of fully paid loans for grade A loans - pct_loans_fully_paid_grade_a
- Percentage of fully paid loans for grade F loans - pct_loans_fully_paid_grade_f
- Percentage of loans on a hardship payment plan - hardship_loans
"""

# Join all the columns
final_result = monthly_loans_issued \
    .join(monthly_36m_loans_issued, ["issue_d"]) \
    .join(monthly_60m_loans_issued, ["issue_d"], 'outer') \
    .join(monthly_total, ["issue_d"], 'outer') \
    .join(monthly_total_36m, ["issue_d"], 'outer') \
    .join(monthly_total_60m, ["issue_d"], 'outer') \
    .join(pct_loans_int_rate_greater_than_10, ["issue_d"], 'outer') \
    .join(pct_loans_fully_paid, ["issue_d"], 'outer') \
    .join(pct_loans_fully_paid_grade_a, ["issue_d"], 'outer') \
    .join(pct_loans_fully_paid_grade_f, ["issue_d"], 'outer') \
    .join(hardship_loans, ["issue_d"], 'outer') \
        
final_result = final_result.filter(final_result.issue_d.isNotNull())

# Write to csv
final_result.write.csv("hdfs://10.0.0.4:9000/mhyeon/mhyeon-hw02.csv",
                       header=True, mode="overwrite")
