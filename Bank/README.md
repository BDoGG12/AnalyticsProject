# Overview

The overview of the case study is about taking a data driven approach to predict whether or not the clients will subscribe to a term deposit. In this data, we are working with Portuguese marketing campaigns that list out the following metrics shown in the data dictionary. The campaigns performed in this data were through phone calls with the clients. Logistic regression will be performed in this analysis.

This data was found on UC Irvine's Machine Learning Repository. It is based off of a case study authored by S. Moro, P. Cortez and P. Rita. Please refer to the citation below:

S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014


# Data Dictionary

input variables:

age (numeric)

job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')

marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)

education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')

default: has credit in default? (categorical: 'no','yes','unknown')

housing: has housing loan? (categorical: 'no','yes','unknown')

loan: has personal loan? (categorical: 'no','yes','unknown')

contact: contact communication type (categorical: 'cellular','telephone') 

month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')

day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')

duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)

previous: number of contacts performed before this campaign and for this client (numeric)

poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

emp.var.rate: employment variation rate - quarterly indicator (numeric)

cons.price.idx: consumer price index - monthly indicator (numeric) 

cons.conf.idx: consumer confidence index - monthly indicator (numeric) 

euribor3m: euribor 3 month rate - daily indicator (numeric)

nr.employed: number of employees - quarterly indicator (numeric)

# Dependent Variable
y: has the client subscribed a term deposit? (binary: 'yes','no')


# Goal
1. Can we build a model where clients will subscribe to a term deposit?

2. What are the variables in the model?

3. Which ones are most impactful and how do they influence the prediction of the client’s response to the telemarketing campaign?

4. If I want to have more clients subscribe to the term deposit, what aspects should I target?
