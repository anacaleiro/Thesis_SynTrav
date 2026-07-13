# Libraries
import os
import re 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
import inspect

## General statistics overview

def overview(train):
    variables = []
    dtypes = []
    count = []
    unique = []
    missing = []
    pc_missing = []

    for item in train.columns:
        variables.append(item)
        dtypes.append(train[item].dtype)
        count.append(len(train[item]))
        unique.append(len(train[item].unique()))
        missing.append(train[item].isna().sum())
        pc_missing.append(round((train[item].isna().sum() / len(train[item])) * 100, 2))  # Multiply by 100 after dividing

    output = pd.DataFrame({
        'variable': variables,
        'dtype': dtypes,
        'count': count,
        'unique': unique,
        'missing': missing,
        'pc_missing': pc_missing
    })

    return output


## Numerical Variables Visulization Function

def plot_numeric_variable(data, variable, variable2=None, hue=None, sqrt_scale=False):
    sns.set_style("white")
    
    # Setting up the figure with 2 subplots, with the adjusted size of (18, 4)
    fig, ax = plt.subplots(1, 2, figsize=(18, 4))  # Ensure ax is a list of Axes
    
     # 1st graph: Histogram
    sns.histplot(data=data, x=variable, ax=ax[0], kde=True, color='navy')
    ax[0].set_title(f"{variable} Distribution")
    ax[0].set_ylabel("Count")
    
    # Apply log scale on y-axis if specified
    if sqrt_scale:
        ax[0].set_yscale('function', functions=(np.sqrt, lambda x: x**2))
    
    
    # 2nd graph. Boxplot for the specified variable (variable2 or variable1 if not provided)
    plot_variable = variable2 if variable2 is not None else variable
    sns.boxplot(data=data, x=plot_variable, ax=ax[1], color='navy')
    ax[1].set_title(f"{plot_variable} Boxplot")
    ax[1].set_ylabel("")
    
    # Show the plot
    plt.tight_layout()
    plt.show()



## Generate the function for categorical variables

# We get 2 parameters:
    #  data: DataFrame containing the data
    #  categorical_var: str, the name of the categorical variable to analyze
def plot_categorical_analysis(data, categorical_var, order_descending=False):
    sns.set_style("white")

    # Calculate the value counts and proportions
    value_counts = data[categorical_var].value_counts()
    proportions = data[categorical_var].value_counts(normalize=True)  

    # Combine counts and proportions into a DataFrame
    summary_df = pd.DataFrame({
        'Count': value_counts,
        'Proportion': proportions
    })
    
    # Get the order for the categories
    category_order = data[categorical_var].value_counts(ascending=not order_descending).index
    
    # Set up the figure for one plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Histogram for categorical data (Countplot)
    sns.countplot(data=data, x=categorical_var, ax=ax, order=category_order, color='navy') 
    ax.set_title(f"{categorical_var} Distribution")
    ax.set_xlabel(categorical_var)
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=65)  # Rotate x labels for better readability

    # Show the plot
    plt.show()

    # Print the value counts and proportions as a Markdown table
    print(summary_df.to_markdown())


## Colaspsing houselhold minor info
def minor_category(row):
    u6  = pd.to_numeric(row['Number of household members under 6 years of age'], errors='coerce') or 0
    u12 = pd.to_numeric(row['Number of household members aged 6 to 11'], errors='coerce') or 0
    u18 = pd.to_numeric(row['Number of household members aged 12 to 17'], errors='coerce') or 0
    if u6 > 0:
        return 'young_children'
    elif u12 > 0:
        return 'school_age_children'
    elif u18 > 0:
        return 'teenagers'
    return 'no_minors'

# Activity status

def activity_status(row):
    paid = str(row['Paid work OP']).lower()
    unpaid = str(row['Unpaid activity OP']).lower()
    student = str(row['OP has a Student OV chip card'])
    is_working = 'hours per week' in paid  # excludes 'no paid work' and 'not asked'
    is_retired = 'retired' in unpaid
    is_student = 'schoolchild' in unpaid or student == 'Yes'
    if is_retired and is_working:
        return 'working_retired'
    elif is_working:
        return 'employed'
    elif is_retired:
        return 'retired'
    elif is_student:
        return 'student'
    elif 'housewife' in unpaid:
        return 'homemaker'
    elif 'incapacitated' in unpaid:
        return 'incapacitated'
    elif 'unemployed' in unpaid:
        return 'unemployed'
    return 'inactive'

## Summary base group
"""
    Duration stays numeric since Activity duration (in minutes) is already a number.
    The dep_time and distance become share distributions like mode and purpose.
    
"""
def group_summary(df, candidate_dims):
    s = {}
    s['n_trips']          = len(df)
    s['n_persons']        = df['Person_index'].nunique()
    s['trips_per_person'] = round(len(df) / df['Person_index'].nunique(), 2)
    s['mode_share'] = (
        df['Main mode of transport travel'].value_counts(normalize=True).mul(100).round(1).to_dict()
    )
    s['purpose_share'] = (
        df['Motive'].value_counts(normalize=True).mul(100).round(1).to_dict()
    )
    s['dep_time_class_share'] = (
        df['Departure time class'].value_counts(normalize=True).mul(100).round(1).to_dict()
    )
    s['distance_class_share'] = (
        df['Travel distance class in the Netherlands'].value_counts(normalize=True).mul(100).round(1).to_dict()
    )
    s['candidate_distributions'] = {}
    for dim in candidate_dims:
        s['candidate_distributions'][dim['name']] = (
            df[dim['col']].value_counts(normalize=True).mul(100).round(1).to_dict()
        )
    return s