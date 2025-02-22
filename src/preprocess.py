import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    sales_df = pd.read_csv(file_path, encoding="unicode_escape")
    sales_df["ORDERDATE"] = pd.to_datetime(sales_df["ORDERDATE"])
    
    df_drop = ['ADDRESSLINE1', 'ADDRESSLINE2', 'POSTALCODE', 'CITY', 'TERRITORY', 'PHONE',
               'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER']
    sales_df = sales_df.drop(df_drop, axis=1)
    
    return sales_df
