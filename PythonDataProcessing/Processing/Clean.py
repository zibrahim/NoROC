import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math


def add_itureadmission (cohort, itureadmission_df):
    result = pd.merge(cohort, itureadmission_df, on='hadm_id', how='inner')
    result.reset_index()
    return result

def clean_cohort(cohort, vitals):
    cohort['weight'] = None
    weights  = vitals.loc[vitals.vitalid=='Weightkg',:]
    for ind in cohort['subject_id']:
        ind_weight = weights.loc[(weights.hadm_id ==ind)]
        ind_weight = ind_weight.valuenum
        if len(ind_weight) > 0:
            ind_weight = ind_weight.iloc[0]
        else:
            ind_weight = np.nan
        cohort.loc[cohort['hadm_id']==ind, 'weight'] = ind_weight

    cohort.drop(['dob', 'hadm_id'], inplace=True, axis=1)
    cohort['admittime'] = [x.replace(':00 ', '') for x in cohort['admittime']]
    cohort['gender'] = np.where(cohort['gender'] == "F", 1, 0)

    cohort['admittime'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in cohort['admittime']]
    cohort['deathtime'] = ['' if str(x) =="nan" else datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in cohort['deathtime'] ]

    death_period = []
    death_period_int = []

    mortality_30days = []


    for index, row in cohort.iterrows():
        if pd.isnull(row['deathtime']) or row['deathtime']=='':
            death_period.append(-1)
            mortality_30days.append(0)
        else:
            print("RTWORIWEUREWORUEORRU ", row['deathtime'] , row['admittime'] )
            death_range = row['deathtime'] - row['admittime']
            death_range_int = int((str(death_range)).split('da')[0])
            death_period_int.append(death_range_int)
            death_period.append(death_range)

            if (death_range <= timedelta(days=30)):
                mortality_30days.append(1)
            else:
                mortality_30days.append(0)
    cohort['30DM'] = mortality_30days
    return cohort

def removeDateSuffix(df):
    dates = []
    for s in df:
        parts = s.split()
        parts[1] = parts[1].strip("stndrh") # remove 'st', 'nd', 'rd', ...
        dates.append(" ".join(parts))

    return dates

def updateDate(df):
    date_output_format = "%Y-%m-%d"

    dates = []
    for t in df:
        if t == '' or pd.isnull(t) :
            d = np.nan
        elif "-" in t :
            fmt = "%y-%m-%d"
            d = pd.to_datetime(t, format=fmt, exact=False, utc=True)
        elif "/" in t :
            fmt = "%d/%m/%y"
            d = pd.to_datetime(t, format=fmt, exact=False)
        else :
            fmt = None
            d = pd.to_datetime(t, format=fmt, exact=False)
        if pd.isnull(d):
            dates.append(np.nan)
        else:
            dates.append(d.strftime(date_output_format))
    return dates

def updateDateTime(df):
    date_output_format = "%Y-%m-%d %H:%M:%S"
    dates = []
    for t in df:
        if "-" in t :
            fmt = "%y-%m-%d %H:%M"
            d = pd.to_datetime(t, format=fmt, exact=False, utc=True)
        elif "/" in t :
            fmt = "%d/%m/%y %H:%M"
            d = pd.to_datetime(t, format=fmt, exact=False)
        else :
            fmt = None
            d = pd.to_datetime(t, format=fmt, exact=False)
        dates.append(d.strftime(date_output_format))
    return dates
