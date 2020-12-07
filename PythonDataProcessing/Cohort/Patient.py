import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
from PythonDataProcessing.Processing.Utils import convert_to_datetime
class Patient:
    def __init__(self, id, los,gender, age,  m30, admitDate=None, deathDate=None):

        print(" IN PTIENT CONSTRUCTOR, PROCESSING: ")
        print(" \t id: ", id, " los: ", los, " gender: ", gender, "age: ", age, "mort: ", m30)
        self.Patient_id = id
        self.Age = age
        self.Gender = gender

        if isinstance(deathDate, str)  and len(deathDate) <= 1:
            self.deathRange = -1
        elif (not (pd.isnull(deathDate)) and not (pd.isnull(admitDate))):
            AdmitDate = convert_to_datetime(admitDate)
            DeathDate = convert_to_datetime(deathDate)
            self.deathRange = DeathDate- AdmitDate
        else:
            self.deathRange = -1


        self.AdmitDate = admitDate
        self.DeathDate = deathDate

        self.los = los
        self.M30 = m30
        self.observations = []

    def addObservations( self, observations ):
        for o in observations:
            self.observations.append(o)


    def printNumObservations( self ):
        print("\t Patient: ", self.Patient_id, "has : ", len(self.observations), "observations")
    def printString( self ):
        print(" Patient: ", self.Patient_id, self.Age, self.Gender)

    def printObservationVolume( self ):
        print(" Patient: ", self.Patient_id," has: ", len(self.observations), "observations")

    def getNumberOfObservations( self ):
        return len(self.observations)

    def as_dict(self):
        patient_row = {'PatientID' : self.Patient_id,
                       'Age' : self.Age,
                       'Gender' : self.Gender,
                       'los': self.los,
                       'DeathPeriod' : self.deathRange,
                       'Mortality30Days' : self.M30
                       }
        return patient_row