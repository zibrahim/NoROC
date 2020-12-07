from PythonDataProcessing.Processing.Serialisation import jsonRead, makeTimeSeriesOneDay
import json
def main():
    configs = json.load(open('Utils/Configuration.json', 'r'))
    data_path = configs['paths']['data_path']

    cohort = jsonRead(data_path+"Cohort.json")
    makeTimeSeriesOneDay(cohort,"Mortality")

if __name__ == "__main__":
    main()