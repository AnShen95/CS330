import math
import statistics
import torch
import random

def read_CSV(file_name, DEVICE, has_title=False):
    year = {}
    X = {}
    Y = {}
    file = open(file_name, 'r')
    if has_title:
        line = file.readline()
    count = 0
    line = file.readline()
    while line:
        #print("line ", str(count))
        temp = line.split(",")
        if not temp[0] in year:
            year[temp[0]] = []
            X[temp[0]] = []
            Y[temp[0]] = []
        year[temp[0]].append(int(temp[2]))
        aLine = []
        for i in range(18):
            aLine.append(float(temp[i + 3]))
        X[temp[0]].append(aLine)
        if temp[1] == "alive":
            Y[temp[0]].append(torch.tensor([0], device=DEVICE, dtype=torch.float))
        else:
            Y[temp[0]].append(torch.tensor([1], device=DEVICE, dtype=torch.float))

        line = file.readline()
        count = count + 1
    return year,X,Y

def normalize(org_file_name, DEVICE, normized_file_name):
    normize_X = {}
    year,X,Y = read_CSV(org_file_name, DEVICE, True)
    # Get normalize factor for each company
    for company, data in X.items():
        total = 0
        for a_number in data[-1]:
            total = total + abs(float(a_number))
        normize_X[company] = total / len(data[-1])
    # Apply normalize factor for each company
    for company, data in X.items():
        nor = normize_X[company]
        for a_row in data:
            for i in range(len(a_row)):
                a_row[i] = str(float(a_row[i]) / nor)
    with open(normized_file_name, 'a') as the_file:
        the_file.write('company_name,status_label,year,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18\n')
        for company, year in year.items():
            for i in range(len(year)):
                the_file.write(company)
                if(Y[company][0] == 1):
                    the_file.write(",failed")
                else:
                    the_file.write(",alive")
                the_file.write("," + str(year[i]))
                for d in X[company][i]:
                    the_file.write("," + str(d))
                the_file.write("\n")
def GenerateDataWithAvgOfPastTwoYear(org_file_name, DEVICE, generate_file_name):
    year,X,Y = read_CSV(org_file_name, DEVICE, True)
    with open(generate_file_name, 'a') as the_file:
        the_file.write('company_name,status_label,year,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18\n')
        for company, data in X.items():
            num_year = len(data)
            for i in range(num_year):
                the_file.write(company)
                if(Y[company][0] == 1):
                    the_file.write(",failed")
                else:
                    the_file.write(",alive")
                the_file.write("," + str(year[company][i]))
                for d in data[i]:
                    the_file.write("," + str(d))
                the_file.write("\n")
            if num_year > 1 and Y[company][0] == 1:
                for i in range(num_year):
                    the_file.write("S_" + company)
                    the_file.write(",failed")
                    the_file.write("," + str(year[company][i]))
                    for d in range(len(data[0])):
                        the_file.write("," + str( (data[-1][d] + data[-2][d]) / 2))
                    the_file.write("\n")

def KNN(all, one, k):
    dis = []
    for i in range(len(all)):
        dis.append((math.dist(one,all[i]),i))
    sd = sorted(dis,key=lambda x: x[0])
    ret = []
    for i in range(1, k+1):
        ret.append(sd[i][1])
    return ret

def GenerateDataSOMTE(org_file_name,dis_file_name, DEVICE, generate_file_name,exclude,num_nei):
    year, X, Y = read_CSV(org_file_name, DEVICE, True)
    year_dis, X_dis, Y_dis = read_CSV(dis_file_name, DEVICE, True)
    all_fail = []
    all_fail_company = {}
    index = 0
    for company, data in X_dis.items():
        if year_dis[company][-1] not in exclude and Y_dis[company][0] == 1:
            all_fail.append(data[-1])
            all_fail_company[index] = company
            index = index + 1

    with open(generate_file_name, 'a') as the_file:
        the_file.write('company_name,status_label,year,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18\n')
        for company, data in X.items():
            num_year = len(data)
            for i in range(num_year):
                the_file.write(company)
                if(Y[company][0] == 1):
                    the_file.write(",failed")
                else:
                    the_file.write(",alive")
                the_file.write("," + str(year[company][i]))
                for d in data[i]:
                    the_file.write("," + str(d))
                the_file.write("\n")
            if Y[company][0] == 1:
                topK = KNN(all_fail, X_dis[company][-1], num_nei)
                for i in topK:
                    #weight = random.random()
                    weight = 0.5
                    for j in range(num_year):
                        the_file.write("S_" + company + "_" + str(i))
                        the_file.write(",failed")
                        the_file.write("," + str(year[company][j]))
                        c = all_fail_company[i]
                        for d in range(len(data[0])):
                            the_file.write("," + str(data[-1][d] * weight + X[c][-1][d] * (1 - weight)))
                        the_file.write("\n")

def GenerateDataSOMTEByYear(org_file_name, DEVICE, generate_file_name,num_copy):
    year, X, Y = read_CSV(org_file_name, DEVICE, True)
    failByYearDict = {}
    for company, data in X.items():
        if (Y[company][0] == 1):
            failInAYear = []
            if year[company][-1] in failByYearDict:
                failInAYear = failByYearDict[year[company][-1]]
            else:
                failByYearDict[year[company][-1]] = failInAYear
            failInAYear.append(data[-1])

    with open(generate_file_name, 'a') as the_file:
        the_file.write('company_name,status_label,year,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18\n')
        for company, data in X.items():
            num_year = len(data)
            for i in range(num_year):
                the_file.write(company)
                if(Y[company][0] == 1):
                    the_file.write(",failed")
                else:
                    the_file.write(",alive")
                the_file.write("," + str(year[company][i]))
                for d in data[i]:
                    the_file.write("," + str(d))
                the_file.write("\n")
            if Y[company][0] == 1 and len(failByYearDict[year[company][-1]]) > num_copy:
                others = random.choices(failByYearDict[year[company][-1]], k = num_copy)
                for i in range(num_copy):
                    weight = random.random()
                    for j in range(num_year):
                        the_file.write("S_" + company + "_" + str(i))
                        the_file.write(",failed")
                        the_file.write("," + str(year[company][j]))
                        for d in range(len(data[0])):
                            the_file.write("," + str(data[-1][d] * weight + others[i][d] * (1- weight)))
                        the_file.write("\n")
def normalize2(org_file_name, DEVICE, normized_file_name,X_size):
    normize_X = {}
    year,X,Y = read_CSV(org_file_name, DEVICE, True)
    X_data = [[] for i in range(X_size)]
    # Get normalize factor for each company
    for company, data in X.items():
        for i in range(X_size):
            X_data[i].append(data[-1][i])
    X_mean = []
    X_std = []
    for i in range(X_size):
        X_mean.append(statistics.mean(X_data[i]))
        X_std.append(statistics.stdev(X_data[i]))
    # Apply normalize factor for each company
    for company, data in X.items():
        for a_row in data:
            for i in range(len(a_row)):
                a_row[i] = str((float(a_row[i]) - X_mean[i])/ X_std[i])
    with open(normized_file_name, 'a') as the_file:
        the_file.write('company_name,status_label,year,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18\n')
        for company, year in year.items():
            for i in range(len(year)):
                the_file.write(company)
                if(Y[company][0] == 1):
                    the_file.write(",failed")
                else:
                    the_file.write(",alive")
                the_file.write("," + str(year[i]))
                for d in X[company][i]:
                    the_file.write("," + str(d))
                the_file.write("\n")

if __name__ == "__main__":
    normalize(r"C:\Temp\CS330\Project\data\american_bankruptcy.csv", torch.device("cpu"),r"C:\Temp\CS330\Project\data\american_bankruptcy_normal.csv")
    GenerateDataWithAvgOfPastTwoYear(r"C:\Temp\CS330\Project\data\american_bankruptcy_normal.csv", torch.device("cpu"), r"C:\Temp\CS330\Project\data\american_bankruptcy_normal_TSSyn.csv")
    #GenerateDataSOMTE(r"C:\Temp\CS330\Project\data\american_bankruptcy_normal.csv", torch.device("cpu"), r"C:\Temp\CS330\Project\data\american_bankruptcy_normal_SMOTE.csv",(2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010,2009,2008,2007),12)
    GenerateDataSOMTEByYear(r"C:\Temp\CS330\Project\data\american_bankruptcy_normal.csv", torch.device("cpu"), r"C:\Temp\CS330\Project\data\american_bankruptcy_normal_SMOTE_By_Year.csv",12)
    normalize2(r"C:\Temp\CS330\Project\data\american_bankruptcy_normal.csv", torch.device("cpu"),r"C:\Temp\CS330\Project\data\american_bankruptcy_normal2.csv", 18)
    GenerateDataSOMTE(r"C:\Temp\CS330\Project\data\american_bankruptcy_normal.csv",r"C:\Temp\CS330\Project\data\american_bankruptcy_normal2.csv", torch.device("cpu"), r"C:\Temp\CS330\Project\data\american_bankruptcy_normal2_SMOTE.csv",(2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010,2009,2008,2007),12)
    pass
