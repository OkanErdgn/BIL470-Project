import pandas
import csv
array = []

file = pandas.read_csv("C:/Users/14oka/Desktop/submissions/submission.csv",header=None)
print(file)


#print(array[0][7])
arr = []
brr = []
count = 0


asd = pandas.DataFrame({"image_name": file[0],
                        "target": file[1]})
asd.to_csv(r'C:/Users/14oka/Desktop/submissions/submission-duzelt.csv',index = False)


