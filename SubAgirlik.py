"""




  AGE AND SEX FACTOR



"""

import pandas
import csv


agirlik = []

file = pandas.read_csv("C:/Users/14oka/Desktop/train.csv")
print(file)
er = 0
ka = 0
her = 0
hka = 0
for i in range(len(file)):
    if file.loc[i,'target'] == 1:
        sex = file.loc[i,'sex']
        tmp = 0
        if sex == "male":
            her +=1
        elif sex == "female":
            hka +=1
    else:
        if file.loc[i,'sex'] == 'male':
            er +=1
        elif file.loc[i,'sex'] == 'female':
            ka +=1

agirlik1 = ((her/(er+her)) * (her/(her+hka)))+1
agirlik2 = ((hka/(ka+hka)) * (hka/(her+hka)))+1
print(agirlik1)
print(agirlik2)


for i in range(len(file)):
    if file.loc[i,'target'] == 1:
        yas = file.loc[i,'age_approx']
        tmp = 0
        for k in range(len(agirlik)):
            if agirlik[k][0] == yas:
                agirlik[k][1] += 1
                tmp = 1
        if tmp == 0:
            data = []
            data.append(yas)
            data.append(1)
            agirlik.append(data)
sum = 0
for i in range(len(agirlik)):
    sum += agirlik[i][1]
    print(agirlik[i][0],"   ", agirlik[i][1])

print(sum)

for i in range(len(agirlik)):
    agirlik[i][1] = agirlik[i][1]/sum +1
    print(agirlik[i][0],"   ", agirlik[i][1])

file = pandas.read_csv("C:/Users/14oka/Desktop/submissions/submission-8356.csv")
file2 = pandas.read_csv("C:/Users/14oka/Desktop/test.csv")




img = []
target = []
for i in range(len(file)):
    yas = file2.loc[i,'age_approx']
    tmp = 0
    for k in range(len(agirlik)):
        if agirlik[k][0] == yas:
            img.append(file.loc[i,'image_name'])
            target.append(file.loc[i,'target'] * agirlik[k][1])
            tmp = 1
    if tmp == 0:
        img.append(file.loc[i, 'image_name'])
        target.append(file.loc[i, 'target'])

for i in range(len(file)):
    if file2.loc[i,'sex'] == 'male':
        target[i] = target[i]*agirlik1
    elif file2.loc[i,'sex'] == 'female':
        target[i] = target[i]*agirlik2

df1 = pandas.DataFrame({"image_name": img,
                    "target":target})
df1.to_csv(r'C:/Users/14oka/Desktop/submissions/submission-with-yas.csv',index = False)
