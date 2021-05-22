import os
os.chdir("D:/YANG/JOBS/Latest profile/Postdoctoral fellow-Lund university/Assignment")


def extract(inF):
    f=open(inF,"r")
    lines=f.readlines()
    f.close()

    outF=inF[:-4]+".csv"
    f=open(outF,"w")
    f.write("No. of Epoch, Batch size, Learning rate, Regulazizer, Optimizer, Average Accuracy, Std\n")

    for i in range(len(lines)):
        line=lines[i]
        if line.find("epoch")>=0:
            line=line.replace(" =","")
            paras=line[:-1].split()
            print(paras)
            print(i, " epoch PARAS LENGTH ", len(paras))
            f.write(paras[1]+","+paras[4]+","+paras[6]+","+paras[8]+","+paras[10]+",")
        if line.find("Average")>=0:
            paras = line[:-1].split()
            print(paras)
            print(i, " acc PARAS LENGTH ", len(paras))
            f.write(paras[-1]+",")

        if line.find("Std")>=0:
            paras = line[:-1].split()
            print(paras)
            print(i, " std PARAS LENGTH ", len(paras))
            f.write(paras[-1]+"\n")

    f.close()

def extract2(inF):
    f=open(inF,"r")
    lines=f.readlines()
    f.close()

    outF=inF[:-4]+".csv"
    f=open(outF,"w")
    f.write("Kernel, gamma, C, Average Accuracy, Std\n")

    for i in range(len(lines)-1):
        line=lines[i]
        if line.find("Average acc of 5 runs")>=0:
            lineK = lines[i-2].replace(" =", "")
            paras = lineK[:-1].split()
            print("with kernel ", paras, " ", len(paras))
            f.write(paras[2] + "," + paras[4] + "," + paras[6]+",")

            line=line.replace(" =","")
            paras=line[:-1].split()
            print("average acc ", paras, " ", len(paras))
            f.write(paras[-1]+",")

            lineS = lines[i+1].replace(" =", "")
            paras = lineS[:-1].split()
            print("std of 5 runs ", paras, " ", len(paras))
            f.write(paras[-1] + "\n")



    f.close()

# extract("Report1-with country.txt")
# extract("Report1-withOUT country.txt")
# extract("Report2-with country.txt")
# extract("Report2-withOUT country.txt")

extract2("SVM-Report-with countryt.txt")
extract2("SVM-Report-withOUT countryt.txt")

