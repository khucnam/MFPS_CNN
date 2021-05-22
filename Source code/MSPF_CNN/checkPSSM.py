import os
import urllib.request

# def takeUniProtInfo(itemID,ext):
#     response = urllib.request.urlopen("https://www.uniprot.org/uniprot/"+itemID+ext)
#     content = response.read()
#     return content
#
#
# f=open("C:/Users/Yang/Downloads/fasta1.txt","r")
# lines=f.readlines()
# f.close()
#
# s="https://www.uniprot.org/uniprot/"
# print(s)
#
# pssm=os.listdir("E:/2.Saved materials/PSSM strorage/MoreThan10000PSSMFilesOfManyKinds")

import shutil

def moveToFolder(path1,filename,path2):
#    print(path1+"\\"+filename+" need moving to", path2 )
    shutil.copyfile ( path1+"/"+filename , path2 +"/"+filename)


fList=os.listdir("D:/YANG/SOFTWARES/blast-Thai/bin/pssm")
for filename in os.listdir("D:/YANG/SOFTWARES/blast-Thai/bin/fasta"):
    # print(filename[:-5]+".fasta")
    if filename[:-6]+".pssm" not in fList:
        print(filename)
        moveToFolder("E:/2.Saved materials/PSSM strorage/MoreThan10000PSSMFilesOfManyKinds", filename[:-6]+".pssm", "D:/YANG/SOFTWARES/blast-Thai/bin/pssm")




#
#
# for line in lines:
#     id=line[len(s):-7]
#     # print(id)
#     if id+".pssm" in pssm:
#         print(id, " is available ")
#         try:
#             content = takeUniProtInfo(id, ".fasta").decode("utf-8")
#             f=open("D:/YANG/SOFTWARES/blast-Thai/bin/fasta/"+id+".fasta","w")
#             f.write(content)
#             f.close()
#         except:
#             print(id, ".fasta can not download")
#     # else:
#     #     st+=id+" "
#     #     # try:
#         #     content = takeUniProtInfo(id, ".fasta").decode("utf-8")
#         #     # f=open("D:/YANG/SOFTWARES/blast-Thai/bin/fasta/"+id+".fasta","w")
#         #     # f.write(content)
#         #     # f.close()
#         # except:
#         #     print(id, ".fasta can not download")




