import os
import sys
from os import rename
import requests
import shutil


def LFW_preprocessor(src_dir, male_folder, female_folder):
    fileList = []
    fileSize = 0
    folderCount = 0
    count = 0
    tmp = ""

    for root, subFolders, files in os.walk(src_dir):
        folderCount += len(subFolders)
        for file in files:
            f = os.path.join(root,file)
            fileSize = fileSize + os.path.getsize(f)
            fileSplit = file.split("_")
            fileList.append(f)
            count += 1

            if count == 1:
                result = requests.get("http://api.genderize.io?name=%s" % fileSplit[0])
                result = result.json()
                tmp = fileSplit[0]
            elif tmp != fileSplit[0]:
                result = requests.get("http://api.genderize.io?name=%s" % fileSplit[0])
                result = result.json()
                tmp = fileSplit[0]
            else:
                tmp = fileSplit[0]

            try:
                if float(result['probability']) > 0.9:
                    if result['gender'] == 'male':
                        shutil.copyfile(f,"%s/%s" % (male_folder,file))
                    elif result['gender'] == 'female':
                        shutil.copyfile(f,"%s/%s" % (female_folder,file))
            except Exception as e:
                print(result)

            print(count)