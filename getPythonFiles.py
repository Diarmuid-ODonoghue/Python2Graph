import os
import shutil
import random


""" finds all the python files in a directory and copies them to a new directory"""


def getPyFiles(location, newLocation):
    for root, dirs, files in os.walk("python-corpus", topdown=False):
        for name in files:
            #print(os.path.join(root, name))
            if name.endswith('.py'):
                try :
                    print(root + name)
                    shutil.copy2(os.path.join(root, name), "F:/pyCorpus/")
                except Exception as er:
                    print(er)
                    file = open("logs.txt", "a")
                    file.write(str(er))
                    file.write(str(er))
                    file.close


""" counting files in a directory """


def countFiles(location):
    count = 0
    count2 = 0
    for root, dirs, files in os.walk(location):
        for name in files:
            count += 1
    print("number of files :" + str(count))


""" selecting a random amount of files from a directory """


def randomSelection(location, newLocation, amount):
    filenames = []
    count = 0
    for root, dirs, files in os.walk(location):
        for name in files:
            filenames.append(name)
    samples = random.sample(filenames, amount)

    for sample in samples:
        print(sample)
        count += 1
        print(count)
        shutil.copy2(location + sample, newLocation)


"""deleting a selected amount of files randomly from a directory"""


def delrandomSelection(location, amount):
    filenames = []
    count = 0
    for root, dirs, files in os.walk(location):
        for name in files:
            filenames.append(name)
    samples = random.sample(filenames, amount)
    for sample in samples:
        print(sample)
        count += 1
        print(count)
        os.remove(location + sample)

