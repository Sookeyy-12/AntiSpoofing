import os
import random
import shutil
from itertools import islice

outputFolderPath = "Datasets/SplitData"
inputFolderPath = "Datasets/All"
splitRatio = {"train": 0.7, "validation": 0.2, "test": 0.1}
classes = ["fake", "real"]

try:
    shutil.rmtree(outputFolderPath)
    print("Removed Directory")
except OSError as e:
    os.mkdir(outputFolderPath)


# ----- Directories to Create ----- #
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/validation/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/validation/labels", exist_ok=True)


# ----- Get the Names ----- #
listNames = os.listdir(inputFolderPath)
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))


# ----- Shuffle ----- #
random.shuffle(uniqueNames)


# ----- Find the number of images for each folder ----- #
lenData = len(uniqueNames)
print(f"Total Images: {len(uniqueNames)}")
lenTrain = int(lenData*splitRatio["train"])
lenValidation = int(lenData*splitRatio["validation"])
lenTest = int(lenData*splitRatio["test"])


# ----- Add Remaining images in training ----- #
if (lenTrain + lenValidation + lenTest) != lenData:
    lenTrain += lenData - (lenTrain + lenValidation + lenTest)

print(f"Train: {lenTrain}, Validation: {lenValidation}, Test: {lenTest}")


# ----- Split the List ----- #
lengthToSplit = [lenTrain, lenValidation, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]

print(f"Split: {len(Output[0])}, {len(Output[1])}, {len(Output[2])}")
print(f"Splitting the Dataset... (This may take some time depending on the number of images)")


# ----- Copy the Files ----- #
sequence = ["train", "validation", "test"]
for i,out in enumerate(Output):
    for fileName in out:
        shutil.copy(f"{inputFolderPath}/{fileName}.jpg", f"{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg")
        shutil.copy(f"{inputFolderPath}/{fileName}.txt", f"{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt")

print("Split Process Completed...")


# ----- Creating Data.yaml file ----- #
dataYAML = f'path: ../Data\n\
train: ../train/images\n\
val: ../validation/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'

f = open(f"{outputFolderPath}/data.yaml", 'a')
f.write(dataYAML)
f.close()

print("Data.yaml File Created...")
print("Zipping the Files... (This may take some time depending on the number of images.)")

# zip all the files in this directory into a zip file named "data.zip"
shutil.make_archive(outputFolderPath, 'zip', outputFolderPath, verbose=True)

print("Zipped the Files...")