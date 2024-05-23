import os
import random
import shutil
from itertools import islice

outputFolderPath = "Datasets/SplitData"
inputFodlerPath = "Datasets/All"
splitRatio = {"train": 0.7, "validation": 0.2, "test": 0.1}

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
listNames = os.listdir(inputFodlerPath)
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


# ----- Copy the Files ----- #
sequence = ["train", "validation", "test"]
for i,out in enumerate(Output):
    for fileName in out:
        shutil.copy(f"{inputFodlerPath}/{fileName}.jpg", f"{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg")
        shutil.copy(f"{inputFodlerPath}/{fileName}.txt", f"{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt")