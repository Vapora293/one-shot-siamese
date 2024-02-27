import os

folder = "ts_new"
eval = "ts_new\\eval"
output_folder = "ts_png"

def iterateThroughAllSubfoldersAndExtractAllPngsToOutputFolder():
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        # if folder
        if os.path.isdir(file_path):
            # iterate around all png files in the folder
            for png in os.listdir(file_path):
                if png.endswith(".png"):
                    # if the dir named as a second element after splitting its name based on a - doesn't exist create it
                    ts_type = "unknown"
                    if png.__contains__("-"):
                        ts_type = png.split("-")[0].strip(".png")
                    else:
                        ts_type = png.strip(".png")
                    future_path = os.path.join(output_folder, ts_type)
                    if not os.path.exists(future_path):
                        os.makedirs(future_path)
                    former_path = os.path.join(folder, filename, png)
                    new_path = os.path.join(output_folder, ts_type, png)
                    os.rename(former_path, new_path)


def convertToPng():
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        # if folder
        if os.path.isdir(file_path):
            # iterate around all png files in the folder
            for png in os.listdir(file_path):
                if png.endswith(".png"):
                    # if the dir named as a second element after splitting its name based on a - doesn't exist create it
                    ts_type = png.split("-")[1].strip(".png")
                    future_path = os.path.join(folder, ts_type)
                    if not os.path.exists(future_path):
                        os.makedirs(future_path)
                    former_path = os.path.join(folder, filename, png)
                    png = png.strip(png.split("-")[0])[1:]
                    new_path = os.path.join(folder, ts_type, png)
                    os.rename(former_path, new_path)

def getEvalDataset():
    eval_dataset = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isdir(file_path):
            appendedImages = 0
            desiredNumber = 0
            if (len(os.listdir(file_path)) > 2):
                desiredNumber = (len(os.listdir(file_path)) / 10 * 2)
                if desiredNumber < 1:
                    desiredNumber = 1
            for png in os.listdir(file_path):
                if appendedImages <= desiredNumber and desiredNumber != 0:
                    eval_dataset.append(os.path.join(file_path, png))
                    appendedImages += 1
    return eval_dataset


# convertToPng()
# iterateThroughAllSubfoldersAndExtractAllPngsToOutputFolder()
eval_dataset = getEvalDataset()
# I need to move all the images to the added folder official_ts_png/eval + the former path. Beware the folders might not exist, so you need to create them
for image in eval_dataset:
    image_stripped_path = image.strip(folder)
    # join eval path and the stripped path
    imageFullPath = os.path.join(eval, *image_stripped_path.split(os.path.sep))
    # If the folder doesn't exist create it
    if not os.path.exists(os.path.dirname(imageFullPath)):
        os.makedirs(os.path.dirname(imageFullPath))
    os.rename(image, imageFullPath)