import os

folder = "official_ts_png"

for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    # if folder
    if os.path.isdir(file_path):
        # iterate around all png files in the folder
        for png in os.listdir(file_path):
            # if the dir named as a second element after splitting its name based on a - doesn't exist create it
            ts_type = png.split("-")[1].strip(".png")
            future_path = os.path.join(folder, ts_type)
            if not os.path.exists(future_path):
                os.makedirs(future_path)
            former_path = os.path.join(folder, filename, png)
            png = png.strip(png.split("-")[0])[1:]
            new_path = os.path.join(folder, ts_type, png)
            os.rename(former_path, new_path)
