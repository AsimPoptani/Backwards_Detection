import glob, os

def get_images_to_dict(directory='.',search="*.png"):
    os.chdir(directory)
    return glob.glob(search):
