import glob,os
from multiprocessing import Pool
from PIL import Image
def get_image_info(image_path):
    # image=Image.open(image_path)
    # Height and width
    # width,height = image.size
    # Dictionary
    image_dir='/'.join(image_path.split("/")[:-1])+'/'
    # Name
    image_name=image_path.split("/")[-1]
    # {imageID , Directory, size}
    return {"image_id":image_name, "dir":image_dir, "height":800, "width":1280}

def get_images_to_dict(directory='./',search="*.png"):
    images=[]
    currentDirectory = os.getcwd()
    os.chdir(directory)
    videos= glob.glob(search)

    videos=[directory + s for s in videos]
    with Pool(8) as p:
        images=p.map(get_image_info,videos)
    os.chdir(currentDirectory)
    
    images = sorted(images, key=lambda image: image['image_id'])
    return images