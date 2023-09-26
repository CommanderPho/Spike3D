#Evan Russenberger-Rosica
#Create a Grid/Matrix of Images
import PIL, os, glob
from pathlib import Path
from PIL import Image
from math import ceil, floor

# PATH = r"C:\Users\path\to\images"

BaseIconParentPath = Path(r'C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\EXTERNAL\Design\Icons\Potential')
OverlayIconsParentPath = Path(r'C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\EXTERNAL\Design\Icons\Potential\Overlays')

def build_icon(baseIconPath, overlayIconPath, outputIconPath=None):
    """ 2022-10-04 - This builds a simple icon with an overlay icon. Not quite working because the overlay icon is rendering with a white background. """
    MAX_IMG_SIZE = 256
    if outputIconPath is None:
        outputIconPath = 'new_im.png'

    base_img = Image.open(baseIconPath)
    img_width, img_height = base_img.size
    print(f'img_width: {img_width}, img_height: {img_height}')
    # find largest dimension:
    if img_width > MAX_IMG_SIZE:
        # clipping on widget
        scaling_factor = float(MAX_IMG_SIZE) / float(img_width) # get the scaling factored needed to bring the img_width down to MAX_IMG_SIZE
        print(f'scaling_factor: {scaling_factor}')
        frame_width = min(img_width, MAX_IMG_SIZE) # clip frame_width to max
        frame_height = ceil(img_height * scaling_factor) # wait this would mess up the aspect ratio if the image wasn't square and larger than 256 yeah?
    else:
        print(f'TODO: pretending everything is okay if the width is not too big.')
        frame_width = min(img_width, MAX_IMG_SIZE)
        frame_height = min(img_height, MAX_IMG_SIZE) # wait this would mess up the aspect ratio if the image wasn't square and larger than 256 yeah?

    base_img.thumbnail((frame_width, frame_height)) # ensures it is the same height as the base image, as it has the whitespace built in the image

    print(f'frame_width: {frame_width}, frame_height: {frame_height}')
    new_im = Image.new('RGB', (frame_width, frame_height))
    new_im.paste(base_img) # add the base image to the new icon

    overlay_img = Image.open(overlayIconPath)
    #Here I resize my opened image, so it is no bigger than 100,100

    overlay_img.thumbnail((frame_width, frame_height)) # ensures it is the same height as the base image, as it has the whitespace built in the image
    #Iterate through a 4 by 4 grid with 100 spacing, to place my image
    # y_cord = (j//images_per_row)*scaled_img_height
    # new_im.paste(im, (i,y_cord))
    new_im.paste(overlay_img)
    new_im.show()
    new_im.save(outputIconPath, "PNG")
    # Close the loaded images:
    base_img.close()
    overlay_img.close()

    return new_im


def build_icon_example_grid(icons_path=Path(r"C:\Users\path\to\images")):

    frame_width = 1920
    images_per_row = 5
    padding = 2

    os.chdir(icons_path)

    images = glob.glob("*.png")
    images = images[:30]                #get the first 30 images

    img_width, img_height = Image.open(images[0]).size
    sf = (frame_width-(images_per_row-1)*padding)/(images_per_row*img_width)       #scaling factor
    scaled_img_width = ceil(img_width*sf)                   #s
    scaled_img_height = ceil(img_height*sf)

    number_of_rows = ceil(len(images)/images_per_row)
    frame_height = ceil(sf*img_height*number_of_rows) 

    new_im = Image.new('RGB', (frame_width, frame_height))

    i,j=0,0
    for num, im in enumerate(images):
        if num%images_per_row==0:
            i=0
        im = Image.open(im)
        #Here I resize my opened image, so it is no bigger than 100,100
        im.thumbnail((scaled_img_width,scaled_img_height))
        #Iterate through a 4 by 4 grid with 100 spacing, to place my image
        y_cord = (j//images_per_row)*scaled_img_height
        new_im.paste(im, (i,y_cord))
        print(i, y_cord)
        i=(i+scaled_img_width)+padding
        j+=1

    new_im.show()
    new_im.save("out.jpg", "JPEG", quality=80, optimize=True, progressive=True)


def main():
    # selectedBaseIconFilename = 'timeline-svgrepo-com.svg' # doesn't work without conversion because it's an SVG
    selectedBaseIconFilename = 'heat-map-icon-21.jpg'

    # 'noise-control-off-remove'
    selectedOverlayIconFilename = r'png\1x\noise-control-off-delete.png'
    # selectedOverlayIconFilename = r'svg\noise-control-off-add.svg'

    baseIconPath = Path(r'C:\Users\pho\repos\VSCode Extensions\vscode-favorites\icons\favorites.png').resolve()
    # baseIconPath = BaseIconParentPath.joinpath(selectedBaseIconFilename)
    overlayIconPath = OverlayIconsParentPath.joinpath(selectedOverlayIconFilename)

    new_icon = build_icon(baseIconPath, overlayIconPath, outputIconPath='new_test_overlay_icon.png')
    return new_icon
    # return build_icon()



if __name__ == '__main__':
    main()
