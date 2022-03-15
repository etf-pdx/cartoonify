# Cartoonify
Make images and videos into cartoons!
Read the [paper here](https://github.com/etf-pdx/cartoonify/blob/master/paper/cartoonify.pdf)!

## Setting Up
You'll need a Python 3 Anaconda environment set up.
The three required dependencies for this project are:
- Numpy
- OpenCV
- Moviepy

## Running the script
### When you're unsure
To see how this script is run you can type:
```bash
python3 cartoonify.py -h
```

### Cartoonifying an image
NOTE: If no ouput path is provided, it will default to cartoon_{input.png}
```bash
# will write to cartoon_input_image.png
python3 cartoonify.py input_image.png

# will write to output_image.png
python3 cartoonify.py input_image.png -o output_image.png
```

### Cartoonifying a video

```bash
# will write to cartoon_input_video.mp4
python3 cartoonify.py -v input_video.mp4

# will write to output_video.mp4
python3 cartoonify.py -v input_video.mp4 -o output_video.mp4
```
