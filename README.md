# cloth-rendering

### Description
* This repo provides a lightweight simulator for cloth using Blender 2.8. It is intended to provide a simulation environment for downstream robotics tasks with deformable objects (towel folding, blanket smoothing, curtain manipulation etc.)
  * `rope-blender.py`: renders images of deformable cloth into directory `cloth-rendering/images` and dumps a JSON file with ground truth annotations into the same folder.
  * `vis.py`: visualizes annotations on rendered images and dumps them into `cloth-rendering/annotated`
  * `mask.py`: renders segmentation masks and dumps them into `cloth-rendering/image_masks.` 
  
### Dependencies
All in Python3:
* Blender (2.80) (Download here: https://www.blender.org/download/Blender2.80/blender-2.80-macOS.dmg/)
* cv2
* numpy
  
### Example Renderings
<p float="left">
 <img src="https://github.com/priyasundaresan/cloth-rendering/blob/master/images/000010_rgb.png" height="200">
 <img src="https://github.com/priyasundaresan/cloth-rendering/blob/master/images/000015_rgb.png" height="200">
 <img src="https://github.com/priyasundaresan/cloth-rendering/blob/master/images/000020_rgb.png" height="200">
</p>

<p float="left">
 <img src="https://github.com/priyasundaresan/cloth-rendering/blob/master/annotated/000003_annotated.png" height="200">
 <img src="https://github.com/priyasundaresan/cloth-rendering/blob/master/annotated/000005_annotated.png" height="200">
 <img src="https://github.com/priyasundaresan/cloth-rendering/blob/master/annotated/000009_annotated.png" height="200">
</p>

<p float="left">
 <img src="https://github.com/priyasundaresan/cloth-rendering/blob/master/texture_images/000003_rgb.png" height="200">
 <img src="https://github.com/priyasundaresan/cloth-rendering/blob/master/texture_images/000005_rgb.png" height="200">
 <img src="https://github.com/priyasundaresan/cloth-rendering/blob/master/texture_images/000009_rgb.png" height="200">
</p>



### Setup
* After downloading Blender version 2.8, do the following steps:
* Add the following line to your .bashrc: 
  * `alias blender="/path/to/blender/blender.app/Contents/MacOS/blender"` replacing the path to blender.app with your downloaded version
* `cd` into the following directory: `/path/to/blender/blender.app/Contents/Resources/2.80/python/bin`
* To install dependencies, optionally make a python3 virtualenv, navigate into `cloth-rendering` and run `pip3 install -r requirements.txt`

### Rendering Usage
* Off-screen rendering: run `blender -b -P cloth-blender.py` (`-b` signals that the process will run in the background (doesn't launch the Blender app), `-P` signals that you're running a Python script)
* On-screen rendering: run `blender -P cloth-blender.py` (launches the Blender app once the script executes)

### Debugging/Development
* Bugs will most likely be caused by Blender version compatibility; note that this codebase is developed for Blender 2.80, so no guarantees about 2.7X or 2.81+
* First thing to check is stdout if you're running `blender -P cloth-blender.py`; (you won't see any output in the Blender app itself). If the error is about an API call, ensure that you're using Blender 2.80 (& if you're trying to make it forward or backward compatible, you may need to swap the call that errors with the version-compatible API call - check the Blender changelog)
* For adding new cloth features, it is almost always easiest to manually play around directly with meshes and objects in the Blender app. Once you get the desired functionality through manually playing around with it, head to the `Scripting` tab and it would have logged the corresponding API calls for everything you did (which you can directly use for scripting the functionality)
* For implementing new things, YouTube Blender tutorials are incredible! Even if they're manual, you can always port the functionality to a script by copying the steps and then converting them to script form by looking at the `Scripting` tab

### Example Workflow
* Run `blender -b -P cloth-blender.py` to produce renderings of the cloth in different states
* Run `python3 vis.py` to visualize the pixel annotations
* Run `python3 mask.py` to produce segmentation masks of images
* Use the images (/images), annotations (/images/knots_info.json), and segmentation masks (/image_masks) as training data
