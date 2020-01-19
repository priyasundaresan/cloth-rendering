# cloth-rendering

### Dependencies
All in Python3:
* Blender (2.80) (Download here: https://www.blender.org/download/Blender2.80/blender-2.80-macOS.dmg/)
* cv2
* numpy

### Description
* This repo provides a lightweight simulator for cloth using Blender 2.8. It is intended to provide a simulation environment for downstream robotics tasks with deformable objects (towel folding, blanket smoothing, curtain manipulation etc.)
  * `rope-blender.py`: renders images of deformable cloth into directory `cloth-rendering/images` and dumps a JSON file with ground truth annotations into the same folder.
  * `vis.py`: visualizes annotations on rendered images and dumps them into `cloth-rendering/annotated`
  * `mask.py`: renders segmentation masks and dumps them into `cloth-rendering/image_masks.` 
  
### Example Renderings
<p float="left">
 <img src="https://github.com/priyasundaresan/cloth-rendering/blob/master/images/000010_rgb.png" height="200">
 <img src="https://github.com/priyasundaresan/cloth-rendering/blob/master/images/000015_rgb.png" height="200">
 <img src="https://github.com/priyasundaresan/cloth-rendering/blob/master/images/000020_rgb.png" height="200">
</p>

### Setup
* After downloading Blender version 2.8, do the following steps:
* Add the following line to your .bashrc: 
  * `alias blender="/path/to/blender/blender.app/Contents/MacOS/blender"` replacing the path to blender.app with your downloaded version
* `cd` into the following directory: `/path/to/blender/blender.app/Contents/Resources/2.80/python/bin`
* To install dependencies, optionally make a python3 virtualenv, navigate into `cloth-rendering` and run pip3 install -r requirements.txt

### Rendering Usage
* Off-screen rendering: run `blender -b -P cloth-blender.py` (-b signals --background, -P signals --python)
* On-screen rendering: run `blender -P cloth-blender.py`
* For debugging purposes, you can also open a scripting window in Blender (on the top nav-bar menu, look for `Scripting`), click `+New`, copy the contents of `rope-blender.py` and hit `Run Script`
  * To manually deform the cloth, select the cloth in `'EDIT'` mode (by pressing `Tab`), press `'G'`, and grab any of the pinned vertices with a mouse to move the rope. You can select groups of nodes at a time by pressing `'Ctrl'` while grabbing nodes.

### Example Workflow
* Run `blender -b -P rope-blender.py` to produce renderings of the cloth in different states
* Run `python3 vis.py` to visualize the pixel annotations
* Run `python3 mask.py` to produce segmentation masks of images
* Use ground truth annotations and segmentation masks as training inputs for your application!
