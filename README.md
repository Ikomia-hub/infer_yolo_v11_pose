<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_yolo_v11_pose</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_yolo_v11_pose">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_yolo_v11_pose">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_yolo_v11_pose/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_yolo_v11_pose.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run YOLOv11 pose estimation.

![MJ YOLOv11 pose estimation](https://raw.githubusercontent.com/Ikomia-hub/infer_yolo_v11_pose_estimation/main/icons/output1.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_yolo_v11_pose_estimation", auto_connect=True)

# Run on your image  
wf.run_on(url="https://cdn.nba.com/teams/legacy/www.nba.com/bulls/sites/bulls/files/jordan_vs_indiana.jpg")

# Inpect your result
display(algo.get_image_with_graphics())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters
- **model_name** (str) - default 'yolo11m-pose': Name of the YOLO11 pre-trained model. Other model available:
    - yolo11n-pose
    - yolo11s-pose
    - yolo11l-pose
    - yolo11x-pose
- **input_size** (int) - default '640': Size of the input image.
- **conf_thres** (float) default '0.25': Box threshold for the prediction [0,1].
- **iou_thres** (float) - default '0.7': Intersection over Union, degree of overlap between two boxes [0,1].
- **cuda** (bool): If True, CUDA-based inference (GPU). If False, run on CPU.
- **model_weight_file** (str, *optional*): Path to model weights file .pt.

**Parameters** should be in **strings format**  when added to the dictionary.

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_yolo_v11_pose_estimation", auto_connect=True)

algo.set_parameters({
    "model_name": "yolo11m-pose",
    "conf_thres": "0.5",
    "input_size": "640",
    "iou_thres": "0.5",
    "cuda": "True"
})

# Run on your image  
wf.run_on(url="https://cdn.nba.com/teams/legacy/www.nba.com/bulls/sites/bulls/files/jordan_vs_indiana.jpg")

# Inpect your result
display(algo.get_image_with_graphics())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_yolo_v11_pose_estimation", auto_connect=True)

# Run on your image  
wf.run_on(url="https://cdn.nba.com/teams/legacy/www.nba.com/bulls/sites/bulls/files/jordan_vs_indiana.jpg")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```