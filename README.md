# Car extract information
Extract car information

## Setup
* Clone project
        
        git clone https://github.com/khaidoandk97/car-extract-information.git
        
* create environment name: **car_infomation_extraction**

        conda env create -f environment.yml
    
* activate environment
        
        conda activate car_infomation_extraction
        
* download **[trained_models](https://drive.google.com/drive/folders/1W7X-JZvLZzNvcUWuBX9iqrjrbzMSuEeX?usp=sharing)** and put it them into the project direction:

## Run test 
* Video:
        
        python test/run_video.py --video-path [absolute path to video] --save-path [absolute path to save video] --show [option]
        
* example:
        
        python test/run_video.py --video-path videos/video02.mp4 --save-path out.mp4 --show
        
        python test/run_video.py --video-path videos/video02.mp4 --save-path out.mp4
        

## Demo
**[video demo](https://drive.google.com/file/d/1p6iv5oTWnoVpTotZw-x0s-Y3ggtJU9Rd/view?usp=sharing)**

![Example Image](/assets/demo.png)