# Results
This folder folder contains all .json and other files that include metrics, parameters, and results per ever version created. 

## coco_instances_results.json
These files contain the output of our trained model on the validation set for all versions. To parse and show these json files, refer to our data pre-processing section [here](https://colab.research.google.com/drive/1Czv3KcuMujaOg27u2mPzB-Pm4_wfrxn7?usp=sharing) Note:The data pre-processing is the same across all versions, so any version may be used to properly parse and annotate the data, however our best performing model is linked here.

## faces_val_coco.format.json
These files are to be used in correspondance with coco_instances_results.json and contain the corresponding image information for coco_instances_results. To correctly muse these json files in unison with coco_instances_results.json, refer to our data pre-processing section [here](https://colab.research.google.com/drive/1Czv3KcuMujaOg27u2mPzB-Pm4_wfrxn7?usp=sharing) Note: Again, any version may be used as the data processing was not altered across versions.

## metrics.json
Contains the accuracy metrics for each version trained, includes a series of AP scores: 
| AP | AP50 | AP75 | AP1 | APm | APs |.

Also contains all of the loss values across these values. 
- fast_rcnn: cls_accuracy, false_negatives, fg_cls_accuracy 
