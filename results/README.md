# Results
This folder folder contains all .json and other files that include metrics, parameters, and results per ever version created. 

## coco_instances_results.json
These files contain the output of our trained model on the validation set for all versions. To parse and show these json files, refer to our data pre-processing section [here](https://colab.research.google.com/drive/1Czv3KcuMujaOg27u2mPzB-Pm4_wfrxn7?usp=sharing) Note:The data pre-processing is the same across all versions, so any version may be used to properly parse and annotate the data, however our best performing model is linked here.

## faces_val_coco.format.json
These files are to be used in correspondance with coco_instances_results.json and contain the corresponding image information for coco_instances_results. To correctly muse these json files in unison with coco_instances_results.json, refer to our data pre-processing section [here](https://colab.research.google.com/drive/1Czv3KcuMujaOg27u2mPzB-Pm4_wfrxn7?usp=sharing) Note: Again, any version may be used as the data processing was not altered across versions.

## metrics.json
Contains the accuracy metrics for each version trained, and contains all of the loss values and others specified below: 

- AP scores for "bbox" and "segm": | AP | AP50 | AP75 | AP1 | APm | APs |
- fast_rcnn: cls_accuracy, false_negatives, fg_cls_accuracy
- loss: loss_box_reg, loss_cls, loss_mask
- loss_rpn: loss_rpn_cls, loss_rpn_loc
- lr
- mask_rcnn: accuracy, false_negative, false_positive
- rank_data_time
- roi_head: num_bg_samples, num_fg_samples, num_neg_anchors, num_pos_anchors
- total_loss

For a more detailed insight to how these metrics were used, refer to our [report](https://docs.google.com/document/d/1jopVcW5oSQAM1AiB77bWeUELJqZ4IWX0DPezHU_gHWk/edit?tab=t.0#heading=h.w6zcozas85jc)
