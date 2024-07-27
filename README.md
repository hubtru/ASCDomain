<!-- # DCASE24_Task1
Data loader and solution method for the DCASE 2024 Challenge Task1 

Task 1: Data-Efficient Low-Complexity Acoustic Scene Classification

Official DCASE Task desciption: https://dcase.community/challenge2024/task-data-efficient-low-complexity-acoustic-scene-classification -->

# ASC Domain
This repository contains the code to reproduce the results of the Truchan_LUH submission to  [DCASE24 Task 1 "Data-Efficient Low-Complexity Acoustic Scene Classification"](https://dcase.community/challenge2024/task-data-efficient-low-complexity-acoustic-scene-classification) challenge.

- Technical Report:[here](https://dcase.community/documents/challenge2024/technical_reports/DCASE2024_Truchan_3_t1.pdf)
- System Results: [here](https://dcase.community/challenge2024/task-data-efficient-low-complexity-acoustic-scene-classification-results)

The codebase for this repository is the baseline for task1: [here](https://github.com/CPJKU/dcase2024_task1_baseline)


## Setup
Create a conda environment
```
conda create -n asc python=3.10
conda activate asc
```
Download the dataset from [this](https://zenodo.org/record/6337421) location and extract the files.

There are a total of 5 architectures:
1. Isotropic
2. Siren
3. Adverserial
4. RSC
5. ASC Domain

Only Isotropic, Siren and RSC were submitted. In each experiment folder there is a ```dataset``` folder with a ```dcase24.py``` file, where the path to the datset has to be specified:
```
dataset_dir = None
```
All experiments have an argument ```split``` which specifies the corresponding split: 5, 10, 25, 50,100 are available

## Device Impulse Response
The device impulse response augmentation has shown great success in previous submissions and is also used in in this submission. The device impulse responses are provided by [MicIPR](http://micirp.blogspot.com/). All files are shared via Creative Commons license. All credits go to MicIRP & Xaudia.com.

## Isotropic Architectures

### Isotropic
Run isotropic training
```
python run_isotropic_training.py 
```

Run isotropic with mixstyle from [here](https://arxiv.org/pdf/2104.02008)
```
python run_isotropic_training.py  --model=mix
```

Run isotropic without activation motivated from [here](https://arxiv.org/pdf/1610.02357)
```
python run_isotropic_training.py  --model=noact
```

### Siren
Run siren training
```
python run_siren_training.py 
```

## Domain Generalization Techniques
Previous domain generalization techniques have used augmentation to generalization. For next two achitectures we conduct representation learning experiments with the isotropic architecture as a backbone model. Two representation learning techniques from [DeepDG](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG) were chosen:
- Domain Adverserial Neural Network (here called adverserial)
- Representation Self Challenging (RSC)

### Adverserial
Run adverserial training
```
python run_adv_training.py 
```

### RSC
Run RSC training
```
python run_rsc_training.py 
```

## ASC Domain
ASC Domain combines the adverserial approach with knowledge distillation. The training procedure and teacher models were taken from [cpjku_dcase23](https://github.com/fschmid56/cpjku_dcase23) and [EfficientAT](https://github.com/fschmid56/EfficientAT). We train a total of 4 differenct architectures:
- MobileNet
- Dynamic MobileNet
- CP-ResNet
- PaSST
each with different training setups and version of the architecture leading to a total of 22 teacher models. Each teacher model is trained on the 5 splits, resulting in 110 models.

Run teacher model training
### MobileNet
Run MobileNet training. Argument width=[0.4, 0.5, 1.0]
```
python run_mn_training.py --width=0.4 
```

### Dynamic MobileNet
Run Dynamic MobileNet training. Argument width=[0.4, 1.0]
```
python run_dymn_training.py --width=0.4 
```

### PaSST
Run PaSST training
```
python run_passt_training.py 
```

### CP-ResNet
Run CP-ResNet training
```
python run_cp-resnet_training.py
```

### Teacher Student Single Training
Run single teacher student training with teacher and Isotropic as student
```
python run_convmixer_training.py --teacher=<teacher_name>
```
Example: Run single teacher student training with PaSST teacher and Isotropic as student
```
python run_convmixer_training.py --teacher=passt_dir_fms 
```

### Teacher Student Ensemble Training
Run ensemble teacher student training with teacher and Isotropic as student
```
python run_convmixer_training.py --teacher=best
```
Run ensemble teacher student training with teacher and Siren as student
```
python run_siren_training.py --teacher=best
```

### Teacher Student Ensemble Adverserial Training
Run ensemble teacher student adverserial training with teacher and Isotropic as student
```
python run_convmixer_adv_training.py --teacher=best
```
Run ensemble teacher student adverserial training with teacher and Siren as student
```
python run_siren_adv_training.py --teacher=best
```








