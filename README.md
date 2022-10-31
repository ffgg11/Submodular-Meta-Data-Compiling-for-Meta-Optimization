# Submodular-Meta-Data-Compiling-for-Meta-Optimization

Here is the code for 'Submodular-Meta-Data-Compiling-for-Meta-Optimization'.

All hyperparameters are in the file "Hyper_paras.py". The hyperparameters "alpha_4" and "alpha_5" are the coefficients for diversity and cleaness criteria. And they satisfy alpha_4+alpha_5 = 1. 
 
Hyperparameter "meta_data_nums" can be used to change the number of the meta data.
"device_idx" is used to set the device id. 
"dataset" is data set you want to use. The default setting is CIFAR10 or CIFAR100. 
"imb_factor" is the imbalance factor of the data set. The hyperparameter is used in the file main.py.

Then use nohup python main.py to run the code.




If you find the paper is usful, please cite the paper as follows.


@inproceedings{submodular,
  title={Submodular Meta Data Compiling for Meta Optimization},
  author={Su, Fengguang and Zhu, Yu and Wu, Ou and Deng, Yingjun},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  year={2022},
  organization={Springer}
}
