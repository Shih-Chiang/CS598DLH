# CS598DLH
Team Number: 113
Team Member: Shih-Chiang Lee, Xiaoyi Jie
# The paper we choose is No.39, "A study of deep learning methods for de-identification of clinical notes in crossinstitute settings"
Citation to the original paper:
Yang, X., Lyu, T., Li, Q. et al. A study of deep learning methods for de-identification of clinical notes in cross-institute settings. BMC Med Inform Decis Mak 19, 232 (2019). https://doi.org/10.1186/s12911-019-0935-4

# Using Explaination
It is a reprocuce of the paper's experiment.
To run this code, first need to use the main.py in the Pre-processing folder to do the pre-processing. Then, change the data path of training data and test data in the task_ner_for_genia.py. In the code we uploaded, it is the case of using i2b2 data to train and using i2b2 to test. We also done the case of using genia data to train and using genia data to test; and the case of using both i2b2 data and genia data to train and genia data to test. The result is stored in the result folder.

# Data download instruction
The i2b2 data could be applied through https://www.i2b2.org/NLP/DataSets/
It guides you to download the n2b2 dataset, which is same as i2b2. It has some explaination about why dataset name is changed to n2b2 on the webset. In the whole dataset, we just choose the one in 2014.
The genia could be applied through https://data.world/chhs/579fbb1f-0078-4fe3-9bc8-c09d91ad3589

# Table of result
https://github.com/Shih-Chiang/CS598DLH/blob/main/result/table.PNG?raw=true
![image](https://github.com/Shih-Chiang/CS598DLH/blob/main/result/table.PNG?raw=true)
