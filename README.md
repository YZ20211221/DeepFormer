# DeepFormer
README for DeepFormer
DeepFormer is a hybrid network based on convolutional neural network and linear attention mechanism for identifying the function of DNA sequence.We compared DeepFormer with five excellent existing models on our own platform. Although the replication results differ from the original reported results of the five models, DeepFormer achieves an advanced performance of 0.9504AV_AUROC, 0.4658AV_AUPR, which is far superior to other prediction methods, both in terms of replication results and original paper results. The performance results of all models described in the original paper are shown in the following table.
	DeepSEA	DanQ	DanQ_JASPAR	DeepATT	DeepFormer
AV_AUROC	0.93260	0.93837	0.94174	0.94519	0.9504
AV_AUPR	0.34163	0.37089	0.37936	0.39522	0.4658

Requirement
Python (3.9) | selene (0.4.4.0) | einops (0.5.0) | pytorch (1.7.1) | torchvision (0.8.2) | torchaudio (0.7.2)

Data
You need to first download the training, validation, and testing sets from DeepSEA. You can download the datasets from 
http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz
After you have extracted the contents of the tar.gz file, move the 3 .mat files into the data/ folder.
We also transform the .mat files to .fasta files, you can get them from there
https://pan.baidu.com/s/1AaMBCgTzysljGg_JhyUKfA,
and the extraction code is a723. Similarly, move the files into the data/ folder.
The maize dataset used in the model generalization capability evaluation is also at the above link.

Usage
If you have everything installed, you can train a model initially as follows
sbatch DeepFormer_train.slurm

We have saved the optimal model at the same link above, you can get it from there.

If you do not want to train a model from scratch and just want to do predictions, please download the optimal model, then
sbatch predict.slurm

Acknowledgement
Thanks for Selene, a PyTorch-based deep learning library for sequence data, makes contribution to our research.
