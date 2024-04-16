
# Classifiers Enhanced by Pre-training

This is a project that builds image classifiers using the CLIP (ViT-B/32) visual encoder. The dataset used is CIFAR-100, where the original training set was split into a new training set and a validation set in an 8:2 ratio. The experiment is divided into four parts:
1. **Zero-shot**: Test the zero-shot classification performance of the pretrained CLIP itself.
2. **Linear-probe**: Add a fully connected layer on top of the pretrained CLIP (ViT-B/32) visual encoder, freeze the visual encoder, and train only the fully connected layer.
3. **Fine-tuning**: Add a fully connected layer on top of the pretrained CLIP (ViT-B/32) visual encoder and train all parameters.
4. **Train-from-scratch**: Use the same model architecture as Linear-probe and Fine-tuning, but train from scratch on the training set without using any pretrained weights.

## Installation Steps

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Gengsheng-Li/Classifiers-enhanced-by-pre-training.git
cd Classifiers-enhanced-by-pre-training
conda create -n classifier python=3.8
conda activate classifier
pip install -r requirements.txt
```

## Equipment
All code was run under Linux with Slurm installed. 

The acceleration device used was: 1x NVIDIA A40.

## Usage

### Download the Trained Models and CIFAR-100 Dataset from Huggingface

From https://huggingface.co/RyukiRi/Classifiers-Enhanced-by-Pre-training, you can download the following trained model weights and CIFAR-100 dataset for running this project:
- `fine-tune-best.pth`: Best model weights after fine-tuning.
- `linear-probe-best.pth`: Best model weights after the linear probe training.
- `train-from-scratch-best.pth`: Best model weights trained from scratch.

Please download these files and place them under the `results/` directory within the project folder.

- `cifar-100-python.tar.gz`: CIFAR-100 dataset.

Please download this file and place it under the `data/` directory within the project folder.

### Training the Model
If you are using Linux with Slurm installed, please choose one of the following commands to train the model, depending on your needs:

- For Linear-probe or Fine-tuning modes, use the following command:
  ```bash
  sbatch run_train_from_pretrain.slurm
  ```

- For Train-from-scratch mode, use the following command:
  ```bash
  sbatch run_train_from_scratch.slurm
  ```
  Note: You need to change some arguments in these .slurm files to fit your need and device.

If you are using Windows or Linux without Slurm, you can also directly run following commands in cmd:

- For Linear-probe or Fine-tuning modes, use the following command:
  ```bash
  python train_from_pretrain.py --train_mode 'fine-tune' \
                            --lr 1e-6 \
                            --epochs 100 \
                            --batch_size 512 \
                            --validation_split 0.2 \
                            --save_path "results" \
                            --exp_name "train_from_pretrain" \
  ```
  You can set `--train_mode` as 'fine-tune' or 'linear-probe' to perfome different training mode.

- For Train-from-scratch mode, use the following command:
  ```bash
  python train_from_scratch.py --lr 5e-5 \
                            --T_max 50 \
                            --weight_decay 1e-4 \
                            --epochs 100 \
                            --batch_size 128 \
                            --validation_split 0.2 \
                            --save_path "results" \
                            --exp_name "train_from_scratch"
  ```

### Testing the Model
We evaluate the performance of classifiers from five perspectives, including `Accuracy`, `Precision`, `Recall`, `F1 Score`, and `Confusion matrix`. The following commands will calculate or draw them automatically.

If you are using Linux with Slurm installed, use the following command:
```bash
sbatch run_test.slurm
```
  Note: You need to change some arguments in this .slurm file to fit your need and device.


If you are using Windows or Linux without Slurm, you can also directly run following commands in cmd:

- For testing CLIP (ViT-B/32) zero-shot performance, use the following command:
  ```bash
  python test.py --zero_shot
  ```

- For testing the performance of the results of Train-from-scratch, Linear-probe, or Fine-tune, use the following commands please:
  ```bash
  python test.py --model_pth 'results/train-from-scratch-best.pth'

  python test.py --model_pth 'results/linear-probe-best.pth'

  python test.py --model_pth 'results/fine-tune-best.pth'
  ```
  You can set `--check_model` to check the model you choosed and print some detailed info layer by layer, just like:
  ```bash
  python test.py --model_pth 'results/train-from-scratch-best.pth' --check_model
  ```

## Contributing

Contributions are welcome! Please discuss the changes you wish to make via issues or pull requests first.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

If you have any questions, please contact us at [ligengchengucd@163.com].
