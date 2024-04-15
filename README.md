
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
All code was run on under Linux with Slurm installed. 

The acceleration device used was: 1x NVIDIA A40.

## Usage

### Training the Model
Choose one of the following commands to train the model, depending on your needs:

- For Linear-probe or Fine-tuning modes, use the following command:
  ```bash
  sbatch run_train_from_pretrain.slurm
  ```

- For Train-from-scratch mode, use the following command:
  ```bash
  sbatch run_train_from_scratch.slurm
  ```

### Testing the Model
To test the model, use the following command:
```bash
sbatch run_test.slurm
```

## Contributing

Contributions are welcome! Please discuss the changes you wish to make via issues or pull requests first.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

If you have any questions, please contact us at [ligengchengucd@163.com].
