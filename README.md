# **Deep Model Compression with PyTorch**  
This project implements **structured pruning** to reduce the size of deep neural networks, specifically **LeNet5, VGG16, ResNet50, and DenseNet121**, while maintaining performance.  

## **Project Structure**
```
Deep_Model_Compression_PyTorch/
│── models/          # Model definitions (LeNet5, VGG16, ResNet50, DenseNet121)
│── weights/         # Pretrained and pruned model weights
│── train/           # Training scripts for different architectures
│── densenet121_prune.py  # Pruning script for DenseNet121
│── utils.py         # Helper functions
│── README.md        # Project documentation
```

## **Installation**
Before running the scripts, install the necessary dependencies:
```bash
pip install -r requirements.txt
```
📌 If you face compatibility issues, you can use the Python and library versions listed in requirements.txt.

## **Training a Model**
To train a specific model (e.g., VGG16), run:
```bash
python train/vgg16_train.py
```

## **Pruning a Model**
To prune **DenseNet121**, run:
```bash
python densenet121_prune.py
```
Modify `densenet121_prune.py` to adjust pruning parameters as needed.

## **Structured Pruning**
This project removes **entire filters** from convolutional layers, reducing the model size while keeping it operational. This method improves efficiency without requiring specialized hardware.

✅ **Reduces computation cost**  
✅ **Decreases model size**  
✅ **Maintains overall architecture**  

## **Results & Benchmarks**
| Model      | Original Size | Pruned Size | Pruned % | Accuracy Drop |
|------------|--------------|-------------|----------|--------------|
| LeNet5     | XX MB        | XX MB       | XX%      | X.XX%       |
| VGG16      | XX MB        | XX MB       | XX%      | X.XX%       |
| ResNet50   | XX MB        | XX MB       | XX%      | X.XX%       |
| DenseNet121| XX MB        | XX MB       | XX%      | X.XX%       |

(*Replace `XX` with actual benchmark results after running experiments.*)

## **Future Improvements**
- Experiment with different structured pruning ratios.
- Implement fine-tuning strategies to recover accuracy.
- Explore deployment optimizations for real-world applications.

---