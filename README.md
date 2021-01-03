# RED-CNN-RAM
Incorporation of Residual Attention Modules into Two Neural Networks for Low-Dose CT Denoising


# Use
1. run `python prep.py` to convert 'dicom file' to 'numpy array'
2. run `python main.py --load_mode=0` to training. If the available memory(RAM) is more than 10GB, it is faster to run `--load_mode=1`.
3. run `python main.py --mode='test' --test_iters=***` to test.
