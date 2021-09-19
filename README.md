# SVDGAN
Targeted Black-Box Implementation of SVDGAN
  1. To provide train and test directory path, use parameters.py file inside utils folder
  2. To change hyper-parameters, use parameters.py file inside utils folder
  3. To provide train and test adversarial samples directory path, use parameters.py file inside utils folder
  4. To provide perturbations directory path, use parameters.py file inside utils folder
  5. To see perturbations generated from Auto-Encoder, set SHOW_PERTURBATIONS=True in parameters.py
  6. To see adversarial examples generated from SVDGAN, set SHOW_TRAIN_ADVERSARIAL_IMAGES=True in parameters.py
  7. To see test adversarial examples generated from the trained SVDGAN, set SHOW_TEST_ADVERSARIAL_IMAGES=True in parameters.py
  8. After providing directory paths, use main.py to initiate the training process.
  9. To test the SVDGAN, provide trained models directory paths in parameters.py and use test.py 
