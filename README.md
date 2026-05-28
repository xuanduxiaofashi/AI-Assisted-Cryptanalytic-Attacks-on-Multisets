# Exploring AI-Assisted Cryptanalytic Attacks on Multisets - Experimental Code

This repository contains the experimental implementation corresponding to Sections 3, 4, and 5 of the paper:

**_Exploring AI-Assisted Cryptanalytic Attacks on Multisets_**

---

## Section 3: Neural Differential-Linear Distinguishers and Key Recovery

- Neural differential-linear distinguishers for the ciphers: **SKINNY**, **LBLOCK**, **PRESENT**, **RECTANGLE**
- A 10-round key recovery attack using neural differential-linear techniques

**Distinguisher Code workflow:**

Step 1: Generate inputs for the neural network

- First, create a directory to store the generated plaintext-ciphertext data:

  mkdir cipher_txt_npy

- Then, compile and run the C++ data generation code:

  g++ step_1_*.cpp -o execute
  ./execute

Step 2: Convert .txt files into .npy format suitable for neural networks

  python3 step_2_process_ciphertext_2.py

Step 3: Train the neural network

- Manually edit `step_3_train_model.py` to specify the names of the generated `.npy` files.
- Then start training:

  python3 step_3_train_model.py

**Key recovery workflow:**
1. Construct a neural differential-linear distinguisher  
2. Build a Wrong Key Response Profile (WKRP)  
3. Recover the key using the distinguisher and WKRP

---

## Section 4: A Hybrid Input Strategy for Integral and Differential-Linear Features

This section explores a hybrid input strategy that combines **differential-linear** and **integral** features in a neural model. Two examples are provided.

For each example, three types of models are trained:

1. A neural distinguisher learning the **integral** feature  
2. A neural distinguisher learning the **differential-linear** feature  
3. A hybrid model learning both **differential-linear** and **integral** features from the combined input

---

## Section 5: Cryptographic Properties Learned from Multisets

This section investigates what kinds of cryptographic features the neural distinguishers learn when trained on multisets, including:

- **Integral properties**  
- **Differential properties**  
- **Differential-linear properties**

---

## Environment

- **Python version:** 3.8.10  
- **TensorFlow version:** 2.13.1  
- **Keras version:** 2.13.1  
- **NumPy version:** 1.24.3

---

## Requirements

You can install all required dependencies with:

```bash
pip install -r requirements.txt
