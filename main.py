from Q1_4 import Q1_4
from Q2_3 import Q2, Q3


def main():
    """
    Entry point to run all parts of the convolutional autoencoder assignment:

    - Q1: Unsupervised autoencoder training (from scratch)
    - Q2: Supervised classification with encoder + MLP
    - Q3: Pretrained encoder (from Q1) + trained MLP
    - Q4: Train decoder using frozen encoder from Q2
    """

    # print("=== Running Q1: Autoencoder from scratch ===")
    # Q1_4(pre_trained=False)
    #
    # print("\n=== Running Q2: Classifier (encoder + MLP) ===")
    # Q2()
    #
    # print("\n=== Running Q3: Frozen encoder + MLP ===")
    # Q3()
    #
    # print("\n=== Running Q4: Train decoder with pretrained classification encoder ===")
    Q1_4(pre_trained=True)


if __name__ == '__main__':
    main()
