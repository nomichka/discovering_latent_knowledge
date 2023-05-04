from sklearn.linear_model import LogisticRegression
from utils import get_parser, load_all_generations, CCS, CCSPCA
import matplotlib.pyplot as plt
import os

def main(args, generation_args):
    # load hidden states and labels
    # shape [num_examples, model_dim, num_layer]
    neg_hs_all, pos_hs_all, y = load_all_generations(generation_args)

    # Make sure the shape is correct
    assert neg_hs_all.shape == pos_hs_all.shape

    # Evaluate across layers
    if args.all_layers:
        num_layers = neg_hs_all.shape[-1]
        ccs_accuracy_all = []
        lr_accuracy_all = []
        pca_accuracy_all = []
        for layer in range(num_layers):
            neg_hs = neg_hs_all[..., layer]
            pos_hs = pos_hs_all[..., layer]

            if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
                neg_hs = neg_hs.squeeze(1)
                pos_hs = pos_hs.squeeze(1)

            # Very simple train/test split (using the fact that the data is already shuffled)
            neg_hs_train, neg_hs_test = neg_hs[:len(neg_hs) // 2], neg_hs[len(neg_hs) // 2:]
            pos_hs_train, pos_hs_test = pos_hs[:len(pos_hs) // 2], pos_hs[len(pos_hs) // 2:]
            y_train, y_test = y[:len(y) // 2], y[len(y) // 2:]

            # Make sure logistic regression accuracy is reasonable; otherwise our method won't have much of a chance of working
            # you can also concatenate, but this works fine and is more comparable to CCS inputs
            x_train = neg_hs_train - pos_hs_train  
            x_test = neg_hs_test - pos_hs_test
            lr = LogisticRegression(class_weight="balanced")
            lr.fit(x_train, y_train)
            lr_acc = lr.score(x_test, y_test)
            print("Logistic regression accuracy: {}".format(lr_acc))

            # Set up CCS. Note that you can usually just use the default args by simply doing ccs = CCS(neg_hs, pos_hs, y)
            ccs = CCS(neg_hs_train, pos_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size, 
                            verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay, 
                            var_normalize=args.var_normalize)
            
            # train and evaluate CCS
            ccs.repeated_train()
            ccs_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
            print("CCS accuracy: {}".format(ccs_acc))

            # Set up PCA.
            pca = CCSPCA(neg_hs_train, pos_hs_train)
            pca_acc = pca.get_acc(neg_hs_test, pos_hs_test, y_test)
            print("PCA accuracy: {}".format(pca_acc))

            ccs_accuracy_all.append(ccs_acc)
            lr_accuracy_all.append(lr_acc)
            pca_accuracy_all.append(pca_acc)
    
    return ccs_accuracy_all, lr_accuracy_all, pca_accuracy_all


if __name__ == "__main__":
    parser = get_parser()
    generation_args = parser.parse_args()  # we'll use this to load the correct hidden states + labels
    # We'll also add some additional args for evaluation
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--ntries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ccs_batch_size", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ccs_device", type=str, default="cuda")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--var_normalize", action="store_true")
    args = parser.parse_args()
    ccs_accs, lr_accs, pca_accuracy_all = main(args, generation_args)
    print(ccs_accs, lr_accs, pca_accuracy_all)
    plt.plot(ccs_accs)
    plt.plot(lr_accs)
    plt.plot(pca_accuracy_all)
    plt.xlabel("layer")
    plt.ylabel("test accuracy")
    if not os.path.exists("figures"):
        os.makedirs("figures")
    filename = ("_").join([args.model_name, args.dataset_name])
    plt.savefig("figures/" + filename)

