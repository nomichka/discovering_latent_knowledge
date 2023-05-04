from sklearn.linear_model import LogisticRegression
from utils import get_parser, load_all_generations, CCS, CCSPCA
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize
from sklearn import preprocessing
import os

def main(generation_args):
    # load hidden states and labels
    # shape [num_examples, model_dim, num_layer]
    neg_hs_all, pos_hs_all, neg_non_hs_all, pos_non_hs_all, y = load_all_generations(generation_args)

    # Make sure the shape is correct
    assert neg_hs_all.shape == pos_hs_all.shape

    neg_hs = neg_hs_all[..., -1]
    pos_hs = pos_hs_all[..., -1]
    neg_non_hs = neg_non_hs_all[..., -1]
    pos_non_hs = pos_non_hs_all[..., -1]
    diff = neg_hs - pos_hs
    diff = normalize(diff)
    # diff = scale(diff)
    scaler = preprocessing.StandardScaler()
    diff = scaler.fit_transform(diff)

    diff_non = neg_non_hs - pos_non_hs
    diff_non = normalize(diff_non)
    # diff_non = scale(diff_non)
    scaler = preprocessing.StandardScaler()
    diff_non = scaler.fit_transform(diff_non)
    
    # Create a PCA instance
    pca = PCA(n_components=2)
    diff_transformed = pca.fit_transform(diff)  # (N, 2)
    
    pca_non = PCA(n_components=2)
    diff_non_transformed = pca_non.fit_transform(diff_non)  # (N, 2)

    assert diff_transformed.shape == diff_non_transformed.shape

    # plt.scatter(diff_transformed[:, 0], diff_transformed[:, 1])
    # plt.scatter(diff_non_transformed[:, 0], diff_non_transformed[:, 1])
    # plt.legend(["with context", "without context"])
    # plt.title("Projection of difference embedding onto first two principal components")
    # plt.xlabel("First component")
    # plt.ylabel("Second component")
    # plt.savefig("figures/compare.png")

    colors = ['red','green']
    plt.xlabel("First component")
    plt.ylabel("Second component")
    plt.title("Projection onto first two principal components (with context)")
    plt.scatter(diff_transformed[:, 0], diff_transformed[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(colors))
    plt.savefig("figures/with_context.png")


    

            
if __name__ == "__main__":
    parser = get_parser()
    generation_args = parser.parse_args()  # we'll use this to load the correct hidden states + labels
    main(generation_args)