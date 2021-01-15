import pickle
import os
import matplotlib.pyplot as plt


def get_cmc_scores(path_to_eval_):

    pickle_in = open(path_to_eval_ + "model_eval.pickle", "rb")
    model_eval = pickle.load(pickle_in)
    cmc_scores = []

    for rank in range(99):

        cmc_sum = 0

        for predictions_and_label in model_eval["model_predictions"]:

            predictions = predictions_and_label["predictions"]
            label = predictions_and_label["label"]

            predictions_with_index = [(predictions[i], i) for i in range(len(predictions))]
            predictions_with_index.sort(key=lambda x: x[0], reverse=True)

            for j in range(rank + 1):
                if predictions_with_index[j][1] == label:
                    cmc_sum += 1
                    break

        cmc = cmc_sum/400
        #print(cmc)
        cmc_scores.append(cmc)

    return cmc_scores


if __name__ == "__main__":

    aug_or_no = ["no_aug", "aug"]
    model_base = ["vgg16", "InceptionResNetV2"]
    epochs_no_aug = ["30_epochs", "50_epochs"]
    epochs_aug = ["30_epochs", "50_epochs", "80_epochs"]
    colors = ["r", "g", "b"]

    for au in aug_or_no:
        for mb in model_base:
            if au == "aug":

                for color, ep in zip(colors, epochs_aug):
                    path_to_eval = os.getcwd() + "/results/" + au + "/" + mb + "/" + ep + "/"
                    cmc_scores = get_cmc_scores(path_to_eval)
                    plt.plot(cmc_scores, label=ep)

                plt_title = ("VGG16" if (mb == "vgg16") else "InceptionResNetV2") + " with " + ("data augmentation." if (au == "aug") else "no data augmentation.")
                plt.suptitle(plt_title)
                plt.xlabel('rank')
                plt.ylabel('cmc')
                plt.legend(loc="upper left")
                plt.savefig("./plots/" + mb + "_" + au + ".png")
                plt.show()


            else:
                for color, ep in zip(colors, epochs_no_aug):
                    path_to_eval = os.getcwd() + "/results/" + au + "/" + mb + "/" + ep + "/"
                    cmc_scores = get_cmc_scores(path_to_eval)
                    plt.plot(cmc_scores, label=ep)

                plt_title = ("VGG16" if (mb == "vgg16") else "InceptionResNetV2") + " with " + ("data augmentation." if (au == "aug") else "no data augmentation.")
                plt.suptitle(plt_title)
                plt.xlabel('rank')
                plt.ylabel('cmc')
                plt.legend(loc="upper left")
                plt.savefig("./plots/" + mb + "_" + au + ".png")
                plt.show()

