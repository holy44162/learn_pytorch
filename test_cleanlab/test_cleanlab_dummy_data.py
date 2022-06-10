import numpy as np
import cleanlab
from cleanlab.classification import CleanLearning
from cleanlab.benchmarking import noise_generation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from numpy.random import multivariate_normal
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


SEED = 0

def make_data(
    means=[[3, 2], [7, 7], [0, 8], [0, 10]],
    covs=[
        [[5, -1.5], [-1.5, 1]],
        [[1, 0.5], [0.5, 4]],
        [[5, 1], [1, 5]],
        [[3, 1], [1, 1]],
    ],
    sizes=[100, 50, 50, 50],
    avg_trace=0.8,
    seed=SEED,  # set to None for non-reproducible randomness
):
    np.random.seed(seed=SEED)

    K = len(means)  # number of classes
    data = []
    labels = []
    test_data = []
    test_labels = []

    for idx in range(K):
        data.append(
            np.random.multivariate_normal(
                mean=means[idx], cov=covs[idx], size=sizes[idx]
            )
        )
        test_data.append(
            np.random.multivariate_normal(
                mean=means[idx], cov=covs[idx], size=sizes[idx]
            )
        )
        labels.append(np.array([idx for i in range(sizes[idx])]))
        test_labels.append(np.array([idx for i in range(sizes[idx])]))
    X_train = np.vstack(data)
    y_train = np.hstack(labels)
    X_test = np.vstack(test_data)
    y_test = np.hstack(test_labels)

    # Compute p(y=k) the prior distribution over true labels.
    py_true = np.bincount(y_train) / float(len(y_train))

    noise_matrix_true = noise_generation.generate_noise_matrix_from_trace(
        K,
        trace=avg_trace * K,
        py=py_true,
        valid_noise_matrix=True,
        seed=SEED,
    )

    # Generate our noisy labels using the noise_marix.
    s = noise_generation.generate_noisy_labels(y_train, noise_matrix_true)
    s_test = noise_generation.generate_noisy_labels(y_test, noise_matrix_true)
    ps = np.bincount(s) / float(len(s))  # Prior distribution over noisy labels

    return {
        "data": X_train,
        "true_labels": y_train,  # You never get to see these perfect labels.
        "labels": s,  # Instead, you have these labels, which have some errors.
        "test_data": X_test,
        "test_labels": y_test,  # Perfect labels used for "true" measure of model's performance during deployment.
        "noisy_test_labels": s_test,  # With IID train/test split, you'd have these labels, which also have some errors.
        "ps": ps,
        "py_true": py_true,
        "noise_matrix_true": noise_matrix_true,
        "class_names": ["purple", "blue", "seafoam green", "yellow"],
    }


data_dict = make_data()
for key, val in data_dict.items():  # Map data_dict to variables in namespace
    exec(key + "=val")

# Display dataset visually using matplotlib
def plot_data(data, circles, title, alpha=1.0):
    plt.figure(figsize=(14, 5))
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=60)
    for i in circles:
        plt.plot(
            data[i][0],
            data[i][1],
            "o",
            markerfacecolor="none",
            markeredgecolor="red",
            markersize=14,
            markeredgewidth=2.5,
            alpha=alpha
        )
    _ = plt.title(title, fontsize=25)

if __name__ == "__main__":
    true_errors = np.where(true_labels != labels)[0]
    plot_data(data, circles=true_errors, title="A realistic, messy dataset with 4 classes", alpha=0.3)
    # plt.show()

    yourFavoriteModel = LogisticRegression(verbose=0, random_state=SEED)

    # CleanLearning: Machine Learning with cleaned data (given messy, real-world data)
    cl = cleanlab.classification.CleanLearning(yourFavoriteModel, seed=SEED)

    # Fit model to messy, real-world data, automatically training on cleaned data.
    _ = cl.fit(data, labels)

    # See the label quality for every example, which data has issues, and more.
    cl.get_label_issues().head()
    
    # For comparison, this is how you would have trained your model normally (without Cleanlab)
    yourFavoriteModel = LogisticRegression(verbose=0, random_state=SEED)
    yourFavoriteModel.fit(data, labels)
    print(f"Accuracy using yourFavoriteModel: {yourFavoriteModel.score(test_data, test_labels):.0%}")

    # But CleanLearning can do anything yourFavoriteModel can do, but enhanced.
    # For example, CleanLearning gives you predictions (just like yourFavoriteModel)
    # but the magic is that CleanLearning was trained as if your data did not have label errors.
    print(f"Accuracy using yourFavoriteModel (+ CleanLearning): {cl.score(test_data, test_labels):.0%}")

    # One line of code. Literally.
    issues = CleanLearning(yourFavoriteModel, seed=SEED).find_label_issues(data, labels)

    issues.head()

    lowest_quality_labels = issues["label_quality"].argsort()[:20]
    plot_data(data, circles=lowest_quality_labels, title="The 20 lowest label quality examples")
    # plt.show()

    # CleanLearning can train faster if issues are provided at fitting time.
    cl.fit(data, labels, label_issues=issues)

    cleanlab.dataset.find_overlapping_classes(
        labels=labels,
        confident_joint=cl.confident_joint,  # cleanlab uses the confident_joint internally to quantify label noise (see cleanlab.count.compute_confident_joint)
        class_names=class_names,
    )

    yourFavoriteModel1 = LogisticRegression(verbose=0, random_state=SEED)
    yourFavoriteModel1.fit(data, labels)
    print(f"[Original classes] Accuracy of yourFavoriteModel: {yourFavoriteModel1.score(test_data, test_labels):.0%}")

    merged_labels, merged_test_labels = np.array(labels), np.array(test_labels)

    # Merge classes: map all yellow-labeled examples to seafoam green
    merged_labels[merged_labels == 3] = 2
    merged_test_labels[merged_test_labels == 3] = 2

    # Re-run our comparison. Re-run your model on the newly labeled dataset.
    yourFavoriteModel2 = LogisticRegression(verbose=0, random_state=SEED)
    yourFavoriteModel2.fit(data, merged_labels)
    print(f"[Modified classes] Accuracy of yourFavoriteModel: {yourFavoriteModel2.score(test_data, merged_test_labels):.0%}")

    # Re-run CleanLearning as well.
    yourFavoriteModel3 = LogisticRegression(verbose=0, random_state=SEED)
    cl3 = cleanlab.classification.CleanLearning(yourFavoriteModel, seed=SEED)
    cl3.fit(data, merged_labels)
    print(f"[Modified classes] Accuracy of yourFavoriteModel (+ CleanLearning): {cl3.score(test_data, merged_test_labels):.0%}")

    # Fit your model on noisily labeled train data
    yourFavoriteModel = LogisticRegression(verbose=0, random_state=SEED)
    yourFavoriteModel.fit(data, labels)

    # Get predicted probabilities for test data (these are out-of-sample)
    my_test_pred_probs = yourFavoriteModel.predict_proba(test_data)
    my_test_preds = my_test_pred_probs.argmax(axis=1)  # predicted labels

    # Find label issues in the test data
    issues_test = CleanLearning(yourFavoriteModel, seed=SEED).find_label_issues(
        labels=noisy_test_labels, pred_probs=my_test_pred_probs)

    # You should inspect issues_test and fix issues to ensure high-quality test data labels.
    corrected_test_labels = test_labels  # Here we'll pretend you have done this perfectly :)

    # Fit more robust version of model on noisily labeled training data
    cl = CleanLearning(yourFavoriteModel, seed=SEED).fit(data, labels)
    cl_test_preds = cl.predict(test_data)

    print(f" Noisy Test Accuracy (on given test labels) using yourFavoriteModel: {accuracy_score(noisy_test_labels, my_test_preds):.0%}")
    print(f" Noisy Test Accuracy (on given test labels) using yourFavoriteModel (+ CleanLearning): {accuracy_score(noisy_test_labels, cl_test_preds):.0%}")
    print(f"Actual Test Accuracy (on corrected test labels) using yourFavoriteModel: {accuracy_score(corrected_test_labels, my_test_preds):.0%}")
    print(f"Actual Test Accuracy (on corrected test labels) using yourFavoriteModel (+ CleanLearning): {accuracy_score(corrected_test_labels, cl_test_preds):.0%}")

    # One line of code.
    health = cleanlab.dataset.overall_label_health_score(
        labels, confident_joint=cl.confident_joint
        # cleanlab uses the confident_joint internally to quantify label noise (see cleanlab.count.compute_confident_joint)
    )

    label_acc = sum(labels != true_labels) / len(labels)
    print(f"Percentage of label issues guessed by cleanlab {1 - health:.0%}")
    print(f"Percentage of (ground truth) label errors): {label_acc:.0%}")

    offset = (1 - label_acc) - health

    print(
        f"\nQuestion: cleanlab seems to be overestimating."
        f" How do we account for this {offset:.0%} difference?"
    )
    print(
        "Answer: Data points that fall in between two overlapping distributions are often "
        "impossible to label and are counted as issues."
    )

    pred_probs = cleanlab.count.estimate_cv_predicted_probabilities(
        data, labels, clf=yourFavoriteModel, seed=SEED
    )
    print(f"pred_probs is a {pred_probs.shape} matrix of predicted probabilites")

    (
        py, noise_matrix, inverse_noise_matrix, confident_joint
    ) = cleanlab.count.estimate_py_and_noise_matrices_from_probabilities(labels, pred_probs)

    # Note: you can also combine the above two lines of code into a single line of code like this
    (
        py, noise_matrix, inverse_noise_matrix, confident_joint, pred_probs
    ) = cleanlab.count.estimate_py_noise_matrices_and_cv_pred_proba(
        data, labels, clf=yourFavoriteModel, seed=SEED
    )

    # Get the joint distribution of noisy and true labels from the confident joint
    # This is the most powerful statistic in machine learning with noisy labels.
    joint = cleanlab.count.estimate_joint(
        labels, pred_probs, confident_joint=confident_joint
    )

    # Pretty print the joint distribution and noise matrix
    cleanlab.internal.util.print_joint_matrix(joint)
    cleanlab.internal.util.print_noise_matrix(noise_matrix)

    cl3 = cleanlab.classification.CleanLearning(yourFavoriteModel, seed=SEED)
    _ = cl3.fit(data, labels, noise_matrix=noise_matrix_true)  # CleanLearning with a prioiri known noise_matrix

    # Get out of sample predicted probabilities via cross-validation.
    # Here we demonstrate the use of sklearn cross_val_predict as another option to get cross-validated predicted probabilities
    cv_pred_probs = cross_val_predict(
        estimator=yourFavoriteModel, X=data, y=labels, cv=3, method="predict_proba"
    )

    # Find label issues
    label_issues_indices = cleanlab.filter.find_label_issues(
        labels=labels,
        pred_probs=cv_pred_probs,
        filter_by="both", # 5 available filter_by options
        return_indices_ranked_by="self_confidence",  # 3 available label quality scoring options for rank ordering
        rank_by_kwargs={
            "adjust_pred_probs": True  # adjust predicted probabilities (see docstring for more details)
        },
    )

    # Return dataset indices of examples with label issues
    label_issues_indices

    plot_data(data, circles=label_issues_indices[:20], title="Top 20 label issues found by cleanlab.filter.find_label_issues()")
    # plt.show()

    # 3 models in ensemble
    model1 = LogisticRegression(penalty="l2", verbose=0, random_state=SEED)
    model2 = RandomForestClassifier(max_depth=5, random_state=SEED)
    model3 = GradientBoostingClassifier(
        n_estimators=100, learning_rate=1.0, max_depth=3, random_state=SEED
    )

    # Get cross-validated predicted probabilities from each model
    cv_pred_probs_1 = cross_val_predict(
        estimator=model1, X=data, y=labels, cv=3, method="predict_proba"
    )
    cv_pred_probs_2 = cross_val_predict(
        estimator=model2, X=data, y=labels, cv=3, method="predict_proba"
    )
    cv_pred_probs_3 = cross_val_predict(
        estimator=model3, X=data, y=labels, cv=3, method="predict_proba"
    )

    # List of predicted probabilities from each model
    pred_probs_list = [cv_pred_probs_1, cv_pred_probs_2, cv_pred_probs_3]

    # Get ensemble label quality scores
    label_quality_scores_best = cleanlab.rank.get_label_quality_ensemble_scores(
        labels=labels, pred_probs_list=pred_probs_list, verbose=False
    )

    # Alternative approach: create single ensemble predictor and get its pred_probs
    cv_pred_probs_ensemble = (cv_pred_probs_1 + cv_pred_probs_2 + cv_pred_probs_3)/3  # uniform aggregation of predictions

    # Use this single set of pred_probs to find label issues
    label_quality_scores_better = cleanlab.rank.get_label_quality_scores(
        labels=labels, pred_probs=cv_pred_probs_ensemble
    )
