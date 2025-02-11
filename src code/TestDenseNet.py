import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout

import os
import sys
import random
import time
import argparse
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
import six
import sys
from tensorflow.keras.regularizers import l2

taxa_num = "128Taxa"
folder_image = r"D:\Bio\NeuralPredictModel\Result\26_12"
folder_image = os.path.join(folder_image, taxa_num)
os.makedirs(folder_image, exist_ok=True)


def plot(file_path, length):
    print(file_path)
    test = pd.read_csv(file_path)
    global f_name
    f_name = file_path
    model_list = {
        "JC+G": 0,
        "K2P+G": 1,
        "F81+G": 2,
        "HKY+G": 3,
        "TN93+G": 4,
        "GTR+G": 5,
    }
    lb_list = ["JC+G", "K2P+G", "F81+G", "HKY+G", "TN93+G", "GTR+G"]

    y_test = test['AlnID']
    y_pred = test['Predict']
    results_count = [[0]*6 for _ in range(6)]
    count_label = [0]*6
    sum_truly_predicted = 0

    print("len(y_test): %d" % len(y_test))
    for idx in range(len(y_test)):
        i = int(y_test[idx])                      # nhãn thật
        j = int(model_list[y_pred[idx]])          # nhãn dự đoán
        if i == j:
            sum_truly_predicted += 1
        results_count[i][j] += 1
        count_label[i] += 1

    print("Sum of truly predicted: %d" % sum_truly_predicted)
    print(np.sum(results_count))
    
    # Tính toán Precision, Recall và F1-score
    results_count = np.array(results_count)
    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(6):
        tp = results_count[c, c]
        fn = sum(results_count[c, :]) - tp
        fp = sum(results_count[:, c]) - tp

        recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_c = 2 * precision_c * recall_c / (precision_c + recall_c) if (precision_c + recall_c) > 0 else 0

        recall_list.append(recall_c)
        precision_list.append(precision_c)
        f1_list.append(f1_c)

    # Tính trung bình (macro average)
    macro_recall = np.mean(recall_list)
    macro_precision = np.mean(precision_list)
    macro_f1 = np.mean(f1_list)

    print("------ METRICS ------")
    for i, label in enumerate(lb_list):
        print(f"Class {label}:")
        print(f"   Precision = {precision_list[i]:.3f}")
        print(f"   Recall    = {recall_list[i]:.3f}")
        print(f"   F1-score  = {f1_list[i]:.3f}")
    print("------ MACRO AVERAGE ------")
    print(f"Precision (macro) = {macro_precision:.3f}")
    print(f"Recall    (macro) = {macro_recall:.3f}")
    print(f"F1-score  (macro) = {macro_f1:.3f}")
    print("----------------------------------------------\n")

    # Tính tỷ lệ phần trăm để vẽ confusion matrix
    results_percentage = results_count.astype(float)
    for i in range(6):
        row_sum = sum(results_count[i])
        if row_sum > 0:
            results_percentage[i] = (results_percentage[i] / row_sum) * 100

    # In ma trận phần trăm
    thresh = 0.0
    for i in range(6):
        line = "["
        for j in range(6):
            val_str = f"{results_percentage[i][j]:.2f}"
            line += f"{val_str} ,"
            if results_percentage[i][j] > thresh:
                thresh = results_percentage[i][j]
        line += "],"
        print(line)
    thresh = thresh / 2  # Ngưỡng để xác định màu chữ khi vẽ

    # Vẽ Heatmap
    fig, ax = plt.subplots(figsize=(10, 8)) 
    im = ax.imshow(results_percentage, cmap='viridis')  # Sử dụng colormap gốc hoặc bạn có thể thay đổi nếu cần
    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(lb_list)
    ax.set_yticklabels(lb_list)
    ax.set_facecolor('white')

    # Xoay nhãn cột x cho gọn
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Thêm con số trong từng ô matrix
    for i in range(6):
        for j in range(6):
            text = ax.text(j, i, f"{results_percentage[i][j]:.2f}",
                           ha="center", va="center",
                           color="black" if results_percentage[i][j] > thresh else "white",
                           fontsize=12)

    # In ra giá trị đường chéo (để tính accuracy trung bình cũ)
    diag_vals = [results_percentage[i][i] for i in range(6)]
    diag_vals_float = [float(v) for v in diag_vals if float(v) > 0]
    avg_acc = np.mean(diag_vals_float) if len(diag_vals_float) > 0 else 0
    print("Đường chéo (theo % hàng):", diag_vals)
    print("Average accuracy (trên mỗi class): %.2f" % avg_acc)

    plt.title(f"{length}bp - {taxa_num}")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # Xoá file csv sau khi xong (nếu bạn vẫn muốn)
    os.remove(file_path)

    # Thêm thông tin Average accuracy, Recall và F1-score vào hình
    fig.subplots_adjust(bottom=0.3)
    metrics_text = (
        f"Average Accuracy: {avg_acc:.2f}%\n"
        f"Macro Precision: {macro_precision:.3f}\n"
        f"Macro Recall: {macro_recall:.3f}\n"
        f"Macro F1-score: {macro_f1:.3f}"
    )
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    fig.subplots_adjust(bottom=0.25)
    # Lưu hình
    os.makedirs(folder_image, exist_ok=True)  # Đảm bảo thư mục tồn tại
    plt.savefig(os.path.join(folder_image, f"{length}{taxa_num}.jpg"), dpi=300)
    plt.show()

def create_densenet_with_dropout(input_shape, num_classes):
    model = Sequential()

    model.add(Dense(128,input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(64, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(32, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(16, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(num_classes, activation='softmax'))
    return model

def predict_aln(h5model, test_csv, length):
    print('Load model...')
    #model = load_model('%s' % h5model)
    test = pd.read_csv("%s" % test_csv, on_bad_lines='skip')
    # Assume the last column is the target variable
    X_new = test.iloc[:, 1:].values
    y_new = test.iloc[:, 0].values

    # One-hot encode the target variable if applicable
    encoder = OneHotEncoder(sparse_output=False)
    y_new = encoder.fit_transform(y_new.reshape(-1, 1))
    
    input_shape = (X_new.shape[1],)
    num_classes = 6
    model = create_densenet_with_dropout(input_shape,num_classes)
    #model = create_densenet(input_shape,num_classes)
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    model.load_weights(h5model)

    # Make predictions on the new data
    predictions = model.predict(X_new)
    #print(predictions)

    # If you want to see the predicted classes
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_new, axis=1)

    #print("Predicted classes:", predicted_classes)
    #print("True classes:", true_classes)


    y_test = test['label']
    # select the indix with the maximum probability
    results = np.argmax(predictions, axis=1)
    #print(results)
    model_list = {
    # 0: "JC",
    # 1: "K2P",
    # 2: "F81",
    # 3: "HKY",
    # 4: "TN93",
    # 5: "GTR",
    # 6: "JC+G",
    # 7: "K2P+G",
    # 8: "F81+G",
    # 9: "HKY+G",
    # 10: "TN93+G",
    # 11: "GTR+G"
    0: "JC+G",
    1: "K2P+G",
    2: "F81+G",
    3: "HKY+G",
    4: "TN93+G",
    5: "GTR+G",
}


    results = pd.Series(results, name="Label")
    print(results)
    list_results = results.to_numpy()
    # print(list_results)
    list_name = ["*"] * len(list_results)
    for id in range(len(list_results)):
        list_name[id] = model_list[list_results[id]]
    # print(list_name)
    name_aln = pd.Series(y_test)
    list_out_label = pd.Series(list_name, name="Predict")
    submission = pd.concat(
        [pd.Series(name_aln, name="AlnID"), list_out_label], axis=1)
    
    submission.to_csv("%s_%s.results.csv" % ( os.path.basename(h5model), os.path.basename(test_csv)), index=False)
    plot("%s_%s.results.csv" % (h5model,test_csv), length)

# call main function
def run(args):
    print(args)
    predict_aln(args.built_model, args.test_dataset, args.length)


start_time = time.time()
if __name__ == '__main__':
    print("STARTING CNN MODEL SELECTOR...")

    # Initialize start time for runtime calculation
    start_time = time.time()

    # Initialize argument parser
    parser = argparse.ArgumentParser(description='DNA Model Selector')

    # Add arguments with appropriate descriptions
    parser.add_argument('-test_dataset', type=str, required=True, 
                        help="Path to the test dataset file")
    parser.add_argument('-built_model', type=str, required=True, 
                        help="Path to the pre-built model file")
    parser.add_argument('-length', type=int, required=False, default=1000, 
                        help="Length of the input sequences (default: 100)")

    # Parse arguments from command line
    args = parser.parse_args(sys.argv[1:])

    # Call the run function with the parsed arguments
    run(args)

    # Calculate and print the total runtime
    end_time = time.time()
    print("Run time: %f seconds" % (end_time - start_time))