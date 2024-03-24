from siamese_network import SiameseNetwork
from ts_loader import TSLoader
import tensorflow as tf
import os
import pandas as pd
import numpy as np

dataset_path = 'ts_4'
tensorboard_log_path = './logs/testing'
general_output_file_path = './logs/testing_txt/'
model_name = 'siamese_net_ts_963599'
weights = 'models/siamese_net_ts_963599.h5'

def get_similarity_matrix():
    use_augmentation = True
    learning_rate = 10e-4
    batch_size = 32
    # Learning Rate multipliers for each layer
    learning_rate_multipliers = {}
    learning_rate_multipliers['Conv1'] = 1
    learning_rate_multipliers['Conv2'] = 1
    learning_rate_multipliers['Conv3'] = 1
    learning_rate_multipliers['Conv4'] = 1
    learning_rate_multipliers['Dense1'] = 1
    # l2-regularization penalization for each layer
    l2_penalization = {}
    l2_penalization['Conv1'] = 1e-2
    l2_penalization['Conv2'] = 1e-2
    l2_penalization['Conv3'] = 1e-2
    l2_penalization['Conv4'] = 1e-2
    l2_penalization['Dense1'] = 1e-4
    # Path where the logs will be saved
    if (not os.path.exists(tensorboard_log_path)):
        os.makedirs(tensorboard_log_path, exist_ok=True)
    siamese_network = SiameseNetwork(
        dataset_path=dataset_path,
        learning_rate=learning_rate,
        batch_size=batch_size, use_augmentation=use_augmentation,
        learning_rate_multipliers=learning_rate_multipliers,
        l2_regularization_penalization=l2_penalization,
        tensorboard_log_path=tensorboard_log_path
    )

    siamese_network.model.load_weights(weights)
    siamese_network.ts_loader.split_train_datasets()
    siamese_network.ts_loader.get_all_classes_test(siamese_network.model, 10, general_output_file_path)
    # evaluation_accuracy = siamese_network.ts_loader.get_all_classes_test(siamese_network.model, 10, general_output_file_path)
    # general_output_file = open(general_output_file_path + "general.txt", 'a')
    # general_output_file.write('\nFinal Evaluation Accuracy = ' + str(evaluation_accuracy))
    # print('\nFinal Evaluation Accuracy = ' + str(evaluation_accuracy))
    # general_output_file.flush()  # Force write to disk


def get_traffic_sign_dataset():
    validation_path = os.path.join(dataset_path, 'validation')
    dictionary = {}
    for trafficSign in os.listdir(validation_path):
        ts_path = os.path.join(validation_path, trafficSign)
        dictionary[trafficSign] = os.listdir(ts_path)
    return list(dictionary.keys())

def get_top_n_classes(row, num_top):
    # Create tuples of (similarity, class_index)
    similarity_pairs = list(row.items())

    # Sort by similarity in descending order
    similarity_pairs.sort(key=lambda item: item[1], reverse=True)

    # Extract top 10 class indices
    top_10_indices = [pair[0] for pair in similarity_pairs[:num_top]]

    return top_10_indices


def get_top_similarities(similarity_matrix, num_top=10):
    top_similarities_table = {}
    for index, row in similarity_matrix.iterrows():
        top_n_classes = get_top_n_classes(row, num_top)
        print(f"Top 10 classes for row {index}: {top_n_classes}")

        top_similarities_table[index] = top_n_classes
    return top_similarities_table


def create_html_table(top_similarities_table):
    html_content = """
    <html>
    <head><title>Similarity Table</title></head>
    <body>
    <table>
    """

    # Header row
    html_content += "<tr><th>Comparing Class</th>"
    for i in range(1, 11):
        html_content += f"<th>Top {i}</th>"
    html_content += "</tr>"

    # Table rows
    for row in top_similarities_table:
        html_content += "<tr>"
        for item in top_similarities_table[row]:
            if item.startswith(dataset_path):  # Assuming 'images/' denotes an image path
                html_content += f"<td><img src='{item}' width='128'></td>"  # Adjust width as needed
            else:
                html_content += f"<td>{item}</td>"
        html_content += "</tr>"

    html_content += """
    </table>
    </body>
    </html>
    """

    return html_content


def get_top_classes():
    ts_loader = TSLoader(
        dataset_path=dataset_path, use_augmentation=False, batch_size=0)
    trafficSigns = get_traffic_sign_dataset()
    trafficSigns.sort()
    similarity_matrix = pd.read_csv('similarity_matrix_ts_9.csv', index_col=0)
    top_similarities_table = get_top_similarities(similarity_matrix)
    for row in top_similarities_table:
        for otherClass in top_similarities_table[row]:  # 1 to 10 for top similarities
            image_path = ts_loader.get_random_image_from_class(otherClass, 1, 'validation')
            # Replace class name with image path
            top_similarities_table[row][top_similarities_table[row].index(otherClass)] = image_path
    html_content = create_html_table(top_similarities_table)
    # Save the HTML
    with open('similarity_table_ts_9.html', 'w') as f:
        f.write(html_content)

    print('HTML table saved as similarity_table.html')


if __name__ == "__main__":
    # Get list of available GPU devices
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Target the 1st available GPU (index 1)
            tf.config.set_visible_devices(gpus[1], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    # get_similarity_matrix()
    get_top_classes()
