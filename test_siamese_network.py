from siamese_network import SiameseNetwork
from ts_loader import TSLoader
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = 'ts_5'
sobel_path = 'ts_5_sobel'
cropped_images_path = 'cropped_images'
tensorboard_log_path = './logs/testing_cropped_images'
general_output_file_path = './logs/testing_cropped_images/'
weights = 'models/siamese_net_ts_5_gray/siamese_net_ts_5_gray47999.h5'
matrix_path = 'siamese_net_ts_5_gray_test_results'
html_path = 'ts_5_frame_sample_rgb'
cross_class_compare = False


def get_similarity_matrix():
    use_augmentation = False
    learning_rate = 10e-4
    batch_size = 32
    grayscale = False
    gpu = 0
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
        tensorboard_log_path=tensorboard_log_path,
        grayscale=grayscale,
        gpu=gpu
    )

    siamese_network.model.load_weights(weights)
    siamese_network.ts_loader.split_train_datasets()
    siamese_network.ts_loader.get_cropped_images_test(siamese_network.model, cropped_images_path, matrix_path)
    # evaluation_accuracy = siamese_network.ts_loader.get_all_classes_test(siamese_network.model, 10, general_output_file_path, matrix_path)
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
    """Extracts top n class indices and their similarity values from a row in the similarity matrix.

    Args:
        row (pandas.Series): A row from the similarity matrix.
        num_top (int): Number of top similar classes to extract.

    Returns:
        list: A list of tuples containing (class_index, similarity_value).
    """

    # Create tuples of (similarity, class_index)
    similarity_pairs = list(row.items())

    # Sort by similarity in descending order
    similarity_pairs.sort(key=lambda item: item[1], reverse=True)

    # Extract top n class indices and similarity values
    top_n_classes = similarity_pairs[:num_top]
    return top_n_classes


def get_top_similarities(similarity_matrix, num_top=10):
    """Creates a dictionary containing top n similar classes for each row in the similarity matrix.

    Args:
        similarity_matrix (pandas.DataFrame): The similarity matrix.
        num_top (int, optional): Number of top similar classes to extract for each row. Defaults to 10.

    Returns:
        dict: A dictionary where keys are row indices and values are lists of tuples containing (class_index, similarity_value).
    """

    top_similarities_table = {}
    for index, row in similarity_matrix.iterrows():
        top_n_classes = get_top_n_classes(row, num_top)

        top_similarities_table[index] = top_n_classes
    return top_similarities_table


def create_html_table(top_similarities_table):
    """Generates HTML content for a table displaying top similar classes with images and similarity values.

    Args:
        top_similarities_table (dict): A dictionary containing top n similar classes for each row in the similarity matrix.

    Returns:
        str: The HTML content for the table.
    """

    html_content = """
    <html>
    <head><title>Similarity Table</title></head>
    <body>
    <table>
    """

    # Header row
    html_content += "<tr><th>Comparing Class</th>"
    html_content += "<th>Augmented Version</th>"
    for i in range(1, 11):
        html_content += f"<th>Top {i}</th>"
    html_content += "</tr>"

    # Table rows
    for row, top_n_classes in top_similarities_table.items():
        html_content += "<tr>"
        # Display the name of the image being compared (assuming it's the row index)
        # augmented_picture = row.replace(cropped_images_path, f'{cropped_images_path}_sobel')
        html_content += f"<td><img src='{row}', width=128'><br></td>"
        # augmented_picture = row.replace(dataset_path, sobel_path)
        augmented_picture = row.replace(cropped_images_path, f"{cropped_images_path}_sobel")
        html_content += f"<td><img src='{augmented_picture}' width='128'><br></td>"

        for class_index, similarity_value in top_n_classes:
            if class_index.startswith(dataset_path) or class_index.startswith(
                    cropped_images_path):  # Assuming 'images/' denotes an image path
                html_content += f"<td><img src='{class_index}' width='128'><br>{similarity_value:.4f}</td>"  # Adjust width as needed
            else:
                html_content += f"<td>{similarity_value:.4f}</td><td>{class_index}</td>"
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
    similarity_matrix = pd.read_csv(f'{matrix_path}.csv', index_col=0)
    top_similarities_table = get_top_similarities(similarity_matrix)
    if cross_class_compare:
        top_similarities_table_new = {}
    for row in top_similarities_table:
        for otherClass in top_similarities_table[row]:  # 1 to 10 for top similarities
            image_path = ts_loader.get_random_image_from_class(otherClass[0], 1, 'validation')
            # Replace class name with image path
            top_similarities_table[row][top_similarities_table[row].index(otherClass)] = [image_path, otherClass[1]]
        if cross_class_compare:
            new_key = ts_loader.get_random_image_from_class(str(row), 1, 'validation')
            similarities = top_similarities_table[row]
            top_similarities_table_new[new_key] = similarities
    if cross_class_compare:
        html_content = create_html_table(top_similarities_table_new)
    else:
        html_content = create_html_table(top_similarities_table)
    # Save the HTML
    with open(f'{html_path}.html', 'w') as f:
        f.write(html_content)


def get_metrics_testing():
    use_augmentation = False
    learning_rate = 10e-4
    batch_size = 32
    grayscale = True
    gpu = 1
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
        tensorboard_log_path=tensorboard_log_path,
        grayscale=grayscale,
        gpu=gpu
    )

    siamese_network.model.load_weights(weights)
    siamese_network.ts_loader.split_train_datasets()
    siamese_network.ts_loader.one_shot_test_additional(siamese_network.model, 5, is_validation=True,
                                                       output_dir=general_output_file_path, csv_name=matrix_path)


def get_similarity_matrix_visualization():
    # Load your similarity matrix
    similarity_matrix = pd.read_csv(f'{matrix_path}.csv', index_col=0)

    # Get the maximum value in the matrix
    vmax = similarity_matrix.values.max()

    # Create the heatmap using a single color sequence
    plt.figure(figsize=(15, 12))
    sns.heatmap(similarity_matrix, annot=True, cmap='Blues', vmax=vmax)  # YlGnBu is a single color heatmap
    plt.title('Similarity Matrix')
    plt.show()


if __name__ == "__main__":
    #     Get list of available GPU devices
    #     gpus = tf.config.list_physical_devices('GPU')

    #     if gpus:
    #         try:
    #             # Target the 1st available GPU (index 1)
    #             tf.config.set_visible_devices(gpus[0], 'GPU')
    #             logical_gpus = tf.config.list_logical_devices('GPU')
    #             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #         except RuntimeError as e:
    #             print(e)
    # get_similarity_matrix()
    # get_top_classes()
    # get_similarity_matrix_visualization()
    get_metrics_testing()
