import os
import random
import numpy as np
import math
from PIL import Image
import pandas as pd
import pickle

from image_augmentor import ImageAugmentor


class TSLoader:
    """Class that loads and prepares the TS dataset

    This Class was constructed to read the Omniglot alphabets, separate the
    training, validation and evaluation test. It also provides function for
    geting one-shot task batches.

    Attributes:
        dataset_path: path of Omniglot Dataset
        train_dictionary: dictionary of the files of the train set (background set).
            This dictionary is used to load the batch for training and validation.
        evaluation_dictionary: dictionary of the evaluation set.
        image_width: self explanatory
        image_height: self explanatory
        batch_size: size of the batch to be used in training
        use_augmentation: boolean that allows us to select if data augmentation is
            used or not
        image_augmentor: instance of class ImageAugmentor that augments the images
            with the affine transformations referred in the paper

    """

    def __init__(self, dataset_path, use_augmentation, batch_size, grayscale=False):
        """Inits OmniglotLoader with the provided values for the attributes.

        It also creates an Image Augmentor object and loads the train set and
        evaluation set into dictionaries for future batch loading.

        Arguments:
            dataset_path: path of Omniglot dataset
            use_augmentation: boolean that allows us to select if data augmentation
                is used or not
            batch_size: size of the batch to be used in training
        """

        self.dataset_path = dataset_path
        self.train_dictionary = {}
        self.evaluation_dictionary = {}
        self.image_width = 128
        self.image_height = 128
        self.dimensions = 3
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self._trainTS = []
        self._validationTS = []
        self._evaluationTS = []
        self._current_validation_ts_index = 0
        self._current_evaluation_ts_index = 0
        self.grayscale = grayscale
        self.load_dataset()
        self.scalers = [pickle.load(open(f'scaler_{i}.pkl', 'rb')) for i in range(4)]

        if (self.use_augmentation):
            self.image_augmentor = self.createAugmentor()
        else:
            self.use_augmentation = []

    def load_dataset(self):
        """Loads the traffic signs into dictionary

        Loads the TS dataset and stores the available images for each
        TS for each of the train and evaluation set.

        """

        train_path = os.path.join(self.dataset_path, 'train')
        validation_path = os.path.join(self.dataset_path, 'validation')

        # First let's take care of the train TS
        for trafficSign in os.listdir(train_path):
            ts_path = os.path.join(train_path, trafficSign)
            self.train_dictionary[trafficSign] = os.listdir(ts_path)

        # Now it's time for the validation TS
        for trafficSign in os.listdir(validation_path):
            ts_path = os.path.join(validation_path, trafficSign)
            self.evaluation_dictionary[trafficSign] = os.listdir(ts_path)

    def createAugmentor(self):
        """ Creates ImageAugmentor object with the parameters for image augmentation

        Rotation range was set in -15 to 15 degrees
        Shear Range was set in between -0.3 and 0.3 radians
        Zoom range between 0.8 and 2
        Shift range was set in +/- 5 pixels

        Returns:
            ImageAugmentor object

        """
        rotation_range = [-15, 15]
        shear_range = [-0.3 * 180 / math.pi, 0.3 * 180 / math.pi]
        zoom_range = [0.8, 2]
        shift_range = [5, 5]

        return ImageAugmentor(1, shear_range, rotation_range, shift_range, zoom_range)

    def split_train_datasets(self):
        """ Splits the train set in train and validation

        Divide the 30 train alphabets in train and validation with
        # a 80% - 20% split (24 vs 6 alphabets)

        """

        # The remaining TS are saved for validation
        self._trainTS = list(self.train_dictionary.keys())
        self._validationTS = list(self.evaluation_dictionary.keys())
        self._evaluationTS = list(self.evaluation_dictionary.keys())

    def _convert_path_list_to_images_and_labels(self, path_list, is_one_shot_task, true_index=None):
        """ Loads the images and its correspondent labels from the path

        Take the list with the path from the current batch, read the images and
        return the pairs of images and the labels
        If the batch is from train or validation the labels are alternately 1's and
        0's. If it is a evaluation set only the first pair has label 1

        Arguments:
            path_list: list of images to be loaded in this batch
            is_one_shot_task: flag sinalizing if the batch is for one-shot task or if
                it is for training

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes

        """
        number_of_pairs = len(path_list)
        labels = np.zeros((number_of_pairs, 1))

        if self.grayscale:
            pairs_of_images = [np.zeros((number_of_pairs,
                                         self.image_height, self.image_height)) for _ in range(2)]
        else:
            pairs_of_images = [np.zeros((number_of_pairs,
                                         self.image_height, self.image_height, self.dimensions)) for _ in range(2)]

        for pair in range(number_of_pairs):
            if self.grayscale:
                image = Image.open(os.path.abspath(path_list[pair][0])).convert('L')  # Convert to grayscale ('L' mode)
                image = np.asarray(image).astype(np.float64)

                # image = (image - image.min()) / (image.max() - image.min())  # Scale to 0-1
                # image = image / image.std() - image.mean()
                pairs_of_images[0][pair] = image

                image = Image.open(os.path.abspath(path_list[pair][1])).convert('L')  # Convert to grayscale ('L' mode)
                image = np.asarray(image).astype(np.float64)
                # image = (image - image.min()) / (image.max() - image.min())  # Scale to 0-1
                # image = image / image.std() - image.mean()
                pairs_of_images[1][pair] = image
            else:
                image = Image.open(os.path.abspath(path_list[pair][0])).convert('RGB')
                image = np.asarray(image).astype(np.float64)
                image_gayscale = Image.open(os.path.abspath(
                    path_list[pair][0].replace(self.dataset_path, "ts_4_norm"))).convert('L')
                image_gayscale = np.asarray(image_gayscale).astype(np.float64)
                image_gayscale = image_gayscale[..., np.newaxis]  # (128, 128, 1)
                image = np.concatenate((image, image_gayscale), axis=2)  # (128, 128, 4)
                for channel in range(4):
                    image_reshaped = image[..., channel].reshape(-1, 1)  # Reshape for a single channel
                    scaler = self.scalers[channel]
                    transformed_array = scaler.transform(image_reshaped).reshape(128, 128)
                    image[..., channel] = transformed_array
                pairs_of_images[0][pair, :, :, :] = image

                image = Image.open(os.path.abspath(path_list[pair][1])).convert('RGB')
                image = np.asarray(image).astype(np.float64)
                image_gayscale = Image.open(
                    os.path.abspath(
                        path_list[pair][1].replace(self.dataset_path, "ts_4_norm"))).convert(
                    'L')
                image_gayscale = np.asarray(image_gayscale).astype(np.float64)
                image_gayscale = image_gayscale[..., np.newaxis]  # (128, 128, 1)
                image = np.concatenate((image, image_gayscale), axis=2)  # (128, 128, 4)
                for channel in range(4):
                    image_reshaped = image[..., channel].reshape(-1, 1)  # Reshape for a single channel
                    scaler = self.scalers[channel]
                    transformed_array = scaler.transform(image_reshaped).reshape(128, 128)
                    image[..., channel] = transformed_array
                pairs_of_images[1][pair, :, :, :] = image
                # OLD COLOUR
                # image = Image.open(os.path.abspath(path_list[pair][0]))
                # image = np.asarray(image).astype(np.float64)
                # # image = (image - image.min()) / (image.max() - image.min())  # Scale to 0-1
                # # image = image / image.std() - image.mean()
                # pairs_of_images[0][pair, :, :, :] = image
                #
                # image = Image.open(os.path.abspath(path_list[pair][1]))
                # image = np.asarray(image).astype(np.float64)
                # # image = (image - image.min()) / (image.max() - image.min())  # Scale to 0-1
                # # image = image / image.std() - image.mean()
                # pairs_of_images[1][pair, :, :, :] = image

            if not is_one_shot_task:
                if (pair < number_of_pairs / 2):
                    labels[pair] = 1
                else:
                    labels[pair] = 0
            if true_index is not None:
                if true_index != -1:
                    if pair == true_index:
                        labels[pair] = 1
                    else:
                        labels[pair] = 0
                if true_index == -1:
                    labels[pair] = 0
            else:
                if pair == 0:
                    labels[pair] = 1
                else:
                    labels[pair] = 0

        # if not is_one_shot_task:
        # random_permutation = np.random.permutation(number_of_pairs)
        # labels = labels[random_permutation]
        # pairs_of_images[0][:] = pairs_of_images[0][random_permutation]
        # pairs_of_images[1][:] = pairs_of_images[1][random_permutation]
        # pairs_of_images[0][:, :, :,
        # :] = pairs_of_images[0][random_permutation, :, :, :]
        # pairs_of_images[1][:, :, :,
        # :] = pairs_of_images[1][random_permutation, :, :, :]

        return pairs_of_images, labels

    def convert_path_list_to_images_and_labels_rgb_singlearray(self, path_list):
        """ Loads the images and its correspondent labels from the path

        Take the list with the path from the current batch, read the images and
        return the pairs of images and the labels
        If the batch is from train or validation the labels are alternately 1's and
        0's. If it is a evaluation set only the first pair has label 1

        Arguments:
            path_list: list of images to be loaded in this batch
            is_one_shot_task: flag sinalizing if the batch is for one-shot task or if
                it is for training

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes
        """

        number_of_images = len(path_list)
        images = [np.zeros((number_of_images,
                            self.image_height, self.image_height, self.dimensions)) for _ in range(1)][0]
        for iteration, image in enumerate(path_list):
            image_array = Image.open(image).convert('RGB')
            image_array = np.asarray(image_array).astype(np.float64)
            images[iteration, :, :, :] = image_array

        return images

    def get_random_image_from_class(self, currentTS, numberOfImages, dataset):
        imagesOfCurrentTS = os.listdir(os.path.join(
            self.dataset_path, dataset, currentTS))
        if len(imagesOfCurrentTS) > 1:
            imagePaths = []
            randomImages = random.sample(imagesOfCurrentTS, numberOfImages)
            if numberOfImages > 1:
                for i in range(0, numberOfImages):
                    image_path = os.path.join(
                        self.dataset_path, dataset, currentTS, randomImages[i])
                    imagePaths.append(image_path)
                return imagePaths
            else:
                return os.path.join(
                    self.dataset_path, dataset, currentTS, randomImages[0])
        else:
            if numberOfImages == 1:
                return os.path.join(
                    self.dataset_path, dataset, currentTS, imagesOfCurrentTS[0])
            else:
                imagePaths = []
                image_path = os.path.join(
                    self.dataset_path, dataset, currentTS, imagesOfCurrentTS[0])
                imagePaths.append(image_path)
                image_path = os.path.join(
                    self.dataset_path, 'train', currentTS, 'AUGMUMENT')
                imagePaths.append(image_path)
                return imagePaths

    def get_train_batch(self):
        """ Loads and returns a batch of train images

        Get a batch of pairs from the training set. Each batch will contain
        images from a single alphabet. I decided to select one single example
        from random n/2 characters in each alphabet. If the current alphabet
        has lower number of characters than n/2 (some of them have 14) we
        sample repeated classed for that batch per character in the alphabet
        to pair with a different categories. In the other half of the batch
        I selected pairs of same characters. In resume we will have a batch
        size of n, with n/2 pairs of different classes and n/2 pairs of the same
        class. Each batch will only contains samples from one single alphabet.

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes

        """
        randomClasses = random.sample(range(0, len(self._trainTS)), int(self.batch_size / 2))
        # currentTS = self._trainTS[self._current_train_alphabet_index]
        # numberOfTS = len(currentTS)

        batch_images_path = []

        # At first I need to take 16 random classes. For each of them,
        # if there are more than 1 image it means I can make a pair
        # without any augmentation. If there is only 1 image I need to
        # do the augmentation to get a pair.
        for index in randomClasses:
            batch_images_path.append(self.get_random_image_from_class(self._trainTS[index],
                                                                      2, 'train'))

        # Now let's take care of the pair of images of different traffic signs
        for index in randomClasses:
            currentTS = self._trainTS[index]
            differentTS = []
            differentTS.append(self.get_random_image_from_class(currentTS, 1, 'train'))
            while 1:
                randomOtherTS = self._trainTS[random.randint(0, len(self._trainTS) - 1)]
                if randomOtherTS != currentTS:
                    differentTS.append(self.get_random_image_from_class(randomOtherTS, 1, 'train'))
                    break
            batch_images_path.append(differentTS)

        images, labels = self._convert_path_list_to_images_and_labels(
            batch_images_path, is_one_shot_task=False)

        # Get random transforms if augmentation is on
        if self.use_augmentation and len(images):
            images = self.image_augmentor.get_random_transform(images)

        return images, labels

    def get_one_shot_batch(self, support_set_size, is_validation):
        """ Loads and returns a batch for one-shot task images

        Gets a one-shot batch for evaluation or validation set, it consists in a
        single image that will be compared with a support set of images. It returns
        the pair of images to be compared by the model and it's labels (the first
        pair is always 1) and the remaining ones are 0's

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes

        """

        # Set some variables that will be different for validation and evaluation sets
        if is_validation:
            trafficSigns = self._validationTS
            image_folder_name = 'validation'
            dictionary = self.train_dictionary
        else:
            trafficSigns = self._evaluationTS
            image_folder_name = 'validation'
            dictionary = self.evaluation_dictionary

        currentTSIndex = random.randint(0, len(trafficSigns) - 1)
        currentTS = trafficSigns[currentTSIndex]

        first_test_images = self.get_random_image_from_class(currentTS, 2, image_folder_name)

        batchImagePath = [first_test_images]

        OtherTSIndexesfromSet = random.sample(range(0, len(trafficSigns)), int(self.batch_size / 2))

        for otherTSIndex in OtherTSIndexesfromSet:
            otherTSClass = trafficSigns[otherTSIndex]
            otherTSImage = self.get_random_image_from_class(otherTSClass, 1, image_folder_name)
            batchImagePath.append([first_test_images[0], otherTSImage])

        images, labels = self._convert_path_list_to_images_and_labels(
            batchImagePath, is_one_shot_task=True)

        return images, labels

    def get_full_one_shot_batch(self, ts_class, traffic_signs, image_folder_name):
        dictionary = self.evaluation_dictionary
        first_test_image = self.get_random_image_from_class(ts_class, 1, image_folder_name)

        batchImagePath = []

        for otherTSClass in traffic_signs:
            otherTSImage = self.get_random_image_from_class(otherTSClass, 1, image_folder_name)
            batchImagePath.append([first_test_image, otherTSImage])

        images, labels = self._convert_path_list_to_images_and_labels(
            batchImagePath, True, true_index=traffic_signs.index(ts_class))

        return images, labels

    def get_full_one_shot_batch_no_labels(self, imageToTest, traffic_signs, image_folder_name):
        dictionary = self.evaluation_dictionary
        batchImagePath = []

        for otherTSClass in traffic_signs:
            otherTSImage = self.get_random_image_from_class(otherTSClass, 1, image_folder_name)
            batchImagePath.append([imageToTest, otherTSImage])

        images, labels = self._convert_path_list_to_images_and_labels(batchImagePath, True)

        return images, labels

    def one_shot_test_old(self, model, support_set_size, number_of_tasks_per_ts,
                          is_validation, output_dir):
        """ Prepare one-shot task and evaluate its performance

        Make one shot task in validation and evaluation sets
        if support_set_size = -1 we perform a N-Way one-shot task with
        N being the total of characters in the alphabet

        Returns:
            mean_accuracy: mean accuracy for the one-shot task
        """

        if is_validation:
            trafficSigns = self._validationTS
            print('\nMaking One Shot Task on validation traffic signs:')
            subfolder = 'validation'
            prefix = 'validation_'
        else:
            trafficSigns = self._evaluationTS
            print('\nMaking One Shot Task on evaluation traffic signs:')
            subfolder = 'evaluation'
            prefix = 'evaluation_'

        mean_global_accuracy = 0

        output_dir = os.path.join(output_dir, subfolder)
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        existing_files = [f for f in os.listdir(output_dir) if f.startswith(prefix)]
        if existing_files:
            last_num = max([int(f[len(prefix):-4]) for f in existing_files])
            file_num = last_num + 1
        else:
            file_num = 1
        output_file = os.path.join(output_dir, prefix + str(file_num) + '.txt')

        with open(output_file, 'a') as f:
            for trafficSign in trafficSigns:
                mean_ts_accuracy = 0
                for _ in range(number_of_tasks_per_ts):
                    images, _ = self.get_one_shot_batch(
                        support_set_size, is_validation=is_validation)
                    probabilities = model.predict_on_batch(images)

                    # Added this condition because noticed that sometimes the outputs
                    # of the classifier was almost the same in all images, meaning that
                    # the argmax would be always by defenition 0.
                    if np.argmax(probabilities) == 0 and probabilities.std() > 0.01:
                        accuracy = 1.0
                    else:
                        accuracy = 0.0

                    mean_ts_accuracy += accuracy
                    mean_global_accuracy += accuracy

                mean_ts_accuracy /= number_of_tasks_per_ts

                print(trafficSign + ' traffic sign' + ', accuracy: ' +
                      str(mean_ts_accuracy))
                f.write(trafficSign + ' traffic sign' + ', accuracy: ' + str(mean_ts_accuracy) + '\n')
                if is_validation:
                    self._current_validation_ts_index += 1
                else:
                    self._current_evaluation_ts_index += 1

            mean_global_accuracy /= (len(trafficSigns) *
                                     number_of_tasks_per_ts)

            print('\nMean global accuracy: ' + str(mean_global_accuracy))
            f.write('\nMean global accuracy: ' + str(mean_global_accuracy))
            f.flush()  # Force write to disk

        # reset counter
        if is_validation:
            self._current_validation_ts_index = 0
        else:
            self._current_evaluation_ts_index = 0

        return mean_global_accuracy

    def get_original_indices(self, arr, flat_indices):
        original_indices = []
        current_index = 0
        for sub_array in arr:
            if current_index in flat_indices:
                original_indices.append(current_index)
            current_index += 1
        return original_indices

    def get_flattened_values(self, arr):
        flattened_values = []
        for sub_array in arr:
            flattened_values.append(sub_array[0])  # Extract the value from the size-1 ndarray
        return flattened_values

    def one_shot_test(self, model, support_set_size, number_of_tasks_per_ts,
                      is_validation, output_dir):
        """ Prepare one-shot task and evaluate its performance

        Make one shot task in validation and evaluation sets
        if support_set_size = -1 we perform a N-Way one-shot task with
        N being the total of characters in the alphabet

        Returns:
            mean_accuracy: mean accuracy for the one-shot task
        """

        if is_validation:
            trafficSigns = self._validationTS
            print('\nMaking One Shot Task on validation traffic signs:')
            subfolder = 'validation'
            prefix = 'validation_'
        else:
            trafficSigns = self._evaluationTS
            print('\nMaking One Shot Task on evaluation traffic signs:')
            subfolder = 'evaluation'
            prefix = 'evaluation_'

        trafficSigns.sort()

        mean_global_accuracy = 0
        mean_global_ten_accuracy = 0

        output_dir = os.path.join(output_dir, subfolder)
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        existing_files = [f for f in os.listdir(output_dir) if f.startswith(prefix)]
        if existing_files:
            last_num = max([int(f[len(prefix):-4]) for f in existing_files])
            file_num = last_num + 1
        else:
            file_num = 1
        output_file = os.path.join(output_dir, prefix + str(file_num) + '.txt')

        with open(output_file, 'a') as f:
            for trafficSign in trafficSigns:
                mean_ts_accuracy = 0
                mean_ten_accuracy = 0
                for _ in range(number_of_tasks_per_ts):
                    images, _ = self.get_full_one_shot_batch(trafficSign, trafficSigns, subfolder)
                    probabilities = model.predict_on_batch(images)
                    top_10_indices_flat = np.argsort(self.get_flattened_values(probabilities))[-10:][::-1]
                    top_10_classes = []
                    for index in top_10_indices_flat:
                        top_10_classes.append(trafficSigns[index])
                    accuracy = 0.0
                    ten_accuracy = 0.0
                    for index, trafficSignToCompare in enumerate(top_10_classes):
                        if trafficSignToCompare == trafficSign:
                            accuracy = 1.0 - (index / 10)
                            ten_accuracy = 1.0
                            print(accuracy)
                    # if trafficSign in top_10_classes:
                    #     accuracy = 1.0
                    # else:
                    #     accuracy = 0.0
                    mean_ts_accuracy += accuracy
                    mean_ten_accuracy += ten_accuracy
                    mean_global_accuracy += accuracy
                    mean_global_ten_accuracy += ten_accuracy

                mean_ts_accuracy /= number_of_tasks_per_ts
                mean_ten_accuracy /= number_of_tasks_per_ts

                print(trafficSign + ' traffic sign' + ', accuracy: ' +
                      str(mean_ts_accuracy))
                f.write(trafficSign + ' traffic sign' + ', accuracy: ' + str(mean_ts_accuracy) + '\n')
                if is_validation:
                    self._current_validation_ts_index += 1
                else:
                    self._current_evaluation_ts_index += 1

            mean_global_accuracy /= (len(trafficSigns) *
                                     number_of_tasks_per_ts)
            mean_global_ten_accuracy /= (len(trafficSigns) *
                                         number_of_tasks_per_ts)

            print('\nMean global accuracy: ' + str(mean_global_accuracy))
            f.write('\nMean global accuracy: ' + str(mean_global_accuracy))
            f.flush()  # Force write to disk

        # reset counter
        if is_validation:
            self._current_validation_ts_index = 0
        else:
            self._current_evaluation_ts_index = 0

        return mean_global_accuracy, mean_global_ten_accuracy

    def get_all_classes_test(self, model, number_of_tasks_per_ts, output_dir, matrix_name):
        self._evaluationTS = list(self.train_dictionary.keys())

        """ Prepare one-shot task and evaluate its performance

        Make one shot task in validation and evaluation sets
        if support_set_size = -1 we perform a N-Way one-shot task with
        N being the total of characters in the alphabet

        Returns:
            mean_accuracy: mean accuracy for the one-shot task
        """
        trafficSigns = self._validationTS
        trafficSigns.sort()
        trafficSigns_length = len(trafficSigns)
        print('\nMaking One Shot Task on validation traffic signs:')
        subfolder = 'validation'
        prefix = 'full_test_'

        mean_global_accuracy = 0

        output_dir = os.path.join(output_dir, subfolder)
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        existing_files = [f for f in os.listdir(output_dir) if f.startswith(prefix)]
        if existing_files:
            last_num = max([int(f[len(prefix):-4]) for f in existing_files])
            file_num = last_num + 1
        else:
            file_num = 1
        output_file = os.path.join(output_dir, prefix + str(file_num) + '.txt')

        similarity_matrix = np.zeros((trafficSigns_length, trafficSigns_length))
        with open(output_file, 'a') as f:
            for ref_class_index, trafficSign in enumerate(trafficSigns):
                current_matrix = np.zeros((trafficSigns_length, 1))
                mean_ts_accuracy = 0
                for _ in range(number_of_tasks_per_ts):
                    images, _ = self.get_full_one_shot_batch(trafficSign, trafficSigns, 'validation')
                    probabilities = model.predict_on_batch(images)
                    # I need to add all the values of probabilities based on index into the current_matrix
                    for i in range(trafficSigns_length):
                        current_matrix[i] += probabilities[i]
                for i in range(trafficSigns_length):
                    current_matrix[i] /= number_of_tasks_per_ts
                print(f"successfully appended class {trafficSign}")
                # Now I need to add the current_matrix to the similarity_matrix as a next row
                similarity_matrix[ref_class_index] = current_matrix.T
            # Create a DataFrame for better output
            df = pd.DataFrame(similarity_matrix, index=trafficSigns, columns=trafficSigns)

            # Save as CSV for readability
            df.to_csv(os.path.join(output_dir, f'{matrix_name}.csv'))
            print('Similarity matrix saved as CSV')

    def get_cropped_images_test(self, model, output_dir, matrix_path, number_of_tasks_per_image=5):
        self._evaluationTS = list(self.train_dictionary.keys())
        traffic_sign_classes = self._validationTS
        traffic_sign_classes.sort()

        cropped_images = []

        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if file_path.__contains__("grouped") is False:
                cropped_images.append(file_path)
        cropped_images.sort()
        batch_length = int(len(cropped_images) / 10)
        print(batch_length)
        starting_index = 0
        for i in range(1, 11):
            cropped_images_divided = []
            if i == 10:
                cropped_images_divided = cropped_images[starting_index:]
            else:
                cropped_images_divided = cropped_images[starting_index:(starting_index + batch_length)]
            starting_index += batch_length
            images_length = len(cropped_images_divided)
            traffic_classes_length = len(traffic_sign_classes)
            print('\nMaking One Shot Task on cropped images traffic signs:')
            subfolder = 'validation'
            prefix = 'full_test_'

            similarity_matrix = np.zeros((images_length, traffic_classes_length))
            for ref_class_index, trafficSign in enumerate(cropped_images_divided):
                current_matrix = np.zeros((traffic_classes_length, 1))
                for _ in range(number_of_tasks_per_image):
                    images, _ = self.get_full_one_shot_batch_no_labels(trafficSign, traffic_sign_classes, 'validation')
                    probabilities = model.predict_on_batch(images)
                    # I need to add all the values of probabilities based on index into the current_matrix
                    for i in range(traffic_classes_length):
                        current_matrix[i] += probabilities[i]
                for i in range(traffic_classes_length):
                    current_matrix[i] /= number_of_tasks_per_image
                print(f"successfully appended class {trafficSign}")
                # Now I need to add the current_matrix to the similarity_matrix as a next row
                similarity_matrix[ref_class_index] = current_matrix.T
            # Create a DataFrame for better output
            df = pd.DataFrame(similarity_matrix, index=cropped_images_divided, columns=traffic_sign_classes)
            # Save as CSV for readability
            df.to_csv(f"{matrix_path}_{starting_index}.csv")
            print(f'Similarity matrix {starting_index} saved as CSV')
