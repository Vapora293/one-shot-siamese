import os
import random
import numpy as np
import math
from PIL import Image

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

    def __init__(self, dataset_path, use_augmentation, batch_size):
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
        self.image_width = 256
        self.image_height = 256
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self._trainTS = []
        self._validationTS = []
        self._evaluationTS = []

        self.load_dataset()

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
        validation_path = os.path.join(self.dataset_path, 'eval')

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

        availableTrafficSigns = list(self.train_dictionary.keys())
        numberOfTS = len(availableTrafficSigns)

        train_indexes = random.sample(
            range(0, numberOfTS - 1), int(0.8 * numberOfTS))

        # If we sort the indexes in reverse order we can pop them from the list
        # and don't care because the indexes do not change
        train_indexes.sort(reverse=True)

        for index in train_indexes:
            self._trainTS.append(availableTrafficSigns[index])
            availableTrafficSigns.pop(index)

        # The remaining TS are saved for validation
        self._validationTS = availableTrafficSigns
        self._evaluationTS = list(self.evaluation_dictionary.keys())

    def _convert_path_list_to_images_and_labels(self, path_list, is_one_shot_task):
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
        pairs_of_images = [np.zeros((number_of_pairs,
                                     self.image_height, self.image_height, 4)) for _ in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):
            image = Image.open(os.path.abspath(path_list[pair][0]))
            image = np.asarray(image).astype(np.float64)
            image = image / image.std() - image.mean()

            pairs_of_images[0][pair, :, :, :] = image
            try:
                image = Image.open(path_list[pair][1])
                image = np.asarray(image).astype(np.float64)
                image = image / image.std() - image.mean()
            except:
                image = self.image_augmentor.get_random_transform_single_image(image)
            pairs_of_images[1][pair, :, :, :] = image

            if not is_one_shot_task:
                if (pair < number_of_pairs / 2):
                    labels[pair] = 1
                else:
                    labels[pair] = 0
            else:
                if pair == 0:
                    labels[pair] = 1
                else:
                    labels[pair] = 0

        if not is_one_shot_task:
            random_permutation = np.random.permutation(number_of_pairs)
            labels = labels[random_permutation]
            pairs_of_images[0][:, :, :,
            :] = pairs_of_images[0][random_permutation, :, :, :]
            pairs_of_images[1][:, :, :,
            :] = pairs_of_images[1][random_permutation, :, :, :]

        return pairs_of_images, labels

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
        randomClasses = random.sample(range(0, len(self._trainTS)), 16)
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
            randomOtherTS = self._trainTS[random.randint(0, len(self._trainTS) - 1)]
            differentTS.append(self.get_random_image_from_class(randomOtherTS, 1, 'train'))
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
            image_folder_name = 'train'
            dictionary = self.train_dictionary
        else:
            trafficSigns = self._evaluationTS
            image_folder_name = 'eval'
            dictionary = self.evaluation_dictionary

        currentTSIndex = random.randint(0, len(trafficSigns) - 1)
        currentTS = trafficSigns[currentTSIndex]
        availableTS = list(dictionary[currentTS].keys())
        numberOfTS = len(availableTS)

        batchImagePath = []

        image_indexes = random.sample(range(0, ), 2)

        test_image = os.path.join(
            image_path, available_images[image_indexes[0]])
        batchImagePath.append(test_image)
        image = os.path.join(
            image_path, available_images[image_indexes[1]])
        batchImagePath.append(image)

        # Let's get our test image and a pair corresponding to
        if support_set_size == -1:
            number_of_support_characters = numberOfTS
        else:
            number_of_support_characters = support_set_size

        different_characters = availableTS[:]
        different_characters.pop(test_character_index[0])

        # There may be some alphabets with less than 20 characters
        if numberOfTS < number_of_support_characters:
            number_of_support_characters = numberOfTS

        support_characters_indexes = random.sample(
            range(0, numberOfTS - 1), number_of_support_characters - 1)

        for index in support_characters_indexes:
            current_character = different_characters[index]
            available_images = (dictionary[currentTS])[
                current_character]
            image_path = os.path.join(
                self.dataset_path, image_folder_name, currentTS, current_character)

            image_indexes = random.sample(range(0, 20), 1)
            image = os.path.join(
                image_path, available_images[image_indexes[0]])
            batchImagePath.append(test_image)
            batchImagePath.append(image)

        images, labels = self._convert_path_list_to_images_and_labels(
            batchImagePath, is_one_shot_task=True)

        return images, labels

    def one_shot_test(self, model, support_set_size, number_of_tasks_per_alphabet,
                      is_validation):
        """ Prepare one-shot task and evaluate its performance

        Make one shot task in validation and evaluation sets
        if support_set_size = -1 we perform a N-Way one-shot task with
        N being the total of characters in the alphabet

        Returns:
            mean_accuracy: mean accuracy for the one-shot task
        """

        # Set some variables that depend on dataset
        if is_validation:
            alphabets = self._validationTS
            print('\nMaking One Shot Task on validation alphabets:')
        else:
            alphabets = self._evaluationTS
            print('\nMaking One Shot Task on evaluation alphabets:')

        mean_global_accuracy = 0

        for alphabet in alphabets:
            mean_alphabet_accuracy = 0
            for _ in range(number_of_tasks_per_alphabet):
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

                mean_alphabet_accuracy += accuracy
                mean_global_accuracy += accuracy

            mean_alphabet_accuracy /= number_of_tasks_per_alphabet

            print(alphabet + ' alphabet' + ', accuracy: ' +
                  str(mean_alphabet_accuracy))
            if is_validation:
                self._current_validation_alphabet_index += 1
            else:
                self._current_evaluation_alphabet_index += 1

        mean_global_accuracy /= (len(alphabets) *
                                 number_of_tasks_per_alphabet)

        print('\nMean global accuracy: ' + str(mean_global_accuracy))

        # reset counter
        if is_validation:
            self._current_validation_alphabet_index = 0
        else:
            self._current_evaluation_alphabet_index = 0

        return mean_global_accuracy
