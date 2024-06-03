import os

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda
from keras.optimizers import Adam, SGD
# from keras import optimizers
from keras.regularizers import l2
# from keras import regularizers
import keras.backend as K

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# from omniglot_loader import OmniglotLoader
from ts_loader import TSLoader
from modified_sgd import Modified_SGD


class SiameseNetwork:
    """Class that constructs the Siamese Net for training

    This Class was constructed to create the siamese net and train it.

    Attributes:
        input_shape: image size
        model: current siamese model
        learning_rate: SGD learning rate
        omniglot_loader: instance of OmniglotLoader
        summary_writer: tensorflow writer to store the logs
    """

    def __init__(self, dataset_path, learning_rate, batch_size, use_augmentation,
                 learning_rate_multipliers, l2_regularization_penalization, tensorboard_log_path, grayscale=False,
                 gpu=1):
        """Inits SiameseNetwork with the provided values for the attributes.

        It also constructs the siamese network architecture, creates a dataset
        loader and opens the log file.

        Arguments:
            dataset_path: path of Omniglot dataset
            learning_rate: SGD learning rate
            batch_size: size of the batch to be used in training
            use_augmentation: boolean that allows us to select if data augmentation
                is used or not
            learning_rate_multipliers: learning-rate multipliers (relative to the learning_rate
                chosen) that will be applied to each fo the conv and dense layers
                for example:
                    # Setting the Learning rate multipliers
                    LR_mult_dict = {}
                    LR_mult_dict['conv1']=1
                    LR_mult_dict['conv2']=1
                    LR_mult_dict['dense1']=2
                    LR_mult_dict['dense2']=2
            l2_regularization_penalization: l2 penalization for each layer.
                for example:
                    # Setting the Learning rate multipliers
                    L2_dictionary = {}
                    L2_dictionary['conv1']=0.1
                    L2_dictionary['conv2']=0.001
                    L2_dictionary['dense1']=0.001
                    L2_dictionary['dense2']=0.01
            tensorboard_log_path: path to store the logs
        """
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Target the 1st available GPU (index 1)
                tf.config.set_visible_devices(gpus[gpu], 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)
        # tf.compat.v1.disable_eager_execution()
        if grayscale:
            self.input_shape = (128, 128, 1)  # Size of images
        else:
            print("It will be 4 thing shape")
            self.input_shape = (128, 128, 4)
        self.model = []
        self.learning_rate = learning_rate
        self.ts_loader = TSLoader(
            dataset_path=dataset_path, use_augmentation=use_augmentation, batch_size=batch_size, grayscale=grayscale)
        self.summary_writer = tf.summary.create_file_writer(tensorboard_log_path)
        self._construct_siamese_architecture(learning_rate_multipliers,
                                             l2_regularization_penalization)

    def _construct_siamese_architecture(self, learning_rate_multipliers,
                                        l2_regularization_penalization):
        """ Constructs the siamese architecture and stores it in the class

        Arguments:
            learning_rate_multipliers
            l2_regularization_penalization
        """

        # Let's define the cnn architecture
        convolutional_net = Sequential()
        convolutional_net.add(Conv2D(filters=64, kernel_size=(10, 10),
                                     activation='relu',
                                     input_shape=self.input_shape,
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv1']),
                                     name='Conv1'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Conv2D(filters=128, kernel_size=(7, 7),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv2']),
                                     name='Conv2'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Conv2D(filters=256, kernel_size=(4, 4),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv3']),
                                     name='Conv3'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Conv2D(filters=512, kernel_size=(4, 4),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv4']),
                                     name='Conv4'))

        convolutional_net.add(Flatten())
        convolutional_net.add(
            Dense(units=4096, activation='sigmoid',
                  kernel_regularizer=l2(
                      l2_regularization_penalization['Dense1']),
                  name='Dense1'))

        # Filter Visualization for TensorBoard
        # for layer in convolutional_net.layers:
        #     if 'Conv' in layer.name:
        #         filters = layer.get_weights()[0]
        #         filter_vis = self.visualize_filters(filters)  # We'll define this function next
        #         tf.summary.image(layer.name, filter_vis, max_outputs=filters.shape[3])

        # Now the pairs of images
        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)

        encoded_image_1 = convolutional_net(input_image_1)
        encoded_image_2 = convolutional_net(input_image_2)

        # L1 distance layer between the two encoded outputs
        # One could use Subtract from Keras, but we want the absolute value
        l1_distance_layer = Lambda(
            lambda tensors: K.abs(tensors[0] - tensors[1]))
        l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])

        # Same class or not prediction
        prediction = Dense(units=1, activation='sigmoid')(l1_distance)
        self.model = Model(
            inputs=[input_image_1, input_image_2], outputs=prediction)

        # Define the optimizer and compile the model
        optimizer = Modified_SGD(
            lr=self.learning_rate,
            lr_multipliers=learning_rate_multipliers,
            momentum=0.0)

        self.model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],
                           optimizer=optimizer)

    def visualize_filters(self, filters, reshape_shape=(10, 10)):
        """Reshapes and transposes convolutional filters for visualization.

        Args:
            filters: A 4D Tensor of convolutional filters (kernel_size, kernel_size, in_channels, out_channels)
            reshape_shape: Shape to reshape the visualization grid

        Returns:
            A reshaped 4D Tensor ready for `tf.summary.image`
        """
        # filters = np.squeeze(filters)  # May need squeezing if there's a batch dim
        # filters = np.transpose(filters, (3, 0, 1, 2))  # Transpose for visualization
        # n_groups, n_rows, n_cols, _ = filters.shape
        # # filters = filters.reshape(n_groups * n_rows, n_cols, *reshape_shape)
        # return tf.expand_dims(filters, -1)  # Add a channel dimension for grayscale

    def _write_logs_to_tensorboard(self, current_iteration, train_losses,
                                   train_accuracies, validation_accuracy,
                                   ten_class_validation_accuracy,
                                   evaluate_each):
        """ Writes the logs to a tensorflow log file

        This allows us to see the loss curves and the metrics in tensorboard.
        If we wrote every iteration, the training process would be slow, so
        instead we write the logs every evaluate_each iteration.

        Arguments:
            current_iteration: iteration to be written in the log file
            train_losses: contains the train losses from the last evaluate_each
                iterations.
            train_accuracies: the same as train_losses but with the accuracies
                in the training set.
            validation_accuracy: accuracy in the current one-shot task in the
                validation set
            evaluate each: number of iterations defined to evaluate the one-shot
                tasks.
        """

        with self.summary_writer.as_default():
            for index in range(0, evaluate_each):
                tf.summary.scalar('Train Loss', train_losses[index], step=current_iteration - evaluate_each + index + 1)
                tf.summary.scalar('Train Accuracy', train_accuracies[index],
                                  step=current_iteration - evaluate_each + index + 1)

                if index == (evaluate_each - 1):
                    tf.summary.scalar('One-Shot Validation Accuracy', validation_accuracy,
                                      step=current_iteration - evaluate_each + index + 1)
                    tf.summary.scalar('Ten Classes Validation Accuracy', ten_class_validation_accuracy,
                                      step=current_iteration - evaluate_each + index + 1)
            self.summary_writer.flush()

    def train_siamese_network(self, number_of_iterations, support_set_size,
                              final_momentum, momentum_slope, evaluate_each,
                              model_name, general_output_file_path, bayesian):
        """ Train the Siamese net

        This is the main function for training the siamese net.
        In each every evaluate_each train iterations we evaluate one-shot tasks in
        validation and evaluation set. We also write to the log file.

        Arguments:
            number_of_iterations: maximum number of iterations to train.
            support_set_size: number of characters to use in the support set
                in one-shot tasks.
            final_momentum: mu_j in the paper. Each layer starts at 0.5 momentum
                but evolves linearly to mu_j
            momentum_slope: slope of the momentum evolution. In the paper we are
                only told that this momentum evolves linearly. Because of that I
                defined a slope to be passed to the training.
            evaluate each: number of iterations defined to evaluate the one-shot
                tasks.
            model_name: save_name of the model

        Returns:
            Evaluation Accuracy
        """

        # First of all let's divide randomly the 30 train alphabets in train
        # and validation with 24 for training and 6 for validation
        # self.omniglot_loader.split_train_datasets()
        self.ts_loader.split_train_datasets()
        # Variables that will store 1000 iterations losses and accuracies
        # after evaluate_each iterations these will be passed to tensorboard logs
        train_losses = np.zeros(shape=(evaluate_each))
        train_accuracies = np.zeros(shape=(evaluate_each))
        count = 0
        # Stop criteria variables
        best_validation_accuracy = 0.0
        best_accuracy_iteration = 0
        validation_accuracy = 0.0
        general_output_file = open(general_output_file_path + 'general.txt', 'a')

        # Train loop
        for iteration in range(number_of_iterations):

            # train set
            # images, labels = self.omniglot_loader.get_train_batch()
            images, labels = self.ts_loader.get_train_batch()
            train_loss, train_accuracy = self.model.train_on_batch(
                images, labels)

            # Decay learning rate 1 % per 500 iterations (in the paper the decay is
            # 1% per epoch). Also update linearly the momentum (starting from 0.5 to 1)
            if (iteration + 1) % 500 == 0:
                current_lr = K.get_value(self.model.optimizer._learning_rate)
                K.set_value(self.model.optimizer._learning_rate, current_lr * 0.99)
            if K.get_value(self.model.optimizer.momentum) < final_momentum:
                K.set_value(self.model.optimizer.momentum, K.get_value(
                    self.model.optimizer.momentum) + momentum_slope)

            train_losses[count] = train_loss
            train_accuracies[count] = train_accuracy

            # validation set
            count += 1
            general_output_file.write('\nIteration %d/%d: Train loss: %f, Train Accuracy: %f, lr = %f' %
                                      (iteration + 1, number_of_iterations, train_loss, train_accuracy,
                                       K.get_value(self.model.optimizer.lr)))
            if count % 5 == 0:
                print('Iteration %d/%d: Train loss: %f, Train Accuracy: %f, lr = %f' %
                      (iteration + 1, number_of_iterations, train_loss, train_accuracy, K.get_value(
                          self.model.optimizer.lr)))
            general_output_file.flush()  # Force write to disk

            # Each 100 iterations perform a one_shot_task and write to tensorboard the
            # stored losses and accuracies
            # if (bayesian is False and ((iteration + 1) % evaluate_each == 0)) or (bayesian and ((iteration + 1) == (evaluate_each)) or ((iteration + 1) == (evaluate_each + 1000)) or ((iteration + 1) == (evaluate_each + 1999))):
            if iteration == 20000:
                evaluate_each = 3000
            if ((iteration + 1) % 3000 == 0):
                if ((iteration + 1) < 20000 and (iteration + 1) % evaluate_each != 0):
                    continue
                number_of_runs_per_class = 5
                # use a support set size equal to the number of character in the alphabet
                validation_accuracy, ten_classes_accuracy = self.ts_loader.one_shot_test(
                    self.model, support_set_size, number_of_runs_per_class, is_validation=True,
                    output_dir=general_output_file_path)

                self._write_logs_to_tensorboard(
                    iteration, train_losses, train_accuracies,
                    validation_accuracy, ten_classes_accuracy, evaluate_each)
                count = 0

                # Some hyperparameters lead to 100%, although the output is almost the same in
                # all images.
                if (validation_accuracy == 1.0 and train_accuracy == 0.5):
                    general_output_file.write('Early Stopping: Gradient Explosion')
                    general_output_file.write('\nValidation Accuracy = ' +
                                              str(best_validation_accuracy))
                    print('Early Stopping: Gradient Explosion')
                    print('Validation Accuracy = ' +
                          str(best_validation_accuracy))
                    return 0
                elif train_accuracy == 0.0:
                    return 0
                else:
                    # Save the model
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        best_accuracy_iteration = iteration

                        model_json = self.model.to_json()

                        if not os.path.exists('./models'):
                            os.makedirs('./models')
                        with open('models/' + model_name + str(iteration) + '.json', "w") as json_file:
                            json_file.write(model_json)
                        if bayesian is False:
                            self.model.save_weights('models/' + model_name + str(iteration) + '.h5')

                    elif (iteration > (best_accuracy_iteration + 10000)):
                        best_accuracy_iteration += 10000
                        model_json = self.model.to_json()

                        if not os.path.exists('./models'):
                            os.makedirs('./models')
                        with open('models/' + model_name + str(iteration) + '.json', "w") as json_file:
                            json_file.write(model_json)
                        if bayesian is False:
                            self.model.save_weights('models/' + model_name + str(iteration) + '.h5')

            # If accuracy does not improve for 10000 batches stop the training
            if bayesian is False and (iteration - best_accuracy_iteration > (evaluate_each * 20)):
                print(
                    'Early Stopping: validation accuracy did not increase for x iterations')
                print('Best Validation Accuracy = ' +
                      str(best_validation_accuracy))
                print('Validation Accuracy = ' + str(best_validation_accuracy))
                general_output_file.write('Early Stopping: validation accuracy did not increase for 10000 iterations')
                general_output_file.write('Best Validation Accuracy = ' +
                                          str(best_validation_accuracy))
                general_output_file.write('Validation Accuracy = ' + str(best_validation_accuracy))
                break

        print('Training Ended!')
        general_output_file.write('\nTraining Ended!')
        return best_validation_accuracy
