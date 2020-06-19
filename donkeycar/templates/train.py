#!/usr/bin/env python3
'''
Scripts to train a keras model using tensorflow.
Basic usage should feel familiar: python train_v2.py --model models/mypilot

Usage:
    train.py [--tubs=tubs] (--model=<model>) [--type=(linear|inferred|tensorrt_linear|tflite_linear)]

Options:
    -h --help              Show this screen.
'''

import os
import random
from pathlib import Path

import cv2
import numpy as np
from docopt import docopt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.utils.data_utils import Sequence

import donkeycar
from donkeycar.parts.keras import KerasInferred
from donkeycar.parts.tub_v2 import Tub
from donkeycar.utils import get_model_by_type, load_scaled_image_arr


class TubDataset(object):
    '''
    Loads the dataset, and creates a train/test split.
    '''
    def __init__(self, tub_paths, test_size=0.2, shuffle=True):
        self.tub_paths = tub_paths
        self.test_size = test_size
        self.shuffle = shuffle
        self.tubs = [Tub(tub_path) for tub_path in self.tub_paths]
        self.records = list()

    def train_test_split(self):
        print('Loading tubs from paths %s' % (self.tub_paths))
        for tub in self.tubs:
            for record in tub:
                record['_image_base_path'] = tub.images_base_path
                self.records.append(record)

        return train_test_split(self.records, test_size=self.test_size, shuffle=self.shuffle)


class TubSequence(Sequence):
    def __init__(self, keras_model, config, records=list()):
        self.keras_model = keras_model
        self.config = config
        self.records = records
        self.batch_size = self.config.BATCH_SIZE

    def __len__(self):
        return len(self.records) // self.batch_size

    def __getitem__(self, index):
        count = 0
        records = []
        images = []
        angles = []
        throttles = []

        is_inferred = type(self.keras_model) is KerasInferred

        while count < self.batch_size:
            i = (index * self.batch_size) + count
            if i >= len(self.records):
                break

            record = self.records[i]
            record = self._transform_record(record)
            records.append(record)
            count += 1

        for record in records:
            image = record['cam/image_array']
            angle = record['user/angle']
            throttle = record['user/throttle']
        
            images.append(image)
            angles.append(angle)
            throttles.append(throttle)

        X = np.array(images)

        if is_inferred:
            Y = np.array(angles)
        else:
            Y = [np.array(angles), np.array(throttles)]

        return X, Y

    def _transform_record(self, record):
        for key, value in record.items():
            if key == 'cam/image_array' and isinstance(value, str):
                image_path = os.path.join(record['_image_base_path'], value)
                image = load_scaled_image_arr(image_path, self.config)
                record[key] = image

        return record


class ImagePreprocessing(Sequence):
    '''
    A Sequence which wraps another Sequence with an Image Augumentation.
    '''
    def __init__(self, sequence, augmentation):
        self.sequence = sequence
        self.augumentation = augmentation

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        X, Y = self.sequence[index]
        return self.augumentation.augment_images(X), Y


def train(cfg, tub_paths, output_path, model_type):
    '''
    Train the model
    '''
    if 'linear' in model_type:
        train_type = 'linear'
    else:
        train_type = model_type

    kl = get_model_by_type(train_type, cfg)
    kl.compile()

    if cfg.PRINT_MODEL_SUMMARY:
        print(kl.model.summary())

<<<<<<< HEAD
    if continuous:
        epochs = 100000
    else:
        epochs = cfg.MAX_EPOCHS

    workers_count = 1
    use_multiprocessing = False

    callbacks_list = [save_best]

    if cfg.USE_EARLY_STOP and not continuous:
        callbacks_list.append(early_stop)

    history = kl.model.fit_generator(
                    train_gen, 
                    steps_per_epoch=steps_per_epoch, 
                    epochs=epochs, 
                    verbose=cfg.VERBOSE_TRAIN,
                    validation_data=val_gen,
                    callbacks=callbacks_list, 
                    validation_steps=val_steps,
                    workers=workers_count,
                    use_multiprocessing=use_multiprocessing)
                    
    full_model_val_loss = min(history.history['val_loss'])
    max_val_loss = full_model_val_loss + cfg.PRUNE_VAL_LOSS_DEGRADATION_LIMIT

    duration_train = time.time() - start
    print("Training completed in %s." % str(datetime.timedelta(seconds=round(duration_train))) )

    print("\n\n----------- Best Eval Loss :%f ---------" % save_best.best)

    if cfg.SHOW_PLOT:
        try:
            if do_plot:
                plt.figure(1)

                # Only do accuracy if we have that data (e.g. categorical outputs)
                if 'angle_out_acc' in history.history:
                    plt.subplot(121)

                # summarize history for loss
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'validate'], loc='upper right')
                
                # summarize history for acc
                if 'angle_out_acc' in history.history:
                    plt.subplot(122)
                    plt.plot(history.history['angle_out_acc'])
                    plt.plot(history.history['val_angle_out_acc'])
                    plt.title('model angle accuracy')
                    plt.ylabel('acc')
                    plt.xlabel('epoch')
                    #plt.legend(['train', 'validate'], loc='upper left')

                plt.savefig(model_path + '_loss_acc_%f.%s' % (save_best.best, figure_format))
                plt.show()
            else:
                print("not saving loss graph because matplotlib not set up.")
        except Exception as ex:
            print("problems with loss graph: {}".format( ex ) )

    #Save tflite, optionally in the int quant format for Coral TPU
    if "tflite" in cfg.model_type:
        print("\n\n--------- Saving TFLite Model ---------")
        tflite_fnm = model_path.replace(".h5", ".tflite")
        assert(".tflite" in tflite_fnm)

        prepare_for_coral = "coral" in cfg.model_type

        if prepare_for_coral:
            #compile a list of records to calibrate the quantization
            data_list = []
            max_items = 1000
            for key, _record in gen_records.items():
                data_list.append(_record)
                if len(data_list) == max_items:
                    break   

            stride = 1
            num_calibration_steps = len(data_list) // stride

            #a generator function to help train the quantizer with the expected range of data from inputs
            def representative_dataset_gen():
                start = 0
                end = stride
                for _ in range(num_calibration_steps):
                    batch_data = data_list[start:end]
                    inputs = []
                
                    for record in batch_data:
                        filename = record['image_path']                        
                        img_arr = load_scaled_image_arr(filename, cfg)
                        inputs.append(img_arr)

                    start += stride
                    end += stride

                    # Get sample input data as a numpy array in a method of your choosing.
                    yield [ np.array(inputs, dtype=np.float32).reshape(stride, cfg.TARGET_H, cfg.TARGET_W, cfg.TARGET_D) ]
        else:
            representative_dataset_gen = None

        from donkeycar.parts.tflite import keras_model_to_tflite
        keras_model_to_tflite(model_path, tflite_fnm, representative_dataset_gen)
        print("Saved TFLite model:", tflite_fnm)
        if prepare_for_coral:
            print("compile for Coral w: edgetpu_compiler", tflite_fnm)
            os.system("edgetpu_compiler " + tflite_fnm)

    #Save tensorrt
    if "tensorrt" in cfg.model_type:
        print("\n\n--------- Saving TensorRT Model ---------")
        # TODO RAHUL
        # flatten model_path
        # convert to uff
        # print("Saved TensorRT model:", uff_filename)

    
def sequence_train(cfg, tub_names, model_name, transfer_model, model_type, continuous, aug):
    '''
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    trains models which take sequence of images
    '''
    assert(not continuous)

    print("sequence of images training")    

    kl = dk.utils.get_model_by_type(model_type=model_type, cfg=cfg)
    
    if cfg.PRINT_MODEL_SUMMARY:
        print(kl.model.summary())
    
    tubs = gather_tubs(cfg, tub_names)
    
    verbose = cfg.VERBOSE_TRAIN

    records = []

    for tub in tubs:
        record_paths = glob.glob(os.path.join(tub.path, 'record_*.json'))
        print("Tub:", tub.path, "has", len(record_paths), 'records')

        record_paths.sort(key=get_record_index)
        records += record_paths


    print('collating records')
    gen_records = {}

    for record_path in records:

        with open(record_path, 'r') as fp:
            json_data = json.load(fp)

        basepath = os.path.dirname(record_path)
        image_filename = json_data["cam/image_array"]
        image_path = os.path.join(basepath, image_filename)
        sample = { 'record_path' : record_path, "image_path" : image_path, "json_data" : json_data }

        sample["tub_path"] = basepath
        sample["index"] = get_image_index(image_filename)

        angle = float(json_data['user/angle'])
        throttle = float(json_data["user/throttle"])

        sample['target_output'] = np.array([angle, throttle])
        sample['angle'] = angle
        sample['throttle'] = throttle


        sample['img_data'] = None

        key = make_key(sample)

        gen_records[key] = sample



    print('collating sequences')

    sequences = []
    
    target_len = cfg.SEQUENCE_LENGTH
    look_ahead = False
    
    if model_type == "look_ahead":
        target_len = cfg.SEQUENCE_LENGTH * 2
        look_ahead = True

    for k, sample in gen_records.items():

        seq = []

        for i in range(target_len):
            key = make_next_key(sample, i)
            if key in gen_records:
                seq.append(gen_records[key])
            else:
                continue

        if len(seq) != target_len:
            continue

        sequences.append(seq)

    print("collated", len(sequences), "sequences of length", target_len)

    #shuffle and split the data
    train_data, val_data  = train_test_split(sequences, test_size=(1 - cfg.TRAIN_TEST_SPLIT))


    def generator(data, opt, batch_size=cfg.BATCH_SIZE):
        num_records = len(data)

        while True:
            #shuffle again for good measure
            random.shuffle(data)

            for offset in range(0, num_records, batch_size):
                batch_data = data[offset:offset+batch_size]

                if len(batch_data) != batch_size:
                    break

                b_inputs_img = []
                b_vec_in = []
                b_labels = []
                b_vec_out = []

                for seq in batch_data:
                    inputs_img = []
                    vec_in = []
                    labels = []
                    vec_out = []
                    num_images_target = len(seq)
                    iTargetOutput = -1
                    if opt['look_ahead']:
                        num_images_target = cfg.SEQUENCE_LENGTH
                        iTargetOutput = cfg.SEQUENCE_LENGTH - 1

                    for iRec, record in enumerate(seq):
                        #get image data if we don't already have it
                        if len(inputs_img) < num_images_target:
                            if record['img_data'] is None:
                                img_arr = load_scaled_image_arr(record['image_path'], cfg)
                                if img_arr is None:
                                    break
                                if aug:
                                    img_arr = augment_image(img_arr)
                                
                                if cfg.CACHE_IMAGES:
                                    record['img_data'] = img_arr
                            else:
                                img_arr = record['img_data']                  
                                
                            inputs_img.append(img_arr)

                        if iRec >= iTargetOutput:
                            vec_out.append(record['angle'])
                            vec_out.append(record['throttle'])
                        else:
                            vec_in.append(0.0) #record['angle'])
                            vec_in.append(0.0) #record['throttle'])
                        
                    label_vec = seq[iTargetOutput]['target_output']

                    if look_ahead:
                        label_vec = np.array(vec_out)

                    labels.append(label_vec)

                    b_inputs_img.append(inputs_img)
                    b_vec_in.append(vec_in)

                    b_labels.append(labels)

                
                if look_ahead:
                    X = [np.array(b_inputs_img).reshape(batch_size,\
                        cfg.TARGET_H, cfg.TARGET_W, cfg.SEQUENCE_LENGTH)]
                    X.append(np.array(b_vec_in))
                    y = np.array(b_labels).reshape(batch_size, (cfg.SEQUENCE_LENGTH + 1) * 2)
                else:
                    X = [np.array(b_inputs_img).reshape(batch_size,\
                        cfg.SEQUENCE_LENGTH, cfg.TARGET_H, cfg.TARGET_W, cfg.TARGET_D)]
                    y = np.array(b_labels).reshape(batch_size, 2)

                yield X, y

    opt = { 'look_ahead' : look_ahead, 'cfg' : cfg }

    train_gen = generator(train_data, opt)
    val_gen = generator(val_data, opt)   

    model_path = os.path.expanduser(model_name)

    total_records = len(sequences)
    total_train = len(train_data)
    total_val = len(val_data)

    print('train: %d, validation: %d' %(total_train, total_val))
    steps_per_epoch = total_train // cfg.BATCH_SIZE
    val_steps = total_val // cfg.BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)

    if steps_per_epoch < 2:
        raise Exception("Too little data to train. Please record more records.")
    
    cfg.model_type = model_type

    go_train(kl, cfg, train_gen, val_gen, gen_records, model_name, steps_per_epoch, val_steps, continuous, verbose)
    
    ''' 
    kl.train(train_gen, 
        val_gen, 
        saved_model_path=model_path,
        steps=steps_per_epoch,
        train_split=cfg.TRAIN_TEST_SPLIT,
        use_early_stop = cfg.USE_EARLY_STOP)
    '''


def multi_train(cfg, tub, model, transfer, model_type, continuous, aug):
    '''
    choose the right regime for the given model type
    '''
    train_fn = train
    if model_type in ("rnn",'3d','look_ahead'):
        train_fn = sequence_train

    train_fn(cfg, tub, model, transfer, model_type, continuous, aug)


def prune(model, validation_generator, val_steps, cfg):
    percent_pruning = float(cfg.PRUNE_PERCENT_PER_ITERATION)
    total_channels = get_total_channels(model)
    n_channels_delete = int(math.floor(percent_pruning / 100 * total_channels))

    apoz_df = get_model_apoz(model, validation_generator)

    model = prune_model(model, apoz_df, n_channels_delete)

    name = '{}/model_pruned_{}_percent.h5'.format(cfg.MODELS_PATH, percent_pruning)

    model.save(name)

    return model, n_channels_delete


def extract_data_from_pickles(cfg, tubs):
    """
    Extracts record_{id}.json and image from a pickle with the same id if exists in the tub.
    Then writes extracted json/jpg along side the source pickle that tub.
    This assumes the format {id}.pickle in the tub directory.
    :param cfg: config with data location configuration. Generally the global config object.
    :param tubs: The list of tubs involved in training.
    :return: implicit None.
    """
    t_paths = gather_tub_paths(cfg, tubs)
    for tub_path in t_paths:
        file_paths = glob.glob(join(tub_path, '*.pickle'))
        print('found {} pickles writing json records and images in tub {}'.format(len(file_paths), tub_path))
        for file_path in file_paths:
            # print('loading data from {}'.format(file_paths))
            with open(file_path, 'rb') as f:
                p = zlib.decompress(f.read())
            data = pickle.loads(p)
           
            base_path = dirname(file_path)
            filename = splitext(basename(file_path))[0]
            image_path = join(base_path, filename + '.jpg')
            img = Image.fromarray(np.uint8(data['val']['cam/image_array']))
            img.save(image_path)
            
            data['val']['cam/image_array'] = filename + '.jpg'

            with open(join(base_path, 'record_{}.json'.format(filename)), 'w') as f:
                json.dump(data['val'], f)


def prune_model(model, apoz_df, n_channels_delete):
    from kerassurgeon import Surgeon
    import pandas as pd

    # Identify 5% of channels with the highest APoZ in model
    sorted_apoz_df = apoz_df.sort_values('apoz', ascending=False)
    high_apoz_index = sorted_apoz_df.iloc[0:n_channels_delete, :]

    # Create the Surgeon and add a 'delete_channels' job for each layer
    # whose channels are to be deleted.
    surgeon = Surgeon(model, copy=True)
    for name in high_apoz_index.index.unique().values:
        channels = list(pd.Series(high_apoz_index.loc[name, 'index'],
                                  dtype=np.int64).values)
        surgeon.add_job('delete_channels', model.get_layer(name),
                        channels=channels)
    # Delete channels
    return surgeon.operate()


def get_total_channels(model):
    start = None
    end = None
    channels = 0
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            channels += layer.filters
    return channels


def get_model_apoz(model, generator):
    from kerassurgeon.identify import get_apoz
    import pandas as pd

    # Get APoZ
    start = None
    end = None
    apoz = []
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            print(layer.name)
            apoz.extend([(layer.name, i, value) for (i, value)
                         in enumerate(get_apoz(model, layer, generator))])

    layer_name, index, apoz_value = zip(*apoz)
    apoz_df = pd.DataFrame({'layer': layer_name, 'index': index,
                            'apoz': apoz_value})
    apoz_df = apoz_df.set_index('layer')
    return apoz_df

    
def removeComments( dir_list ):
    for i in reversed(range(len(dir_list))):
        if dir_list[i].startswith("#"):
            del dir_list[i]
        elif len(dir_list[i]) == 0:
            del dir_list[i]

def preprocessFileList( filelist ):
    dirs = []
    if filelist is not None:
        for afile in filelist:
            with open(afile, "r") as f:
                tmp_dirs = f.read().split('\n')
                dirs.extend(tmp_dirs)

    removeComments( dirs )
    return dirs

if __name__ == "__main__":
=======
    batch_size = cfg.BATCH_SIZE
    dataset = TubDataset(tub_paths, test_size=(1. - cfg.TRAIN_TEST_SPLIT))
    training_records, validation_records = dataset.train_test_split()
    print('Records # Training %s' % (len(training_records)))
    print('Records # Validation %s' % (len(validation_records)))

    training = TubSequence(kl, cfg, training_records)
    validation = TubSequence(kl, cfg, validation_records)

    # Setup early stoppage callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=cfg.EARLY_STOP_PATIENCE),
        ModelCheckpoint(
            filepath=output_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        )
    ]

    kl.model.fit_generator(
        generator=training,
        steps_per_epoch=len(training),
        callbacks=callbacks,
        validation_data=validation,
        validation_steps=len(validation),
        epochs=cfg.MAX_EPOCHS,
        verbose=cfg.VERBOSE_TRAIN,
        workers=1,
        use_multiprocessing=False
    )


def main():
>>>>>>> Add a new datastore format.
    args = docopt(__doc__)
    cfg = donkeycar.load_config()
    tubs = args['--tubs']
    model = args['--model']
    model_type = args['--type']
<<<<<<< HEAD

    if model_type is None:
        model_type = cfg.DEFAULT_MODEL_TYPE
        print("using default model type of", model_type)

    if args['--figure_format']:
        figure_format = args['--figure_format']
    continuous = args['--continuous']
    aug = args['--aug']
    
    dirs = preprocessFileList( args['--file'] )
    if tub is not None:
        tub_paths = [os.path.expanduser(n) for n in tub.split(',')]
        dirs.extend( tub_paths )
=======
>>>>>>> Add a new datastore format.

    if not model_type:
        model_type = cfg.DEFAULT_MODEL_TYPE

    tubs = tubs.split(',')
    data_paths = [Path(os.path.expanduser(tub)).absolute().as_posix() for tub in tubs]
    output_path = os.path.expanduser(model)
    train(cfg, data_paths, output_path, model_type)

if __name__ == "__main__":
    main()
