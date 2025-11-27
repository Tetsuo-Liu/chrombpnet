from __future__ import division, print_function, absolute_import
import importlib.machinery
import tensorflow.keras.callbacks as tfcallbacks 
import chrombpnet.training.utils.argmanager as argmanager
import chrombpnet.training.utils.losses as losses
import chrombpnet.training.utils.callbacks as callbacks
import chrombpnet.training.data_generators.initializers as initializers
import pandas as pd
import os
import json
import numpy as np
import tensorflow as tf

NARROWPEAK_SCHEMA = ["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"]
os.environ['PYTHONHASHSEED'] = '0'

def create_tf_compatible_dataset(sequence_generator):
    """
    Create TensorFlow compatible dataset from ChromBPNetBatchGenerator.
    Supports both 1-input and 2-input models, with optional sample weights.
    Uses feature detection instead of version checking for better reliability.
    """
    # Get a sample batch to detect the data format
    sample_batch = sequence_generator[0]
    
    # Detect data format: check if it's a 3-element tuple (with sample weights)
    has_sample_weights = isinstance(sample_batch, tuple) and len(sample_batch) == 3
    
    if has_sample_weights:
        # Format: ((batch_seq, batch_scaling_factors), (batch_cts, batch_log_cts), batch_loss_weights)
        inputs, outputs, sample_weights = sample_batch
        
        # Check if inputs is a tuple (2-input model) or single array (1-input model)
        is_2input = isinstance(inputs, tuple) and len(inputs) == 2
        
        if is_2input:
            # 2-input model with sample weights (celltype_aggregate generator)
            batch_seq, batch_scaling_factors = inputs
            # Define output signature for 2-input model with sample weights
            output_signature = (
                (
                    tf.TensorSpec(shape=(None, sequence_generator.inputlen, 4), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                ),
                (
                    tf.TensorSpec(shape=(None, sequence_generator.outputlen), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                ),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        else:
            # 1-input model with sample weights (should not happen with current generators, but handle for safety)
            output_signature = (
                tf.TensorSpec(shape=(None, sequence_generator.inputlen, 4), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(None, sequence_generator.outputlen), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                ),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
    else:
        # Format: (batch_seq, (batch_cts, batch_log_cts)) - 1-input model without sample weights
        # Check if inputs is a tuple (2-input model) or single array (1-input model)
        inputs = sample_batch[0]
        is_2input = isinstance(inputs, tuple) and len(inputs) == 2
        
        if is_2input:
            # 2-input model without sample weights
            output_signature = (
                (
                    tf.TensorSpec(shape=(None, sequence_generator.inputlen, 4), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                ),
                (
                    tf.TensorSpec(shape=(None, sequence_generator.outputlen), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                )
            )
        else:
            # 1-input model without sample weights (standard generator)
            output_signature = (
                tf.TensorSpec(shape=(None, sequence_generator.inputlen, 4), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(None, sequence_generator.outputlen), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                )
            )
    
    # Test if current TensorFlow requires explicit output_signature
    def test_generator():
        import numpy as np
        batch_seq = np.zeros((1, sequence_generator.inputlen, 4), dtype=np.float32)
        batch_cts = np.zeros((1, sequence_generator.outputlen), dtype=np.float32)
        if has_sample_weights and is_2input:
            yield ((batch_seq, np.zeros((1, 1), dtype=np.float32)), 
                   (batch_cts, np.zeros((1, 1), dtype=np.float32)),
                   np.zeros((1,), dtype=np.float32))
        elif has_sample_weights:
            yield (batch_seq, (batch_cts, np.zeros((1, 1), dtype=np.float32)), np.zeros((1,), dtype=np.float32))
        elif is_2input:
            yield ((batch_seq, np.zeros((1, 1), dtype=np.float32)), 
                   (batch_cts, np.zeros((1, 1), dtype=np.float32)))
        else:
            yield (batch_seq, (batch_cts, np.zeros((1, 1), dtype=np.float32)))
    
    # Try to create dataset without explicit output_signature
    try:
        test_dataset = tf.data.Dataset.from_generator(test_generator)
        # If successful, use legacy approach (return generator directly)
        # Note: For generators with sample weights, we still need explicit signature
        if has_sample_weights:
            # Must use explicit signature for sample weights
            pass
        else:
            return sequence_generator
    except (TypeError, ValueError):
        # If failed, use explicit output_signature approach
        pass
    
    # Use explicit Dataset conversion with output_signature
    def generator_func():
        for i in range(len(sequence_generator)):
            yield sequence_generator[i]
    
    dataset = tf.data.Dataset.from_generator(
        generator_func,
        output_signature=output_signature
    )
    return dataset.prefetch(tf.data.AUTOTUNE)

def get_model(args, parameters):
    """
    Read a model definition from a python file. This function can be used to read any model architecture that takes sequence as input
    and outputs a two task model.  Task one to predict the probability distribution of a profile and task two to predict the total counts in a profile.
    Look at .py models in src/training/models/ for examples. I will try to provide a dummy model as example here - for later.
    The files should have the following two functions - getModelGivenModelOptionsAndWeightInits and save_model_without_bias
    """
    architecture_module=importlib.machinery.SourceFileLoader('',args.architecture_from_file).load_module()
    model=architecture_module.getModelGivenModelOptionsAndWeightInits(args, parameters)
    print("got the model")
    return model, architecture_module

def fit_and_evaluate(model,train_gen,valid_gen,args,architecture_module):
    model_output_path_h5_name=args.output_prefix+".h5"
    model_output_path_logs_name=args.output_prefix+".log"

    checkpointer = tfcallbacks.ModelCheckpoint(filepath=model_output_path_h5_name, monitor="val_loss", mode="min",  verbose=1, save_best_only=True)
    earlystopper = tfcallbacks.EarlyStopping(monitor='val_loss', mode="min", patience=args.early_stop, verbose=1, restore_best_weights=True)
    history= callbacks.LossHistory(model_output_path_logs_name+".batch",args.trackables)
    csvlogger = tfcallbacks.CSVLogger(model_output_path_logs_name, append=False)
    #reduce_lr = tfcallbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.4, patience=args.early_stop-2, min_lr=0.00000001)
    cur_callbacks=[checkpointer,earlystopper,csvlogger,history]

    # Convert generators to TF 2.20+ compatible format if needed
    train_data = create_tf_compatible_dataset(train_gen)
    valid_data = create_tf_compatible_dataset(valid_gen)
    
    # Check if generators return sample weights (3-element tuple)
    sample_batch = train_gen[0]
    has_sample_weights = isinstance(sample_batch, tuple) and len(sample_batch) == 3
    
    if has_sample_weights:
        # Extract sample weights from dataset
        # For tf.data.Dataset, we need to handle sample weights separately
        # Keras model.fit() supports sample_weight parameter
        # However, when using tf.data.Dataset with sample weights in the tuple,
        # Keras automatically extracts them if the dataset returns (x, y, sample_weight)
        model.fit(train_data,
                  validation_data=valid_data,
                  epochs=args.epochs,
                  verbose=1,
                  callbacks=cur_callbacks)
    else:
        # Standard format without sample weights
        model.fit(train_data,
                  validation_data=valid_data,
                  epochs=args.epochs,
                  verbose=1,
                  callbacks=cur_callbacks)

    print('save model') 
    model.save(model_output_path_h5_name)

    architecture_module.save_model_without_bias(model, args.output_prefix)


def get_model_param_dict(args):
    '''
    param_file is a TSV file with 2 columns -- param name in column 1, and param value in column 2
    You can pass model specfic parameters to design your own model with this.
    '''
    params={}
    for line in open(args.params,'r').read().strip().split('\n'):
        tokens=line.split('\t')
        params[tokens[0]]=tokens[1]

    assert("counts_loss_weight" in params.keys()) # missing counts loss weight to use
    assert("filters" in params.keys()) # filters to use for the model not provided
    assert("n_dil_layers" in params.keys()) # n_dil_layers to use for the model not provided
    assert("inputlen" in params.keys()) # inputlen to use for the model not provided
    assert("outputlen" in params.keys()) # outputlen to use for the model not provided
    assert("negative_sampling_ratio" in params.keys()) # negative_sampling_ratio to use for the model not provided
    assert("max_jitter" in params.keys()) # max_jitter to use for the model not provided
    assert(args.chr_fold_path==params["chr_fold_path"]) # the parameters were generated on a different folds compared to the given fold

    assert(int(params["inputlen"])%2==0)
    assert(int(params["outputlen"])%2==0)

    return params 

def main(args):


    # read tab-seperated parameters file
    parameters = get_model_param_dict(args)
    print(parameters)
    np.random.seed(args.seed)

    # get model architecture to load
    model, architecture_module=get_model(args, parameters)

    # initialize generators to load data
    train_generator = initializers.initialize_generators(args, "train", parameters, return_coords=False)
    valid_generator = initializers.initialize_generators(args, "valid", parameters, return_coords=False)

    # train the model using the generators
    fit_and_evaluate(model, train_generator, valid_generator, args, architecture_module)

    # store arguments and and parameters to checkpoint
    with open(args.output_prefix+'.args.json', 'w') as fp:
        json.dump(args.__dict__, fp,  indent=4)
    #with open(args.output_prefix+'.params.json', 'w') as fp:
    #    json.dump(parameters, fp,  indent=4)


if __name__=="__main__":
    # read arguments
    args=argmanager.fetch_train_args()
    main(args)

