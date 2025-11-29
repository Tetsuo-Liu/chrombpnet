import numpy as np ;
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Input, Cropping1D, add, Conv1D, GlobalAvgPool1D, Dense, Add, Concatenate, Lambda, Flatten, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from chrombpnet.training.utils.losses import multinomial_nll
import tensorflow as tf
import random as rn
import os 

os.environ['PYTHONHASHSEED'] = '0'


def load_pretrained_bias(model_hdf5):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import get_custom_objects
    custom_objects={"multinomial_nll":multinomial_nll, "tf":tf}    
    get_custom_objects().update(custom_objects)
    pretrained_bias_model=load_model(model_hdf5)
    #freeze the model
    num_layers=len(pretrained_bias_model.layers)
    for i in range(num_layers):
        pretrained_bias_model.layers[i].trainable=False
    return pretrained_bias_model


def bpnet_model(filters, n_dil_layers, sequence_len, out_pred_len):

    conv1_kernel_size=21
    profile_kernel_size=75
    num_tasks=1 # not using multi tasking

    #define inputs
    inp = Input(shape=(sequence_len, 4),name='sequence')    

    # first convolution without dilation
    x = Conv1D(filters,
                kernel_size=conv1_kernel_size,
                padding='valid', 
                activation='relu',
                name='wo_bias_bpnet_1st_conv')(inp)

    layer_names = [str(i) for i in range(1,n_dil_layers+1)]
    for i in range(1, n_dil_layers + 1):
        # dilated convolution
        conv_layer_name = 'wo_bias_bpnet_{}conv'.format(layer_names[i-1])
        conv_x = Conv1D(filters, 
                        kernel_size=3, 
                        padding='valid',
                        activation='relu', 
                        dilation_rate=2**i,
                        name=conv_layer_name)(x)

        x_len = int_shape(x)[1]
        conv_x_len = int_shape(conv_x)[1]
        assert((x_len - conv_x_len) % 2 == 0) # Necessary for symmetric cropping

        x = Cropping1D((x_len - conv_x_len) // 2, name="wo_bias_bpnet_{}crop".format(layer_names[i-1]))(x)
        x = add([conv_x, x])

    # Branch 1. Profile prediction
    # Step 1.1 - 1D convolution with a very large kernel
    prof_out_precrop = Conv1D(filters=num_tasks,
                        kernel_size=profile_kernel_size,
                        padding='valid',
                        name='wo_bias_bpnet_prof_out_precrop')(x)

    # Step 1.2 - Crop to match size of the required output size
    cropsize = int(int_shape(prof_out_precrop)[1]/2)-int(out_pred_len/2)
    assert cropsize>=0
    assert (int_shape(prof_out_precrop)[1] % 2 == 0) # Necessary for symmetric cropping

    prof = Cropping1D(cropsize,
                name='wo_bias_bpnet_logitt_before_flatten')(prof_out_precrop)
    
    profile_out = Flatten(name="wo_bias_bpnet_logits_profile_predictions")(prof)

    # Branch 2. Counts prediction
    # Step 2.1 - Global average pooling along the "length", the result
    #            size is same as "filters" parameter to the BPNet function
    gap_combined_conv = GlobalAvgPool1D(name='gap')(x) # acronym - gapcc

    # Step 2.3 Dense layer to predict final counts
    count_out = Dense(num_tasks, name="wo_bias_bpnet_logcount_predictions")(gap_combined_conv)

    # instantiate keras Model with inputs and outputs
    model=Model(inputs=[inp],outputs=[profile_out, count_out], name="model_wo_bias")

    return model


def getModelGivenModelOptionsAndWeightInits(args, model_params):   
    
    assert("bias_model_path" in model_params.keys()) # bias model path not specfied for model
    filters=int(model_params['filters'])
    n_dil_layers=int(model_params['n_dil_layers'])
    counts_loss_weight=float(model_params['counts_loss_weight'])
    bias_model_path=model_params['bias_model_path']
    sequence_len=int(model_params['inputlen'])
    out_pred_len=int(model_params['outputlen'])

    # Check if dynamic scaling is enabled (for celltype_aggregate generator)
    # This is determined by checking if data_generator_type is 'celltype_aggregate'
    use_dynamic_scaling = hasattr(args, 'data_generator_type') and args.data_generator_type == 'celltype_aggregate'
    
    # Check if multitask learning is enabled (for multitask_celltype generator)
    use_multitask = hasattr(args, 'data_generator_type') and args.data_generator_type == 'multitask_celltype'
    
    # For multitask learning, celltype list is required
    if use_multitask:
        assert('celltypes' in model_params.keys()), "celltypes list not specified for multitask model"
        celltype_list = model_params['celltypes']
        if isinstance(celltype_list, str):
            # If it's a string, try to parse it (e.g., comma-separated or JSON)
            import json
            try:
                celltype_list = json.loads(celltype_list)
            except:
                celltype_list = [ct.strip() for ct in celltype_list.split(',')]
        assert(isinstance(celltype_list, list) and len(celltype_list) > 0), "celltypes must be a non-empty list"

    bias_model = load_pretrained_bias(bias_model_path)
    bpnet_model_wo_bias = bpnet_model(filters, n_dil_layers, sequence_len, out_pred_len)

    #read in arguments
    seed=args.seed
    np.random.seed(seed)    
    tf.random.set_seed(seed)
    rn.seed(seed)
    
    # Define inputs: sequence is always required, scaling_factor is optional
    inp_seq = Input(shape=(sequence_len, 4), name='sequence')
    
    if use_multitask:
        # Multitask learning: shared encoder with cell-type-specific decoder heads
        # Requires dynamic scaling (2-input model)
        inp_scale = Input(shape=(1,), name='scaling_factor')
        
        # Get shared encoder output (once)
        output_wo_bias = bpnet_model_wo_bias(inp_seq)
        
        # Get bias output and apply dynamic scaling
        bias_output = bias_model(inp_seq)
        scaled_bias_profile = Multiply(name="scaled_bias_profile_logits")([bias_output[0], inp_scale])
        scaled_bias_logcounts = Lambda(
            lambda x: x[0] + tf.math.log(x[1] + 1e-6),
            name="scaled_bias_logcounts"
        )([bias_output[1], inp_scale])
        
        # Create decoder heads for each cell type
        profile_outputs = []
        count_outputs = []
        
        # Access shared encoder's intermediate layers for decoder heads
        # Get the layer before profile/count predictions (the dilated conv output)
        # This is the input to the GAP layer (for counts) and the profile Conv1D layer
        shared_encoder_intermediate = bpnet_model_wo_bias.get_layer('gap').input
        
        # Create profile and count decoder heads for each cell type
        for cell_type in celltype_list:
            # Profile decoder head: Conv1D -> Crop -> Flatten -> Add bias
            # Use the same structure as bpnet_model but with cell-type-specific names
            prof_out_precrop = Conv1D(filters=1,
                                    kernel_size=75,
                                    padding='valid',
                                    name=f'prof_out_precrop_{cell_type}')(shared_encoder_intermediate)
            
            # Crop to match output length
            prof_out_precrop_shape = int_shape(prof_out_precrop)
            cropsize = int(prof_out_precrop_shape[1]/2) - int(out_pred_len/2)
            assert cropsize >= 0
            assert (prof_out_precrop_shape[1] % 2 == 0)  # Necessary for symmetric cropping
            
            prof_cropped = Cropping1D(cropsize,
                                    name=f'prof_crop_{cell_type}')(prof_out_precrop)
            prof_flattened = Flatten(name=f'prof_flatten_{cell_type}')(prof_cropped)
            
            # Add scaled bias (each sample has its own scaling_factor, so this works correctly)
            final_prof = Add(name=f'logits_profile_{cell_type}')([prof_flattened, scaled_bias_profile])
            profile_outputs.append(final_prof)
            
            # Count decoder head: Dense -> Add bias
            # Use GAP output from shared encoder
            gap_output = bpnet_model_wo_bias.get_layer('gap').output
            count_head = Dense(1, name=f'logcount_pred_{cell_type}')(gap_output)
            concat_counts = Concatenate(axis=-1)([count_head, scaled_bias_logcounts])
            final_count = Lambda(lambda x: tf.math.reduce_logsumexp(x, axis=-1, keepdims=True),
                                name=f'logcount_{cell_type}')(concat_counts)
            count_outputs.append(final_count)
        
        # Create model with all outputs
        all_outputs = profile_outputs + count_outputs
        model = Model(inputs=[inp_seq, inp_scale], outputs=all_outputs)
        
        # Define loss functions and loss weights as dictionaries
        loss_dict = {}
        loss_weights_dict = {}
        for cell_type in celltype_list:
            loss_dict[f'logits_profile_{cell_type}'] = multinomial_nll
            loss_dict[f'logcount_{cell_type}'] = 'mse'
            loss_weights_dict[f'logits_profile_{cell_type}'] = 1.0
            loss_weights_dict[f'logcount_{cell_type}'] = counts_loss_weight
        
        model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                     loss=loss_dict,
                     loss_weights=loss_weights_dict)
        
        return model
    
    elif use_dynamic_scaling:
        # For celltype_aggregate generator: 2-input model with dynamic bias scaling
        inp_scale = Input(shape=(1,), name='scaling_factor')
        
        ## get bias output (using sequence input)
        bias_output = bias_model(inp_seq)
        ## get wo bias output (using sequence input)
        output_wo_bias = bpnet_model_wo_bias(inp_seq)
        
        assert(len(bias_output[1].shape)==2) # bias model counts head is of incorrect shape (None,1) expected
        assert(len(bias_output[0].shape)==2) # bias model profile head is of incorrect shape (None,out_pred_len) expected
        assert(len(output_wo_bias[0].shape)==2)
        assert(len(output_wo_bias[1].shape)==2)
        assert(bias_output[1].shape[1]==1) #  bias model counts head is of incorrect shape (None,1) expected
        assert(bias_output[0].shape[1]==out_pred_len) # bias model profile head is of incorrect shape (None,out_pred_len) expected
        
        # Apply dynamic scaling to bias outputs
        # Profile: scale in linear space (multiply)
        scaled_bias_profile = Multiply(name="scaled_bias_profile_logits")([bias_output[0], inp_scale])
        # Counts: scale in log space (add log of scaling factor)
        scaled_bias_logcounts = Lambda(
            lambda x: x[0] + tf.math.log(x[1] + 1e-6),
            name="scaled_bias_logcounts"
        )([bias_output[1], inp_scale])
        
        # Combine scaled bias with TF model outputs
        profile_out = Add(name="logits_profile_predictions")([output_wo_bias[0], scaled_bias_profile])
        concat_counts = Concatenate(axis=-1)([output_wo_bias[1], scaled_bias_logcounts])
        count_out = Lambda(lambda x: tf.math.reduce_logsumexp(x, axis=-1, keepdims=True),
                            name="logcount_predictions")(concat_counts)
        
        # Instantiate keras Model with 2 inputs
        model = Model(inputs=[inp_seq, inp_scale], outputs=[profile_out, count_out])
    else:
        # For standard/weighted_dynamic generators: 1-input model (backward compatible)
        inp = inp_seq  # Use same input variable name for compatibility
        
        ## get bias output
        bias_output = bias_model(inp)
        ## get wo bias output
        output_wo_bias = bpnet_model_wo_bias(inp)
        assert(len(bias_output[1].shape)==2) # bias model counts head is of incorrect shape (None,1) expected
        assert(len(bias_output[0].shape)==2) # bias model profile head is of incorrect shape (None,out_pred_len) expected
        assert(len(output_wo_bias[0].shape)==2)
        assert(len(output_wo_bias[1].shape)==2)
        assert(bias_output[1].shape[1]==1) #  bias model counts head is of incorrect shape (None,1) expected
        assert(bias_output[0].shape[1]==out_pred_len) # bias model profile head is of incorrect shape (None,out_pred_len) expected

        # Standard combination without scaling
        profile_out = Add(name="logits_profile_predictions")([output_wo_bias[0], bias_output[0]])
        concat_counts = Concatenate(axis=-1)([output_wo_bias[1], bias_output[1]])
        count_out = Lambda(lambda x: tf.math.reduce_logsumexp(x, axis=-1, keepdims=True),
                            name="logcount_predictions")(concat_counts)

        # Instantiate keras Model with 1 input (backward compatible)
        model = Model(inputs=[inp], outputs=[profile_out, count_out])

    model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                    loss=[multinomial_nll,'mse'],
                    loss_weights=[1,counts_loss_weight])

    return model 


def save_model_without_bias(model, output_prefix):
    """
    Extract and save the TF model without bias component.
    Works for both 1-input and 2-input models.
    For 2-input models, only the sequence input is used (scaling_factor is not needed for TF-only predictions).
    """
    model_wo_bias = model.get_layer("model_wo_bias").output
    #counts_output_without_bias = model.get_layer("wo_bias_bpnet_logcount_predictions").output
    
    # Get inputs from model_wo_bias (always single input: sequence only)
    # This works for both 1-input and 2-input parent models
    model_without_bias = Model(inputs=model.get_layer("model_wo_bias").inputs, outputs=[model_wo_bias[0], model_wo_bias[1]])
    print('save model without bias') 
    model_without_bias.save(output_prefix+"_nobias.h5")
