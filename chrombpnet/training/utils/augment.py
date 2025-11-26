import numpy as np

# https://stackoverflow.com/questions/46091111/python-slice-array-at-different-position-on-every-row
def take_per_row(A, indx, num_elem):
    """
    Matrix A, indx is a vector for each row which specifies 
    slice beginning for that row. Each has width num_elem.
    """

    all_indx = indx[:,None] + np.arange(num_elem)
    return A[np.arange(all_indx.shape[0])[:,None], all_indx]


def random_crop(seqs, labels, seq_crop_width, label_crop_width, coords, rng=None):
    """
    Takes sequences and corresponding counts labels. They should have the same
    #examples. The widths would correspond to inputlen and outputlen respectively,
    and any additional flanking width for jittering which should be the same
    for seqs and labels. Each example is cropped starting at a random offset. 

    seq_crop_width - label_crop_width should be equal to seqs width - labels width,
    essentially implying they should have the same flanking width.
    
    Args:
        seqs: B x IL x 4 array of sequences
        labels: B x OL array of labels
        seq_crop_width: Width to crop sequences to
        label_crop_width: Width to crop labels to
        coords: B x 3 array of coordinates
        rng: numpy.random.RandomState instance. If None, uses global np.random.
    
    Returns:
        Cropped sequences, labels, and updated coordinates
    """

    assert(seqs.shape[1]>=seq_crop_width)
    assert(labels.shape[1]>=label_crop_width)
    assert(seqs.shape[1] - seq_crop_width == labels.shape[1] - label_crop_width)

    if rng is None:
        rng = np.random

    max_start = seqs.shape[1] - seq_crop_width # This should be the same for both input and output

    starts = rng.choice(range(max_start+1), size=seqs.shape[0], replace=True)

    new_coords = coords.copy()
    #new_coords[:,1] = new_coords[:,1].astype(int) - (seqs.shape[1]//2) + starts
    new_coords[:,1] = new_coords[:,1].astype(int) + starts

    return take_per_row(seqs, starts, seq_crop_width), take_per_row(labels, starts, label_crop_width), new_coords

def random_rev_comp(seqs, labels, coords, frac=0.5, rng=None):
    """
    Data augmentation: applies reverse complement randomly to a fraction of 
    sequences and labels.

    Assumes seqs are arranged in ACGT. Then ::-1 gives TGCA which is revcomp.

    NOTE: Performs in-place modification.
    
    Args:
        seqs: B x IL x 4 array of sequences
        labels: B x OL array of labels
        coords: B x 3 array of coordinates
        frac: Fraction of sequences to apply reverse complement to
        rng: numpy.random.RandomState instance. If None, uses global np.random.
    
    Returns:
        Modified seqs, labels, coords (in-place modification)
    """
    if rng is None:
        rng = np.random
    
    pos_to_rc = rng.choice(range(seqs.shape[0]), 
            size=int(seqs.shape[0]*frac),
            replace=False)

    seqs[pos_to_rc] = seqs[pos_to_rc, ::-1, ::-1]
    labels[pos_to_rc] = labels[pos_to_rc, ::-1]
    coords[pos_to_rc,2] =  "r"
	
    return seqs, labels, coords

def crop_revcomp_augment(seqs, labels, coords, seq_crop_width, label_crop_width, add_revcomp, rc_frac=0.5, shuffle=False, rng=None):
    """
    seqs: B x IL x 4
    labels: B x OL

    Applies random crop to seqs and labels and reverse complements rc_frac.
    
    Args:
        seqs: B x IL x 4 array of sequences
        labels: B x OL array of labels
        coords: B x 3 array of coordinates
        seq_crop_width: Width to crop sequences to (currently unused, kept for API compatibility)
        label_crop_width: Width to crop labels to (currently unused, kept for API compatibility)
        add_revcomp: Whether to apply reverse complement augmentation
        rc_frac: Fraction of sequences to apply reverse complement to
        shuffle: Whether to shuffle the data
        rng: numpy.random.RandomState instance. If None, uses global np.random.
    
    Returns:
        Modified copies of seqs, labels, coords (original arrays are not modified)
    """

    assert(seqs.shape[0]==labels.shape[0])
    
    if rng is None:
        rng = np.random

    # Create copies to avoid in-place modification of input arrays
    # This ensures that the original arrays passed to this function are not modified
    # CRITICAL: This prevents unintended side effects when the function is called
    # with arrays that may be used elsewhere (e.g., self.seqs in CelltypeGenerator)
    mod_seqs = seqs.copy()
    mod_labels = labels.copy()
    mod_coords = coords.copy()

    # Apply reverse complement augmentation (modifies copies, not original arrays)
    if add_revcomp:
        mod_seqs, mod_labels, mod_coords = random_rev_comp(mod_seqs, mod_labels, mod_coords, frac=rc_frac, rng=rng)

    # Apply shuffling if requested (creates new arrays with shuffled order)
    if shuffle:
        perm = rng.permutation(mod_seqs.shape[0])
        mod_seqs = mod_seqs[perm]
        mod_labels = mod_labels[perm]
        mod_coords = mod_coords[perm]

    return mod_seqs, mod_labels, mod_coords
