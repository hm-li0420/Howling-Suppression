adptive_filter_conf = {
    # audio configure
    'sample_rates': 16000,
    'win_len': 320,
    'win_inc': 160,
    'fft_len': 480,
    'win_type': 'hann',

    # rir configure
    'room_size': [10, 10, 10],
    'receiver': [3, 5, 1],
    'speaker': [3, 5.05, 1],
    't60': 0.3,
    'rir_length': 512,

    # megaphone configure
    'delay': 4,
    'gain': 0.6,
    'N': 201,

    # af configure
    'M': 64,
    'step': 0.002,
    'leak': 0
}

notch_filter_conf = {
    # audio configure
    'sample_rates': 16000,
    'win_len': 320,
    'win_inc': 160,
    'fft_len': 480,
    'win_type': 'hann',

    # rir configure
    'room_size': [10, 10, 10],
    'receiver': [3, 5, 1],
    'speaker': [3, 5.05, 1],
    't60': 0.3,
    'rir_length': 512,

    # megaphone configure
    'delay': 4,
    'gain': 0.6,
    'N': 201,
}


shift_freq_conf = {
    # audio configure
    'sample_rates': 16000,
    'win_len': 320,
    'win_inc': 160,
    'fft_len': 480,
    'win_type': 'hann',

    # rir configure
    'room_size': [10, 10, 10],
    'receiver': [3, 5, 1],
    'speaker': [3, 5.05, 1],
    't60': 0.3,
    'rir_length': 512,

    # megaphone configure
    'delay': 4,
    'gain': 0.6,
    'N': 201,

    # af configure
    'M': 64,
    'step': 0.002,
    'leak': 0
}