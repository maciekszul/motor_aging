import sys
import time
import utils
import numpy as np
from copy import copy
from pathlib import Path
from mne import set_log_level
from meegkit.dss import dss_line_iter
from mne.io import (
    read_raw_fif,
    RawArray
)
from mne.preprocessing import ICA
from mne import (
    find_events,
    annotations_from_events
)

set_log_level("ERROR")

# access to the paths and settings
settings = utils.load_json("settings.json")
trigger_mapping = utils.load_json("trigger_mapping.json")
trigger_mapping = {int(i): trigger_mapping[i] for i in trigger_mapping.keys()}


dataset_path = Path(settings["dataset_path"])

exclude_sub = settings["exclude_sub"]

subject_dirs = sorted(dataset_path.glob("sub-*"))

task_raw_path = "meg/sensorimotor/ses-smt/meg"
task_maxwell_path = "meg/sensorimotor_maxwell"
rest_raw_path = "meg/resting_state/ses-rest/meg"
rest_maxwell_path = "meg/resting_state_maxwell"

all_files = list(
    dataset_path.glob(f"sub-*/{task_raw_path}/*_task-smt_meg.fif")
)

proc_path = dataset_path.joinpath("processed")

all_files = [
    i
    for i in all_files
    if not any(sub in i.parts for sub in exclude_sub)
]

print(f"Found {len(all_files)} participant files to process.")

# function that can also be accessed by importing this file
def process_ds(ds, trigger_mapping, proc_path, block_type=0):

    subject = ds.stem.split("_")[0]
    subject_path = utils.make_directory(proc_path, f"{subject}")

    filename = f"{subject}_task-smt_proc-clean_raw.fif"
    raw_output = subject_path.joinpath(filename)
    filename = f"{subject}_task-smt_ica.fif"
    ica_output = subject_path.joinpath(filename)
    
    raw = read_raw_fif(ds, preload=True, verbose=False)
    
    raw = raw.filter(None, 125)

    set_ch = {
    "STI101": "stim",
    "EOG061": "eog",
    "EOG062": "eog",
    "ECG063": "ecg",
    }

    raw = raw.set_channel_types(set_ch)

    events = find_events(raw, "STI101", min_duration=0.003)
    print(events)

    # zapline

    raw_data = raw.get_data()
    raw_info = raw.info
    first_samp = raw.first_samp
    mag_ix = np.array([i for i, lab in enumerate(raw.get_channel_types()) if lab == "mag"])

    rd = np.array_split(raw_data[mag_ix], 10, axis=1)
    rd = [np.moveaxis(i, [0,1], [1,0]) for i in rd]
    rd_f = []

    for ix, ss in enumerate(rd):
        print(ds.name, f"{ix+1}/10")
        data, iters = dss_line_iter(
            ss, fline=50.0, sfreq=raw_info["sfreq"], 
            spot_sz=5.5, win_sz=10, nfft=1024, n_iter_max=30
        )
        rd_f.append(data)
    del rd
    
    rd_f = [np.moveaxis(i, [0,1], [1,0]) for i in rd_f]
    rd_f = np.hstack(rd_f)

    new_raw_data = copy(raw_data)
    del raw_data
    new_raw_data[mag_ix, :] = rd_f

    new_raw = RawArray(
        new_raw_data,
        raw_info,
        first_samp=first_samp
    )

    del raw

    # Set trial annotations from event triggers
    afe = annotations_from_events(

        events, 
        new_raw.info["sfreq"], 
        trigger_mapping, 
        new_raw.first_samp
        )

    new_raw = new_raw.set_annotations(afe)
    
    new_raw.save(raw_output, fmt="single", overwrite=True, verbose=False)
    
    new_raw = new_raw.filter(1, 40)
    ica = ICA(n_components=20, random_state=55)
    ica.fit(new_raw)
    ica.save(ica_output, overwrite=True, verbose=False)

if __name__ == '__main__':
    try:
        index = int(sys.argv[1])
    except:
        raise IndexError("no file index")
    
    ds = all_files[index]
    
    start_time = time.time()

    status = "START"
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(all_files)}"
    print(to_print)
    
    process_ds(ds, trigger_mapping, proc_path)

    status = "END"
    time_elapsed =  np.round((time.time() - start_time)/60, 2)
    to_print = f"{status} File:{str(index+1).zfill(3)}/{len(all_files)} Time elapsed: {time_elapsed} min"
    print(to_print)