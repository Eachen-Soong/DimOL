import h5py
import numpy as np

def merge_dataset(f_in, f_out, chunk_size=20):
    # Copy 'train' datasets if not exist in output
    for data_name in f_in['train']:
        if data_name not in f_out:
            data = f_in['train'][data_name]
            # Create chunked dataset
            f_out.create_dataset(data_name, data=data, chunks=(min(chunk_size, data.shape[0]),) + data.shape[1:])

    for group_name in ['test', 'valid']:
        for data_name in f_in[group_name]:
            ds_in = f_in[group_name][data_name]
            if data_name in f_out:
                # Get existing dataset
                ds_out = f_out[data_name]
                # Determine the name for a temporary dataset
                temp_name = data_name + "_temp"
                # If a temporary dataset already exists, delete it
                if temp_name in f_out:
                    del f_out[temp_name]
                # Create a new dataset with chunking and expanded size
                new_shape = (ds_out.shape[0] + ds_in.shape[0],) + ds_out.shape[1:]
                chunked_shape = (min(chunk_size, new_shape[0]),) + new_shape[1:]
                temp_ds = f_out.create_dataset(temp_name, new_shape, chunks=chunked_shape)
                # Copy old data to new dataset
                temp_ds[:ds_out.shape[0]] = ds_out[:]
                # Append new data in chunks
                for i in range(0, ds_in.shape[0], chunk_size):
                    chunk = ds_in[i:i + chunk_size]
                    temp_ds[ds_out.shape[0] + i:ds_out.shape[0] + i + chunk.shape[0]] = chunk
                # Delete the old dataset
                del f_out[data_name]
                # Create a new dataset with the original name and copy the data from the temporary dataset
                f_out.create_dataset(data_name, data=temp_ds, chunks=chunked_shape)
                # Delete the temporary dataset
                del f_out[temp_name]
            else:
                # Create new dataset if not exist
                f_out.create_dataset(data_name, data=ds_in, chunks=(min(chunk_size, ds_in.shape[0]),) + ds_in.shape[1:])

input_file = "./neuralop/datasets/data/ns_random_forces_mini.h5"
output_file = "./neuralop/datasets/data/ns_random_forces_1.h5"
with h5py.File(input_file, 'r') as f_in:
    with h5py.File(output_file, 'a') as f_out:
        merge_dataset(f_in, f_out)

        # Read the data into memory
        a_data = f_out['a'][:]
        u_data = f_out['u'][:]

        # Process the data
        expanded_a_data = np.expand_dims(a_data, axis=-1)
        concatenated_data = np.concatenate((expanded_a_data, u_data), axis=-1)

        # Replace the 'u' dataset
        del f_out['u']  # Delete the existing 'u' dataset
        f_out.create_dataset('u', data=concatenated_data)

        # Optionally, delete the 'a' dataset if no longer needed
        del f_out['a']
