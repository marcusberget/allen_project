from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
from PIL import Image
import imageio

boc = BrainObservatoryCache(manifest_file='boc/manifest.json')

data_set = boc.get_ophys_experiment_data(501498760)
time, raw_traces = data_set.get_fluorescence_traces()
_, dff_traces = data_set.get_dff_traces()
# _, running_speed = data_set.get_running_speed()

# the natural scenes stimulus table describes when each scene is on the screen
stim_table = data_set.get_stimulus_table('natural_scenes')
# read in the array of images
scenes = data_set.get_stimulus_template('natural_scenes')

events = np.load('events/ophys_events.npy')
# events = dff_traces
# scenes = np.repeat(scenes, 50, axis=0)
num_images = stim_table['frame'].max() - stim_table['frame'].min() + 1 #119
num_neurons = events.shape[0] # Varies
num_times_shown = 50
timesteps_shown = 7
num_responses_per_image = int(num_images * num_times_shown) #5950

use_pupil_data = True

def set_up_data(stim_table, events, pupil_size, pupil_location, use_pupil_data):
    scene0_events = np.zeros((num_neurons, num_times_shown))
    max_events = np.zeros((num_neurons, int(num_images * num_times_shown)))
    im_labels = np.array([], dtype=str)

    if use_pupil_data:
            pupil_size_shortened = np.zeros(5950)
            pupil_location_shortened = np.zeros((5950, 2))

    k=0
    for i, trial in stim_table.iterrows():
        scene = trial['frame']
        start = trial.start
        end = trial.end
        # if (end - start) != 7:
        #     end = start + 7
        pupil_size_shortened[i] = np.mean(pupil_size[start:end])
        print(np.mean(pupil_location, axis=1))
        quit()
        pupil_location_shortened[i, :] = np.mean(pupil_location, axis=1)
        max_events[:, i] = np.max(events[:, start:end] , axis=1)
        if scene == 4:
            scene0_events[:, k] = np.max(events[:, start:end] , axis=1)
            k += 1

        im_labels = np.append(im_labels, f'{scene}')
    return max_events, im_labels, scene0_events

if use_pupil_data:
    _, pupil_size = data_set.get_pupil_size()
    _, pupil_location = data_set.get_pupil_location()
    responses, pupil_size_shortened, pupil_location_shortened = set_up_data(stim_table, dff_traces, pupil_size, pupil_location, use_pupil_data=use_pupil_data)
    np.save('data/ophys_pupil_size', pupil_size_shortened)
    np.save('data/ophys_pupil_location', pupil_location_shortened)
if not use_pupil_data:
    pupil_size = None
    pupil_location = None
    max_events, im_labels, scene0_events = set_up_data(stim_table, events)#, pupil_size, pupil_location, use_pupil_data=use_pupil_data)
    np.save('data/ophys_dff_traces', max_events)
    np.save('data/ophys_im_labels', im_labels)
    np.save('data/ophys_scene0_events.npy', scene0_events)
#     cnt_array = np.zeros((num_images, num_times_shown), dtype=int)
#     # fstart = stim_table.start.min()
#     # fend = stim_table.end.max()
#     if use_pupil_data:
#         pupil_size_shortened = np.zeros(5950*7)
#         pupil_location_shortened = np.zeros((5950*7, 2))
#     # running_speed_shortened = np.array([])
#     # cnt = 0
#     k = 0
#     cnt_array = np.ones(num_images, dtype=int)*-1
#     for i, trial in stim_table.iterrows():
#         scene = trial['frame']
#         start = trial.start
#         end = trial.end
#         if (end - start) != 7:
#             end = start + 7
#         # cnt_array[scene] += 1

#         max_event = np.max(events[j][start:end])

#         responses[j][scene][cnt_array[scene]] = max_event
#         if use_pupil_data:
#             if j == 0:
#                 pupil_size_shortened[k:(k+7)] = pupil_size[start:end]
#                 pupil_location_shortened[k:(k+7)] = pupil_location[start:end]
#                 k += 7
#             # running_speed_shortened = np.append(running_speed_shortened, running_speed[start:end])


#     # responses = np.mean(responses, axis=3)
#     # responses = responses[:, 0:num_images, :]
#     # responses = responses.reshape((num_responses_per_image, num_neurons))
#     np.save('data/max_events.npy', responses)

#     if use_pupil_data:
#         reshaped_pupil_size = pupil_size_shortened[:len(pupil_size_shortened)//7 * 7].reshape(-1, 7)
#         mean_pupil_size = np.mean(reshaped_pupil_size, axis=1)

#         reshaped_pupil_location = pupil_location_shortened[:pupil_location_shortened.shape[0]].reshape(5950, 7, 2)
#         mean_pupil_location = np.mean(reshaped_pupil_location, axis=1)
#         return responses, mean_pupil_size, mean_pupil_location
    
#     if not use_pupil_data:
#         return responses


# if not use_pupil_data:
#     pupil_size = None
#     pupil_location = None
#     responses = set_up_data(stim_table, events, pupil_size, pupil_location, use_pupil_data=use_pupil_data)
# # for i in range(-1, 118):
# #     img = Image.fromarray(scenes[i, :, :])
# #     if img.mode != 'RGB':
# #         img = img.convert('RGB')
# #     img.save(f'scenes/scene_{i}.jpeg')


# # np.save('data/ophys_events.npy', responses)
# # np.save('ophys_pupil_size.npy', pupil_size_shortened)
# # np.save('ophys_pupil_location.npy', pupil_location_shortened)
# # np.save('ophys_running_speed.npy', running_speed_shortened)