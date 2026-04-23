# Imitation Dataset Format

The raw dataset is one folder per real-robot demonstration episode.

```text
episode_000001/
  meta.json
  success.json
  timestamps.npy
  joint_state_timestamps.npy
  joint_pos.npy
  joint_vel.npy
  actions.npy
  action_valid.npy
  gripper.npy
  rgb_head/
    frame_000001.jpg
    frame_000002.jpg
  rgb_head_timestamps.npy
  rgb_head_ros_timestamps.npy
```

## Episode Lifecycle

1. Start the robot stack and keyboard teleop.
2. Call `/demo_recorder/start`.
3. Operate the robot with keyboard teleop.
4. Call `/demo_recorder/stop` with `true` for success or `false` for failure.
5. The recorder writes arrays, images, metadata, and success label.

## Observations

`joint_pos.npy`: shape `(T, 16)`, ordered as:

- left arm 6 joints
- right arm 6 joints
- neck 2 joints
- left and right gripper joints

`joint_vel.npy`: same shape and order as `joint_pos.npy`.

Camera observations are saved independently as JPEG files. Camera timestamp arrays have one timestamp per saved frame. Training code can later associate camera frames to state/action samples by nearest timestamp.

## Actions

`actions.npy`: shape `(T, 16)`.

Action mode: `joint_position_targets`.

- `[0:6]`: latest `/left_joint_command`
- `[6:12]`: latest `/right_joint_command`
- `[12:14]`: latest `/neck_joint_command`
- `[14:16]`: latest left/right gripper open command, encoded as `1.0` open and `0.0` close

At episode start, arm and neck target actions are initialized from the current `/joint_states` so the held keyboard target is explicit. The recorder does not smooth or alter keyboard commands.

`action_valid.npy`: shape `(T, 16)`, boolean. A value is true when the corresponding action component is known.

`gripper.npy`: shape `(T, 2)`, duplicate convenience field for the gripper command/state channel used by v1.

## Time Synchronization

`timestamps.npy` is the recorder sample time from the ROS node clock. State/action arrays are sampled at `sample_rate_hz` using the latest received `/joint_states` and command messages.

Camera frames are not forced to align one-to-one with state/action samples. Each enabled camera writes:

- `<camera_name>_timestamps.npy`: recorder receive/write time
- `<camera_name>_ros_timestamps.npy`: original `sensor_msgs/Image.header.stamp`

This makes missing or delayed frames detectable during validation and conversion.

## Success / Failure

`success.json` contains:

```json
{"success": true}
```

Failed episodes are kept. Converters and training scripts can filter them later.

## Conversion Format

`convert_to_hdf5` creates:

- `dataset.npz`
  - `obs_state`: `(N, 16)`
  - `actions`: `(N, 16)`
  - `episode_index`: `(N,)`
- `dataset.hdf5` if `h5py` is installed
  - `obs/state`
  - `actions`
  - `episode_index`
- `splits.json`
- `manifest.json`

The first baseline training script consumes `dataset.npz`.

## VR Extension

VR can be added later by recording its twist or pose topics as an alternate action mode, for example `ee_twist_delta` or `pose_delta`. The raw schema already records `action_mode` and `action_components` in `meta.json`, so new action definitions can coexist with keyboard joint-target episodes.

Training code should not mix action modes unless a converter explicitly maps them into one representation.
