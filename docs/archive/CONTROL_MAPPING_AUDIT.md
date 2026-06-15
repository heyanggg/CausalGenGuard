# CONTROL_MAPPING_AUDIT

Generated from server path: `/home/heyang/projects/CausalGenGuard`
Python: `/home/heyang/miniconda3/envs/smartguard_env/bin/python`
Write scope: this audit report only. No source project files were modified.

## Executive Summary

- SmartGuard numeric id to device/action mapping: **no explicit high-confidence dictionary was found** in sampled SmartGuard or SmartGen mapping candidates.
- SmartGen textual control to device/action mapping: **yes, textual catalogs exist** in `fr_keys_best.txt`, `sp_keys_best.txt`, `us_keys_best.txt`, and textual transition files such as `action_transitions.json`. These expose `Device:action` controls but not numeric ids.
- Shared device/action space: **partial / likely structural overlap**. SmartGuard FR raw samples and SmartGen offline generated sequences both use flat 4-tuples; the third/fourth columns look like numeric device/action ids. The numeric-to-text pairing is still missing.
- Current named SmartGuard-style attacks: **not reliable yet on prepared FR numeric data**. Semantic attacks like camera/lock/window require a numeric id mapper or curated dictionary; numeric fallback remains the only safe automated option.
- Mapping Feasibility Decision: **B. 只能建立部分映射，需要人工补充**.

## Candidate Scan Scope

Requested paths were scanned read-only. The broad filename-keyword scan is intentionally noisy because many generated per-context files include `fr`, `sp`, `us`, or generation keywords.

- Filename-keyword candidates found: **5274**
- SmartGen synthetic/offline candidate files found: **2068**
- High-value mapping candidates sampled in detail: **45**

## Candidate Files Table

| path | type | size | useful_for_mapping | notes |
| --- | --- | --- | --- | --- |
| /home/heyang/projects/SmartGen/SmartGen/fr_keys_best.txt | .txt | 4.3 KB | yes | SmartGen textual device/action catalog; no numeric ids visible |
| /home/heyang/projects/SmartGen/SmartGen/sp_keys_best.txt | .txt | 4.4 KB | yes | SmartGen textual device/action catalog; no numeric ids visible |
| /home/heyang/projects/SmartGen/SmartGen/us_keys_best.txt | .txt | 4.8 KB | yes | SmartGen textual device/action catalog; no numeric ids visible |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/daytime/action_transitions.json | .json | 4.2 KB | yes | Textual Device:action transition hints; no numeric ids visible |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/single/action_transitions.json | .json | 7.1 KB | yes | Textual Device:action transition hints; no numeric ids visible |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/winter/action_transitions.json | .json | 14.4 KB | yes | Textual Device:action transition hints; no numeric ids visible |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/sp/daytime/action_transitions.json | .json | 10.6 KB | yes | Textual Device:action transition hints; no numeric ids visible |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/sp/single/action_transitions.json | .json | 14.6 KB | yes | Textual Device:action transition hints; no numeric ids visible |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/sp/winter/action_transitions.json | .json | 20.1 KB | yes | Textual Device:action transition hints; no numeric ids visible |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/us/daytime/action_transitions.json | .json | 27.9 KB | yes | Textual Device:action transition hints; no numeric ids visible |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/us/single/action_transitions.json | .json | 20.2 KB | yes | Textual Device:action transition hints; no numeric ids visible |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/us/winter/action_transitions.json | .json | 46.7 KB | yes | Textual Device:action transition hints; no numeric ids visible |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/an/trn.pkl | .pkl | 68.1 KB | unknown | Numeric flat 4-tuples; useful for layout and id-space overlap |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/an/winter/trn.pkl | .pkl | 46.1 KB | unknown | Numeric flat 4-tuples; useful for layout and id-space overlap |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/daytime/trn.pkl | .pkl | 4.1 KB | unknown | Numeric flat 4-tuples; useful for layout and id-space overlap |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/trn.pkl | .pkl | 1.4 KB | unknown | Numeric flat 4-tuples; useful for layout and id-space overlap |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/night/trn.pkl | .pkl | 1.1 KB | unknown | Numeric flat 4-tuples; useful for layout and id-space overlap |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/single/trn.pkl | .pkl | 26.7 KB | unknown | Numeric flat 4-tuples; useful for layout and id-space overlap |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/spring/trn.pkl | .pkl | 4.3 KB | unknown | Numeric flat 4-tuples; useful for layout and id-space overlap |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/trn.pkl | .pkl | 33.8 KB | unknown | Numeric flat 4-tuples; useful for layout and id-space overlap |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/winter/trn.pkl | .pkl | 26.2 KB | unknown | Numeric flat 4-tuples; useful for layout and id-space overlap |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/sp/daytime/trn.pkl | .pkl | 16.7 KB | unknown | Numeric flat 4-tuples; useful for layout and id-space overlap |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/sp/night/trn.pkl | .pkl | 3.8 KB | unknown | Numeric flat 4-tuples; useful for layout and id-space overlap |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/sp/single/trn.pkl | .pkl | 74.5 KB | unknown | Numeric flat 4-tuples; useful for layout and id-space overlap |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 1.9 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 3.8 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_0_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 1.7 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_0_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.6 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_1_0_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.7 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_1_1_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.2 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_1_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 1.9 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_2_0_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.0 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_2_1_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.4 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_2_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 1.7 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_3_0_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 1.9 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_3_1_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.3 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_3_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 1.5 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_4_0_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.1 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_4_1_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 1.3 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_4_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 2.3 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_5_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 1.9 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_5_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.3 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_6_0_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 1.6 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_6_1_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.1 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_6_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 1.7 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/night/fr_night_generation_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 1.2 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/night/fr_night_generation_SPPC_th=0.918_gpt-4o_seq.pkl | .pkl | 1.2 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/night/fr_night_generation_SPPC_th=0.919_gpt-4o_seq.pkl | .pkl | 1.5 KB | unknown | SmartGen offline synthetic numeric sequence candidate |
| /home/heyang/projects/SmartGen/SmartGen/split.py | .py | 4.8 KB | unknown | SmartGen code searched for mapping logic |
| /home/heyang/projects/SmartGen/SmartGen/sppc.py | .py | 5.6 KB | unknown | SmartGen code searched for mapping logic |
| /home/heyang/projects/SmartGen/SmartGen/text_translation_matrix.py | .py | 4.7 KB | unknown | SmartGen code searched for mapping logic |
| /home/heyang/projects/SmartGen/SmartGen/security_check.py | .py | 13.0 KB | unknown | SmartGen code searched for mapping logic |
| /home/heyang/projects/SmartGen/SmartGen/main.py | .py | 10.0 KB | unknown | SmartGen code searched for mapping logic |
| /home/heyang/projects/SmartGuard/data/fr_data/fr_trn_instance_10.pkl | .pkl | 1.6 MB | unknown | SmartGuard numeric 10x4 FR samples |
| /home/heyang/projects/SmartGuard/data/fr_data/fr_vld_instance_10.pkl | .pkl | 240.4 KB | unknown | SmartGuard numeric 10x4 FR samples |
| /home/heyang/projects/SmartGuard/data/fr_data/fr_test_instance_10.pkl | .pkl | 478.4 KB | unknown | SmartGuard numeric 10x4 FR samples |
| /home/heyang/projects/SmartGuard/data/data/an/dictionary.py | .py | 9.8 KB | unknown | SmartGuard candidate by mapping keyword |
| /home/heyang/projects/SmartGuard/data/data/fr/dictionary.py | .py | 10.6 KB | unknown | SmartGuard candidate by mapping keyword |
| /home/heyang/projects/SmartGuard/data/data/fr/routine_device_corpus.txt | .txt | 76.5 KB | unknown | SmartGuard candidate by mapping keyword |
| /home/heyang/projects/SmartGuard/data/data/sp/dictionary.py | .py | 10.2 KB | unknown | SmartGuard candidate by mapping keyword |
| /home/heyang/projects/SmartGuard/data/data/sp/routine_device_corpus.txt | .txt | 76.5 KB | unknown | SmartGuard candidate by mapping keyword |

## Candidate File Samples

### `/home/heyang/projects/SmartGen/SmartGen/fr_keys_best.txt`

- Type: `.txt`
- Size: `4.3 KB`
- Useful for mapping: `yes`
- Notes: SmartGen textual device/action catalog; no numeric ids visible

```text
AirConditioner: fanspeedDown, fanspeedUp, notification, setAcOptionalMode, setAirConditionerMode, setCoolingSetpoint, setFanMode, setOutingMode, switch off, switch on, switch toggle, temperatureDown, temperatureUp; AirPurifier: notification...
```

### `/home/heyang/projects/SmartGen/SmartGen/sp_keys_best.txt`

- Type: `.txt`
- Size: `4.4 KB`
- Useful for mapping: `yes`
- Notes: SmartGen textual device/action catalog; no numeric ids visible

```text
AirConditioner:fanspeedDown, fanspeedUp, notification, setAcOptionalMode, setAirConditionerMode, setCoolingSetpoint, setFanMode, setOutingMode, setThermostatMode, setVolume, switch off, switch on, switch toggle, temperatureDown, temperature...
AirPurifier:notification, setAirPurifierMode, setFanMode, switch off, switch on, turnWelcomeCareOn;
Blind:refresh refresh, statelessCurtainPowerButton setButton, switch off, switch on, switchLevel setLevel, windowShade close, windowShade open, windowShadeLevel setShadeLevel, windowShadePreset presetPosition;
Camera:alarm off, cameraPreset execute, imageCapture take, notification, switch off, switch on, videoCapture capture;
ClothingCareMachine:dryerOperatingState setMachineState run, notification, setting;
Computer:notification, switch off;
ContactSensor:doorControl close, lock lock, lock unlock, switch off, switch on, switch toggle;
CurbPowerMeter:energyMeter resetEnergyMeter;
Dishwasher:refresh refresh, start;
Dryer:dryerOperatingState setMachineState run, dryerOperatingState setMachineState stop, notification, setVolume, switch on;
Elevator:refresh refresh;
Fan:fanSpeed setFanSpeed, notification, switch off, switch on;
GarageDoor:doorControl close, doorControl open, switch off, switch on;
Light:refresh refresh, setColor, setColorTemperature, setLevel, setLightingMode, switch off, switch on, switch toggle;
Microwave:notification, switch on;
MotionSensor:circlemusic21301.motionCommands active;
NetworkAudio:audioMute mute, audioVolume setVolume, bixbyCommand, mediaPlayback pause, mediaPlayback play, mediaPlayback stop, mediaTrackControl nextTrack, mute, notification, setVolume, switch off, switch on, unmute;
Other:airConditionerFanMode setFanMode, audioMute mute, audioMute unmute, colorControl setColor, colorTemperature setColorTemperature, custom.picturemode setPictureMode, custom.soundmode setSoundMode, eventstreet19532.chmode setMode, notifi...
Oven:signalahead13665.ovenprogramsv2 setProgram, signalahead13665.pauseresumev2 setPauseState, signalahead13665.programdurationv2 setProgramDuration, signalahead13665.startstopprogramv2 setStartstop, switch off, switch on;
PresenceSensor:switch off, switch on;
```

### `/home/heyang/projects/SmartGen/SmartGen/us_keys_best.txt`

- Type: `.txt`
- Size: `4.8 KB`
- Useful for mapping: `yes`
- Notes: SmartGen textual device/action catalog; no numeric ids visible

```text
AirConditioner:fanspeedDown, fanspeedUp, notification, setAcOptionalMode, setAirConditionerMode, setCoolingSetpoint, setFanMode, setFanOscillationMode, setThermostatMode, switch off, switch on, temperatureDown, temperatureUp;
AirPurifier:notification, refresh, setAirPurifierMode, setCleaningOff, setFanMode, setFanSpeed, setSleepModeOff, setSleepModeOn, setTimer, switch off, switch on, turnWelcomeCareOn;
Blind:refresh refresh, statelessCurtainPowerButton setButton, switch off, switch on, switchLevel setLevel, windowShade close, windowShade open, windowShade pause, windowShadeLevel setShadeLevel, windowShadePreset presetPosition;
Camera:cameraPreset execute, notification, switch off, switch on, videoCapture capture;
ClothingCareMachine:dryerOperatingState setMachineState run;
Computer:mute, notification, switch off, switch on, unmute;
ContactSensor:lock unlock, momentary push, switch off, switch on;
Dishwasher:notification, setRun, setStop, start;
Dryer:dryerOperatingState setMachineState pause, dryerOperatingState setMachineState run, dryerOperatingState setMachineState stop, notification, open, setMode, setTimer, setVolume, switch off, switch on;
Elevator:elevatorCall call, notification;
Fan:abateachieve62503.statelessPowerOff powerOff, fanSpeed setFanSpeed, notification, setRotation off, statelessPowerToggleButton setButtonPowerToggle, switch off, switch on, switch toggle;
GarageDoor:doorControl close, doorControl open, garageDoorControl close, garageDoorControl open, notification, switch off, switch on;
Humidifier:notification, setAutohumidity, setFanSpeed, setNightmodeswitch off, setNightmodeswitch on, switch off, switch on;
Irrigation:switch off, switch on;
Light: setColor, setColorTemperature, setLevel, setLightingMode, switch off, switch on, switch toggle;
LightSensor:switch off, switch on;
Microwave:notification, switch off, switch on;
MotionSensor:setColor, setStatusLedColor, switch off, switch on;
MultiFunctionalSensor:switch off, switch on;
NetworkAudio:audioMute mute, audioVolume setVolume, bixbyCommand, mediaPlayback pause, mediaPlayback play, mediaPlayback stop, musicPlayer setLevel, notification, switch off, switch on;
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/daytime/action_transitions.json`

- Type: `.json`
- Size: `4.2 KB`
- Useful for mapping: `yes`
- Notes: Textual Device:action transition hints; no numeric ids visible

```text
top-level: dict
keys: 14
'Blind:windowShade close': {'message': 'The most common action following it', 'transitions': [{'next_action': 'RobotCleaner:setRobotCleanerMovement charging', 'count': 6}, {'next_action': 'Camera:notification', 'count': 1}]}
'Blind:windowShade open': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Blind:windowShade close', 'count': 3}, {'next_action': 'Camera:notification', 'count': 3}, {'next_action': 'Blind:windowShade open', 'count': 1}]}
'Camera:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Blind:windowShade open', 'count': 3}, {'next_action': 'Blind:windowShade close', 'count': 2}, {'next_action': 'RobotCleaner:setRobotCleanerMovement charging', 'count': 1}]}
'NetworkAudio:audioMute mute': {'message': 'It is usually not followed by the next action.'}
'NetworkAudio:audioVolume setVolume': {'message': 'It is usually not followed by the next action.'}
'None:location': {'message': 'The most common action following it', 'transitions': [{'next_action': 'None:location', 'count': 61}]}
'RobotCleaner:setRobotCleanerMovement charging': {'message': 'The most common action following it', 'transitions': [{'next_action': 'RobotCleaner:setRobotCleanerMovement charging', 'count': 5}, {'next_action': 'RobotCleaner:setRobotCleanerMovement cleaning', 'count': 4}, {'next_action': 'Camera:notification'...
'RobotCleaner:setRobotCleanerMovement cleaning': {'message': 'The most common action following it', 'transitions': [{'next_action': 'RobotCleaner:setRobotCleanerMovement cleaning', 'count': 3}, {'next_action': 'Camera:notification', 'count': 1}]}
'Television:audioMute unmute': {'message': 'It is usually not followed by the next action.'}
'Television:setChannel': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Television:volumeDown', 'count': 5}, {'next_action': 'Television:setChannel', 'count': 1}]}
'Television:setInputSource': {'message': 'It is usually not followed by the next action.'}
'Television:setPictureMode': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Television:setPictureMode', 'count': 4}]}
'Television:switch on': {'message': 'It is usually not followed by the next action.'}
'Television:volumeDown': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Television:audioMute unmute', 'count': 2}, {'next_action': 'Television:setChannel', 'count': 1}, {'next_action': 'Television:volumeDown', 'count': 1}]}
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/single/action_transitions.json`

- Type: `.json`
- Size: `7.1 KB`
- Useful for mapping: `yes`
- Notes: Textual Device:action transition hints; no numeric ids visible

```text
top-level: dict
keys: 30
'AirConditioner:switch off': {'message': 'It is usually not followed by the next action.'}
'AirConditioner:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirConditioner:switch on', 'count': 7}, {'next_action': 'AirConditioner:switch off', 'count': 1}]}
'AirPurifier:setAirPurifierMode': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirPurifier:setAirPurifierMode', 'count': 6}]}
'Blind:windowShade close': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Blind:windowShade open', 'count': 1}]}
'Blind:windowShade open': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Blind:windowShade close', 'count': 1}]}
'Camera:notification': {'message': 'It is usually not followed by the next action.'}
'Fan:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Fan:switch on', 'count': 6}]}
'GarageDoor:doorControl close': {'message': 'It is usually not followed by the next action.'}
'GarageDoor:doorControl open': {'message': 'The most common action following it', 'transitions': [{'next_action': 'GarageDoor:doorControl open', 'count': 3}, {'next_action': 'GarageDoor:doorControl close', 'count': 2}]}
'Light:setLevel': {'message': 'It is usually not followed by the next action.'}
'Light:switch off': {'message': 'It is usually not followed by the next action.'}
'NetworkAudio:audioMute mute': {'message': 'It is usually not followed by the next action.'}
'NetworkAudio:audioVolume setVolume': {'message': 'It is usually not followed by the next action.'}
'NetworkAudio:mediaPlayback play': {'message': 'It is usually not followed by the next action.'}
'NetworkAudio:switch on': {'message': 'It is usually not followed by the next action.'}
'None:location': {'message': 'The most common action following it', 'transitions': [{'next_action': 'None:location', 'count': 663}]}
'Other:custom.soundmode setSoundMode': {'message': 'It is usually not followed by the next action.'}
'Other:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Other:notification', 'count': 16}]}
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/winter/action_transitions.json`

- Type: `.json`
- Size: `14.4 KB`
- Useful for mapping: `yes`
- Notes: Textual Device:action transition hints; no numeric ids visible

```text
top-level: dict
keys: 40
'AirPurifier:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirPurifier:setFanSpeed', 'count': 1}]}
'AirPurifier:refresh refresh': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Heater:switch off', 'count': 1}, {'next_action': 'Television:setChannel', 'count': 1}]}
'AirPurifier:setAirPurifierMode': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirPurifier:setAirPurifierMode', 'count': 3}]}
'AirPurifier:setFanSpeed': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Camera:notification', 'count': 16}, {'next_action': 'Television:setChannel', 'count': 5}, {'next_action': 'Blind:windowShade open', 'count': 3}, {'next_action': 'Dryer:notifica...
'Blind:windowShade close': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Camera:notification', 'count': 22}, {'next_action': 'Blind:windowShade open', 'count': 20}, {'next_action': 'RobotCleaner:setRobotCleanerMovement charging', 'count': 16}, {'nex...
'Blind:windowShade open': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Blind:windowShade close', 'count': 20}, {'next_action': 'Camera:notification', 'count': 19}, {'next_action': 'RobotCleaner:setRobotCleanerMovement cleaning', 'count': 5}, {'nex...
'Camera:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Blind:windowShade close', 'count': 33}, {'next_action': 'Blind:windowShade open', 'count': 14}, {'next_action': 'AirPurifier:setFanSpeed', 'count': 13}, {'next_action': 'Televi...
'Dryer:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'NetworkAudio:audioVolume setVolume', 'count': 5}, {'next_action': 'Blind:windowShade close', 'count': 3}]}
'GarageDoor:doorControl close': {'message': 'It is usually not followed by the next action.'}
'GarageDoor:doorControl open': {'message': 'The most common action following it', 'transitions': [{'next_action': 'GarageDoor:doorControl open', 'count': 3}, {'next_action': 'GarageDoor:doorControl close', 'count': 2}, {'next_action': 'Television:setChannel', 'count': 1}]}
'Light:setLevel': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Television:setChannel', 'count': 3}]}
'Light:setLightingMode': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Light:setLightingMode', 'count': 1}]}
'Light:switch off': {'message': 'It is usually not followed by the next action.'}
'Light:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Light:setLightingMode', 'count': 4}]}
'NetworkAudio:audioMute mute': {'message': 'The most common action following it', 'transitions': [{'next_action': 'None:location', 'count': 4}]}
'NetworkAudio:audioVolume setVolume': {'message': 'The most common action following it', 'transitions': [{'next_action': 'RobotCleaner:setRobotCleanerMovement charging', 'count': 5}]}
'NetworkAudio:mediaPlayback play': {'message': 'It is usually not followed by the next action.'}
'NetworkAudio:refresh refresh': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Blind:windowShade open', 'count': 1}]}
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/sp/daytime/action_transitions.json`

- Type: `.json`
- Size: `10.6 KB`
- Useful for mapping: `yes`
- Notes: Textual Device:action transition hints; no numeric ids visible

```text
top-level: dict
keys: 37
'AirConditioner:setAirConditionerMode': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirConditioner:setAirConditionerMode', 'count': 6}, {'next_action': 'Dryer:dryerOperatingState setMachineState run', 'count': 5}]}
'AirConditioner:switch off': {'message': 'It is usually not followed by the next action.'}
'AirConditioner:temperatureUp': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirConditioner:temperatureUp', 'count': 1}]}
'Camera:cameraPreset execute': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Other:notification', 'count': 3}, {'next_action': 'None:location', 'count': 1}]}
'Camera:switch on': {'message': 'It is usually not followed by the next action.'}
'Dryer:dryerOperatingState setMachineState run': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Refrigerator:samsungce.powerCool activate', 'count': 12}, {'next_action': 'Dryer:dryerOperatingState setMachineState run', 'count': 2}]}
'Fan:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Fan:switch off', 'count': 2}]}
'Fan:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Fan:switch on', 'count': 3}]}
'GarageDoor:doorControl close': {'message': 'The most common action following it', 'transitions': [{'next_action': 'GarageDoor:doorControl open', 'count': 13}, {'next_action': 'SmartLock:alarm off', 'count': 5}, {'next_action': 'Other:notification', 'count': 4}]}
'GarageDoor:doorControl open': {'message': 'The most common action following it', 'transitions': [{'next_action': 'GarageDoor:doorControl open', 'count': 33}, {'next_action': 'GarageDoor:doorControl close', 'count': 5}]}
'NetworkAudio:audioMute mute': {'message': 'The most common action following it', 'transitions': [{'next_action': 'NetworkAudio:audioVolume setVolume', 'count': 1}]}
'NetworkAudio:audioVolume setVolume': {'message': 'It is usually not followed by the next action.'}
'NetworkAudio:mediaPlayback play': {'message': 'It is usually not followed by the next action.'}
'None:location': {'message': 'The most common action following it', 'transitions': [{'next_action': 'None:location', 'count': 65}, {'next_action': 'Other:switch off', 'count': 3}]}
'Other:custom.soundmode setSoundMode': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Television:volumeDown', 'count': 5}]}
'Other:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Other:thermostatCoolingSetpoint setCoolingSetpoint', 'count': 4}, {'next_action': 'Other:notification', 'count': 1}]}
'Other:robotCleanerTurboMode setRobotCleanerTurboMode': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Other:robotCleanerTurboMode setRobotCleanerTurboMode', 'count': 9}, {'next_action': 'Other:notification', 'count': 6}]}
'Other:samsungvd.mediaInputSource setInputSource': {'message': 'It is usually not followed by the next action.'}
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/sp/single/action_transitions.json`

- Type: `.json`
- Size: `14.6 KB`
- Useful for mapping: `yes`
- Notes: Textual Device:action transition hints; no numeric ids visible

```text
top-level: dict
keys: 47
'AirConditioner:setAirConditionerMode': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirConditioner:setAirConditionerMode', 'count': 4}]}
'AirConditioner:switch off': {'message': 'It is usually not followed by the next action.'}
'AirConditioner:temperatureUp': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirConditioner:temperatureUp', 'count': 1}]}
'Camera:cameraPreset execute': {'message': 'It is usually not followed by the next action.'}
'Camera:switch on': {'message': 'It is usually not followed by the next action.'}
'Computer:notification': {'message': 'It is usually not followed by the next action.'}
'Dryer:dryerOperatingState setMachineState run': {'message': 'It is usually not followed by the next action.'}
'Fan:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Fan:switch on', 'count': 132}, {'next_action': 'Fan:switch off', 'count': 50}]}
'Fan:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Fan:switch off', 'count': 204}, {'next_action': 'Fan:switch on', 'count': 41}]}
'GarageDoor:doorControl close': {'message': 'The most common action following it', 'transitions': [{'next_action': 'GarageDoor:doorControl open', 'count': 27}, {'next_action': 'GarageDoor:doorControl close', 'count': 2}]}
'GarageDoor:doorControl open': {'message': 'The most common action following it', 'transitions': [{'next_action': 'GarageDoor:doorControl open', 'count': 116}, {'next_action': 'GarageDoor:doorControl close', 'count': 26}]}
'Light:setColor': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Light:setColor', 'count': 254}, {'next_action': 'Light:setLevel', 'count': 49}, {'next_action': 'Light:setColorTemperature', 'count': 8}, {'next_action': 'Light:setLightingMode...
'Light:setColorTemperature': {'message': 'It is usually not followed by the next action.'}
'Light:setLevel': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Light:setLevel', 'count': 30}, {'next_action': 'Light:setColor', 'count': 5}, {'next_action': 'Light:switch on', 'count': 3}, {'next_action': 'Light:setLightingMode', 'count': ...
'Light:setLightingMode': {'message': 'It is usually not followed by the next action.'}
'Light:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Light:switch on', 'count': 6}]}
'Light:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Light:switch on', 'count': 55}, {'next_action': 'Light:setColor', 'count': 13}, {'next_action': 'Light:setLightingMode', 'count': 4}, {'next_action': 'Light:setLevel', 'count':...
'NetworkAudio:audioMute mute': {'message': 'The most common action following it', 'transitions': [{'next_action': 'NetworkAudio:audioVolume setVolume', 'count': 1}]}
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/sp/winter/action_transitions.json`

- Type: `.json`
- Size: `20.1 KB`
- Useful for mapping: `yes`
- Notes: Textual Device:action transition hints; no numeric ids visible

```text
top-level: dict
keys: 56
'AirPurifier:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Other:notification', 'count': 5}]}
'Blind:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Other:switch on', 'count': 1}]}
'Blind:windowShade close': {'message': 'It is usually not followed by the next action.'}
'Camera:cameraPreset execute': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Other:notification', 'count': 3}, {'next_action': 'None:location', 'count': 1}]}
'Camera:notification': {'message': 'It is usually not followed by the next action.'}
'Camera:switch on': {'message': 'It is usually not followed by the next action.'}
'Computer:notification': {'message': 'It is usually not followed by the next action.'}
'Dryer:dryerOperatingState setMachineState run': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Refrigerator:samsungce.powerCool activate', 'count': 42}, {'next_action': 'Light:setColor', 'count': 10}, {'next_action': 'Dryer:dryerOperatingState setMachineState run', 'coun...
'GarageDoor:doorControl close': {'message': 'The most common action following it', 'transitions': [{'next_action': 'GarageDoor:doorControl open', 'count': 28}, {'next_action': 'GarageDoor:doorControl close', 'count': 5}, {'next_action': 'SmartLock:alarm off', 'count': 3}, {'next_action': 'Ot...
'GarageDoor:doorControl open': {'message': 'The most common action following it', 'transitions': [{'next_action': 'GarageDoor:doorControl open', 'count': 116}, {'next_action': 'GarageDoor:doorControl close', 'count': 29}, {'next_action': 'SmartLock:lock lock', 'count': 7}]}
'Light:setColor': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Light:setColor', 'count': 264}, {'next_action': 'Light:setLevel', 'count': 55}, {'next_action': 'Refrigerator:samsungce.powerCool activate', 'count': 23}, {'next_action': 'Drye...
'Light:setColorTemperature': {'message': 'It is usually not followed by the next action.'}
'Light:setLevel': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Light:setLevel', 'count': 32}, {'next_action': 'Television:volumeDown', 'count': 5}, {'next_action': 'Light:setColor', 'count': 5}, {'next_action': 'Light:switch on', 'count': ...
'Light:setLightingMode': {'message': 'It is usually not followed by the next action.'}
'Light:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Light:switch on', 'count': 6}]}
'Light:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Television:setChannel', 'count': 92}, {'next_action': 'Light:switch on', 'count': 75}, {'next_action': 'None:location', 'count': 27}, {'next_action': 'Light:setColor', 'count':...
'NetworkAudio:audioMute mute': {'message': 'The most common action following it', 'transitions': [{'next_action': 'NetworkAudio:audioVolume setVolume', 'count': 1}, {'next_action': 'NetworkAudio:audioMute mute', 'count': 1}]}
'NetworkAudio:audioVolume setVolume': {'message': 'The most common action following it', 'transitions': [{'next_action': 'NetworkAudio:switch off', 'count': 12}, {'next_action': 'Other:custom.soundmode setSoundMode', 'count': 5}, {'next_action': 'NetworkAudio:audioMute mute', 'count': 2}]}
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/us/daytime/action_transitions.json`

- Type: `.json`
- Size: `27.9 KB`
- Useful for mapping: `yes`
- Notes: Textual Device:action transition hints; no numeric ids visible

```text
top-level: dict
keys: 63
'AirConditioner:setAirConditionerMode': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Television:switch off', 'count': 1}]}
'AirConditioner:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirConditioner:switch on', 'count': 1}, {'next_action': 'AirConditioner:switch off', 'count': 1}, {'next_action': 'Computer:switch off', 'count': 1}, {'next_action': 'AirCondit...
'AirConditioner:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirPurifier:switch on', 'count': 2}, {'next_action': 'AirConditioner:switch off', 'count': 1}, {'next_action': 'AirConditioner:switch on', 'count': 1}, {'next_action': 'Dishwas...
'AirConditioner:temperatureUp': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirConditioner:switch on', 'count': 2}]}
'AirPurifier:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Dryer:notification', 'count': 10}, {'next_action': 'Refrigerator:setCoolTemperature', 'count': 8}, {'next_action': 'AirPurifier:notification', 'count': 5}, {'next_action': 'Dry...
'AirPurifier:setAirPurifierMode': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirPurifier:setAirPurifierMode', 'count': 1}, {'next_action': 'Projector:switch on', 'count': 1}]}
'AirPurifier:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'RobotCleaner:setRobotCleanerMovement cleaning', 'count': 23}, {'next_action': 'SmartPlug:switch on', 'count': 10}, {'next_action': 'Dryer:notification', 'count': 6}, {'next_act...
'AirPurifier:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirPurifier:notification', 'count': 1}, {'next_action': 'AirPurifier:switch off', 'count': 1}]}
'Camera:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Dryer:switch on', 'count': 66}, {'next_action': 'NetworkAudio:audioMute mute', 'count': 28}, {'next_action': 'RobotCleaner:setRobotCleanerMovement charging', 'count': 19}, {'ne...
'ClothingCareMachine:dryerOperatingState setMachineState run': {'message': 'The most common action following it', 'transitions': [{'next_action': 'NetworkAudio:mediaPlayback play', 'count': 8}, {'next_action': 'NetworkAudio:audioVolume setVolume', 'count': 5}, {'next_action': 'ClothingCareMachine:dryerOperatingState setMa...
'Computer:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Projector:switch off', 'count': 1}]}
'Computer:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Computer:switch off', 'count': 2}]}
'Dishwasher:setStop': {'message': 'The most common action following it', 'transitions': [{'next_action': 'ClothingCareMachine:dryerOperatingState setMachineState run', 'count': 11}, {'next_action': 'Dryer:switch on', 'count': 3}, {'next_action': 'NetworkAudio:mediaPlayback play', '...
'Dishwasher:start': {'message': 'It is usually not followed by the next action.'}
'Dryer:dryerOperatingState setMachineState run': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Dryer:notification', 'count': 1}]}
'Dryer:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirPurifier:switch off', 'count': 8}, {'next_action': 'AirPurifier:notification', 'count': 7}, {'next_action': 'RobotCleaner:setRobotCleanerMovement charging', 'count': 6}, {'n...
'Dryer:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Television:switch off', 'count': 1}]}
'Dryer:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Camera:notification', 'count': 72}, {'next_action': 'NetworkAudio:audioMute mute', 'count': 23}, {'next_action': 'RobotCleaner:setRobotCleanerMovement charging', 'count': 8}, {...
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/us/single/action_transitions.json`

- Type: `.json`
- Size: `20.2 KB`
- Useful for mapping: `yes`
- Notes: Textual Device:action transition hints; no numeric ids visible

```text
top-level: dict
keys: 53
'AirConditioner:setAirConditionerMode': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirConditioner:switch on', 'count': 11}, {'next_action': 'AirConditioner:switch off', 'count': 10}, {'next_action': 'AirConditioner:setAirConditionerMode', 'count': 7}]}
'AirConditioner:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirConditioner:switch on', 'count': 65}, {'next_action': 'AirConditioner:setAirConditionerMode', 'count': 8}, {'next_action': 'AirConditioner:switch off', 'count': 7}]}
'AirConditioner:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirConditioner:switch off', 'count': 94}, {'next_action': 'AirConditioner:setAirConditionerMode', 'count': 24}, {'next_action': 'AirConditioner:switch on', 'count': 21}, {'next...
'AirConditioner:temperatureUp': {'message': 'It is usually not followed by the next action.'}
'AirPurifier:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirPurifier:switch on', 'count': 1}]}
'AirPurifier:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirPurifier:switch on', 'count': 1}]}
'AirPurifier:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirPurifier:switch off', 'count': 3}]}
'Camera:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Camera:switch on', 'count': 1}]}
'Camera:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Camera:switch off', 'count': 4}]}
'Computer:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Computer:switch off', 'count': 1}]}
'Computer:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Computer:switch on', 'count': 2}]}
'Computer:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Computer:switch on', 'count': 7}, {'next_action': 'Computer:switch off', 'count': 3}, {'next_action': 'Computer:notification', 'count': 2}]}
'Dishwasher:start': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Dishwasher:start', 'count': 12}]}
'Fan:fanSpeed setFanSpeed': {'message': 'It is usually not followed by the next action.'}
'Fan:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Fan:switch on', 'count': 30}, {'next_action': 'Fan:switch off', 'count': 1}]}
'Fan:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Fan:switch off', 'count': 37}, {'next_action': 'Fan:switch on', 'count': 3}]}
'GarageDoor:doorControl close': {'message': 'The most common action following it', 'transitions': [{'next_action': 'GarageDoor:doorControl close', 'count': 14}]}
'GarageDoor:doorControl open': {'message': 'It is usually not followed by the next action.'}
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/us/winter/action_transitions.json`

- Type: `.json`
- Size: `46.7 KB`
- Useful for mapping: `yes`
- Notes: Textual Device:action transition hints; no numeric ids visible

```text
top-level: dict
keys: 90
'AirPurifier:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Dryer:notification', 'count': 20}, {'next_action': 'Refrigerator:setCoolTemperature', 'count': 8}, {'next_action': 'AirPurifier:switch off', 'count': 6}, {'next_action': 'Robot...
'AirPurifier:setAirPurifierMode': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Computer:switch off', 'count': 4}, {'next_action': 'Projector:switch on', 'count': 2}, {'next_action': 'AirPurifier:setAirPurifierMode', 'count': 2}]}
'AirPurifier:setSleepModeOn': {'message': 'It is usually not followed by the next action.'}
'AirPurifier:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'RobotCleaner:setRobotCleanerMovement cleaning', 'count': 36}, {'next_action': 'SmartPlug:switch on', 'count': 16}, {'next_action': 'NetworkAudio:audioMute mute', 'count': 12}, ...
'AirPurifier:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'AirPurifier:switch off', 'count': 14}, {'next_action': 'Projector:switch on', 'count': 2}, {'next_action': 'Heater:switch on', 'count': 2}, {'next_action': 'Light:switch off', ...
'Blind:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'SmartLock:switch off', 'count': 7}, {'next_action': 'Blind:switch on', 'count': 2}, {'next_action': 'Blind:windowShade close', 'count': 2}, {'next_action': 'Television:switch o...
'Blind:switch on': {'message': 'It is usually not followed by the next action.'}
'Blind:windowShade close': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Television:switch off', 'count': 5}, {'next_action': 'Blind:switch off', 'count': 3}, {'next_action': 'Light:switch on', 'count': 3}, {'next_action': 'Light:switch off', 'count...
'Blind:windowShade open': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Light:switch on', 'count': 6}, {'next_action': 'Blind:windowShade close', 'count': 4}, {'next_action': 'Blind:windowShade open', 'count': 2}, {'next_action': 'Television:setInp...
'Camera:cameraPreset execute': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Heater:switch on', 'count': 1}]}
'Camera:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Dryer:switch on', 'count': 127}, {'next_action': 'NetworkAudio:audioMute mute', 'count': 44}, {'next_action': 'RobotCleaner:setRobotCleanerMovement charging', 'count': 43}, {'n...
'Camera:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Camera:switch on', 'count': 1}]}
'Camera:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Camera:switch off', 'count': 4}, {'next_action': 'GarageDoor:switch on', 'count': 1}]}
'ClothingCareMachine:dryerOperatingState setMachineState run': {'message': 'The most common action following it', 'transitions': [{'next_action': 'NetworkAudio:audioVolume setVolume', 'count': 10}, {'next_action': 'NetworkAudio:mediaPlayback play', 'count': 10}, {'next_action': 'Refrigerator:setCoolTemperature', 'count': ...
'Computer:mute': {'message': 'It is usually not followed by the next action.'}
'Computer:notification': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Computer:switch off', 'count': 1}]}
'Computer:switch off': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Computer:switch on', 'count': 12}, {'next_action': 'Computer:switch off', 'count': 5}, {'next_action': 'AirPurifier:switch off', 'count': 5}, {'next_action': 'Projector:switch ...
'Computer:switch on': {'message': 'The most common action following it', 'transitions': [{'next_action': 'Computer:switch off', 'count': 12}, {'next_action': 'Computer:switch on', 'count': 10}, {'next_action': 'Television:switch off', 'count': 10}, {'next_action': 'Light:switch off...
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/an/trn.pkl`

- Type: `.pkl`
- Size: `68.1 KB`
- Useful for mapping: `unknown`
- Notes: Numeric flat 4-tuples; useful for layout and id-space overlap

```text
top-level: list
len: 830
[4, 5, 3, 18, 4, 5, 1, 27, 4, 5, 1, 30, 4, 5, 3, 20, 4, 5, 29, 115, 4, 5, 3, 16, 4, 5, 30, 121, 4, 5, 3, 19, 4, 5, 29, 117, 4, 5, 29, 120]
[0, 6, 29, 115, 0, 6, 8, 40, 0, 6, 24, 102, 0, 6, 6, 34, 0, 6, 6, 32, 0, 6, 29, 119, 0, 6, 23, 93, 0, 6, 30, 121, 0, 6, 23, 93, 0, 6, 18, 79]
[0, 4, 16, 72, 0, 4, 3, 20, 0, 4, 3, 19, 0, 4, 27, 109, 0, 4, 9, 46, 0, 4, 6, 34, 0, 4, 3, 16, 0, 4, 3, 18, 0, 4, 6, 32, 0, 4, 9, 47]
[2, 3, 35, 68, 2, 3, 23, 93, 2, 3, 11, 55, 2, 3, 30, 121, 2, 3, 23, 94, 2, 3, 35, 68, 2, 3, 17, 77, 2, 3, 1, 30, 2, 3, 1, 27, 2, 3, 13, 60]
[3, 2, 35, 68, 3, 2, 3, 16, 3, 2, 3, 20, 3, 2, 29, 119, 3, 2, 3, 19, 3, 2, 29, 115, 3, 2, 29, 115, 3, 2, 29, 119, 3, 2, 3, 18, 3, 2, 13, 60]
[6, 5, 1, 62, 6, 5, 23, 93, 6, 5, 30, 121, 6, 5, 3, 16, 6, 5, 29, 119, 6, 5, 29, 115, 6, 5, 1, 63, 6, 5, 23, 94, 6, 5, 13, 60, 6, 5, 13, 61]
[6, 5, 11, 55, 6, 5, 23, 93, 6, 5, 23, 93, 6, 5, 35, 68, 6, 5, 1, 62, 6, 5, 1, 63, 6, 5, 35, 68, 6, 5, 17, 76, 6, 5, 13, 60, 6, 5, 13, 61]
[6, 7, 23, 93, 6, 7, 6, 32, 6, 7, 29, 115, 6, 7, 6, 34, 6, 7, 4, 22, 6, 7, 9, 47, 6, 7, 16, 72, 6, 7, 6, 35, 6, 7, 8, 41, 6, 7, 12, 59]
[3, 5, 1, 27, 3, 5, 23, 93, 3, 5, 1, 30, 3, 5, 0, 3, 3, 5, 1, 30, 3, 5, 1, 27, 3, 5, 6, 34, 3, 5, 35, 68, 3, 5, 6, 32, 3, 5, 0, 4]
[0, 4, 23, 93, 0, 4, 13, 60, 0, 4, 23, 94, 0, 4, 13, 61, 0, 4, 26, 107, 0, 4, 1, 62, 0, 4, 29, 115, 0, 4, 1, 63, 0, 4, 12, 58, 0, 4, 12, 59]
[3, 2, 34, 136, 3, 2, 34, 137, 3, 2, 29, 115, 3, 2, 29, 119, 3, 2, 17, 74, 3, 2, 17, 76, 3, 2, 6, 32, 3, 2, 6, 34, 3, 2, 11, 55, 3, 2, 1, 62]
[1, 7, 2, 11, 1, 7, 2, 13, 1, 7, 23, 93, 1, 7, 0, 1, 1, 7, 1, 27, 1, 7, 1, 29, 1, 7, 1, 31, 1, 7, 11, 55, 1, 7, 4, 22, 1, 7, 0, 1]
[3, 4, 23, 93, 3, 4, 32, 129, 3, 4, 6, 34, 3, 4, 16, 72, 3, 4, 29, 115, 3, 4, 16, 73, 3, 4, 6, 32, 3, 4, 34, 136, 3, 4, 6, 35, 3, 4, 34, 137]
[5, 5, 1, 62, 5, 5, 23, 93, 5, 5, 1, 63, 5, 5, 23, 94, 5, 5, 26, 107, 5, 5, 1, 27, 5, 5, 29, 119, 5, 5, 29, 115, 5, 5, 13, 60, 5, 5, 13, 61]
[6, 2, 34, 136, 6, 2, 34, 137, 6, 2, 11, 55, 6, 2, 29, 115, 6, 2, 29, 119, 6, 2, 17, 74, 6, 2, 17, 76, 6, 2, 6, 32, 6, 2, 6, 34, 6, 2, 1, 62]
[0, 6, 30, 121, 0, 6, 17, 76, 0, 6, 11, 55, 0, 6, 6, 32, 0, 6, 3, 16, 0, 6, 17, 76, 0, 6, 6, 34, 0, 6, 6, 35, 0, 6, 35, 68, 0, 6, 27, 109]
[1, 7, 2, 11, 1, 7, 2, 13, 1, 7, 0, 1, 1, 7, 1, 27, 1, 7, 1, 29, 1, 7, 1, 31, 1, 7, 29, 119, 1, 7, 4, 22, 1, 7, 0, 1, 1, 7, 6, 32]
[3, 6, 36, 111, 3, 6, 36, 113, 3, 6, 23, 93, 3, 6, 36, 112, 3, 6, 6, 35, 3, 6, 6, 32, 3, 6, 35, 68, 3, 6, 6, 32, 3, 6, 6, 34, 3, 6, 33, 131]
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/an/winter/trn.pkl`

- Type: `.pkl`
- Size: `46.1 KB`
- Useful for mapping: `unknown`
- Notes: Numeric flat 4-tuples; useful for layout and id-space overlap

```text
top-level: list
len: 562
[4, 5, 3, 18, 4, 5, 1, 27, 4, 5, 1, 30, 4, 5, 3, 20, 4, 5, 29, 115, 4, 5, 3, 16, 4, 5, 30, 121, 4, 5, 3, 19, 4, 5, 29, 117, 4, 5, 29, 120]
[0, 6, 29, 115, 0, 6, 8, 40, 0, 6, 24, 102, 0, 6, 6, 34, 0, 6, 6, 32, 0, 6, 29, 119, 0, 6, 23, 93, 0, 6, 30, 121, 0, 6, 23, 93, 0, 6, 18, 79]
[0, 4, 16, 72, 0, 4, 3, 20, 0, 4, 3, 19, 0, 4, 27, 109, 0, 4, 9, 46, 0, 4, 6, 34, 0, 4, 3, 16, 0, 4, 3, 18, 0, 4, 6, 32, 0, 4, 9, 47]
[3, 2, 35, 68, 3, 2, 3, 16, 3, 2, 3, 20, 3, 2, 29, 119, 3, 2, 3, 19, 3, 2, 29, 115, 3, 2, 29, 115, 3, 2, 29, 119, 3, 2, 3, 18, 3, 2, 13, 60]
[6, 5, 1, 62, 6, 5, 23, 93, 6, 5, 30, 121, 6, 5, 3, 16, 6, 5, 29, 119, 6, 5, 29, 115, 6, 5, 1, 63, 6, 5, 23, 94, 6, 5, 13, 60, 6, 5, 13, 61]
[6, 7, 23, 93, 6, 7, 6, 32, 6, 7, 29, 115, 6, 7, 6, 34, 6, 7, 4, 22, 6, 7, 9, 47, 6, 7, 16, 72, 6, 7, 6, 35, 6, 7, 8, 41, 6, 7, 12, 59]
[0, 4, 23, 93, 0, 4, 13, 60, 0, 4, 23, 94, 0, 4, 13, 61, 0, 4, 26, 107, 0, 4, 1, 62, 0, 4, 29, 115, 0, 4, 1, 63, 0, 4, 12, 58, 0, 4, 12, 59]
[3, 4, 23, 93, 3, 4, 32, 129, 3, 4, 6, 34, 3, 4, 16, 72, 3, 4, 29, 115, 3, 4, 16, 73, 3, 4, 6, 32, 3, 4, 34, 136, 3, 4, 6, 35, 3, 4, 34, 137]
[5, 5, 1, 62, 5, 5, 23, 93, 5, 5, 1, 63, 5, 5, 23, 94, 5, 5, 26, 107, 5, 5, 1, 27, 5, 5, 29, 119, 5, 5, 29, 115, 5, 5, 13, 60, 5, 5, 13, 61]
[3, 6, 36, 111, 3, 6, 36, 113, 3, 6, 23, 93, 3, 6, 36, 112, 3, 6, 6, 35, 3, 6, 6, 32, 3, 6, 35, 68, 3, 6, 6, 32, 3, 6, 6, 34, 3, 6, 33, 131]
[6, 3, 30, 121, 6, 3, 8, 40, 6, 3, 1, 30, 6, 3, 30, 121, 6, 3, 1, 27, 6, 3, 29, 120, 6, 3, 29, 118, 6, 6, 23, 93, 6, 6, 23, 93, 6, 6, 18, 79]
[6, 6, 13, 60, 6, 6, 3, 20, 6, 6, 3, 18, 6, 6, 1, 27, 6, 6, 30, 121, 6, 6, 3, 16, 6, 6, 29, 115, 6, 6, 29, 119, 6, 6, 3, 19, 6, 6, 13, 61]
[2, 3, 9, 46, 2, 3, 16, 71, 2, 3, 1, 62, 2, 3, 1, 63, 2, 3, 10, 51, 2, 3, 10, 53, 2, 3, 35, 68, 2, 3, 13, 60, 2, 3, 35, 69, 2, 3, 9, 47]
[3, 5, 13, 60, 3, 5, 3, 20, 3, 5, 3, 18, 3, 5, 13, 61, 3, 5, 3, 19, 3, 5, 23, 93, 3, 5, 3, 16, 3, 5, 23, 94, 3, 5, 26, 107, 3, 5, 26, 108]
[5, 7, 6, 34, 5, 7, 6, 35, 5, 7, 1, 62, 5, 7, 1, 64, 5, 7, 36, 111, 5, 7, 30, 121, 5, 7, 6, 32, 5, 7, 1, 63, 5, 7, 36, 113, 5, 7, 1, 65]
[3, 2, 3, 19, 3, 2, 35, 68, 3, 2, 3, 16, 3, 2, 1, 27, 3, 2, 3, 20, 3, 2, 30, 121, 3, 2, 1, 62, 3, 2, 23, 93, 3, 2, 3, 18, 3, 2, 1, 63]
[2, 7, 6, 35, 2, 7, 6, 34, 2, 7, 6, 32, 2, 7, 30, 121, 2, 7, 4, 22, 2, 7, 3, 16, 2, 7, 9, 47, 2, 7, 16, 72, 2, 7, 8, 41, 2, 7, 12, 59]
[4, 3, 3, 19, 4, 3, 1, 30, 4, 3, 1, 27, 4, 3, 30, 121, 4, 3, 23, 93, 4, 3, 3, 18, 4, 3, 3, 20, 4, 3, 3, 16, 4, 3, 13, 60, 4, 3, 13, 61]
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/daytime/trn.pkl`

- Type: `.pkl`
- Size: `4.1 KB`
- Useful for mapping: `unknown`
- Notes: Numeric flat 4-tuples; useful for layout and id-space overlap

```text
top-level: list
len: 180
[3, 2, 18, 96, 4, 5, 18, 96]
[4, 2, 29, 202]
[0, 4, 18, 96, 1, 5, 18, 96, 3, 4, 18, 96, 4, 5, 18, 96]
[1, 2, 29, 205]
[0, 3, 29, 205, 1, 3, 29, 205]
[3, 2, 29, 205, 4, 2, 29, 205, 5, 2, 29, 205]
[1, 5, 29, 192]
[0, 3, 29, 205, 1, 3, 29, 205]
[4, 5, 29, 196, 0, 5, 29, 192]
[4, 5, 18, 96, 2, 4, 18, 96, 4, 4, 18, 96, 4, 5, 18, 96, 6, 2, 18, 96, 6, 3, 18, 96, 6, 4, 18, 96]
[1, 4, 29, 205, 2, 3, 29, 205]
[4, 2, 29, 198, 4, 2, 29, 198]
[1, 5, 29, 192]
[4, 2, 29, 196, 4, 5, 29, 205]
[1, 3, 29, 205, 1, 3, 29, 196]
[4, 2, 29, 196, 4, 5, 29, 205, 0, 5, 29, 205]
[1, 5, 29, 192]
[4, 2, 29, 196, 4, 5, 29, 205, 0, 5, 29, 205]
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/trn.pkl`

- Type: `.pkl`
- Size: `1.4 KB`
- Useful for mapping: `unknown`
- Notes: Numeric flat 4-tuples; useful for layout and id-space overlap

```text
top-level: list
len: 41
[1, 2, 13, 78, 1, 2, 0, 7, 1, 2, 29, 196, 1, 2, 24, 156]
[0, 0, 29, 205, 0, 0, 29, 196, 0, 0, 0, 6, 0, 0, 13, 75]
[0, 7, 29, 196, 0, 7, 26, 170, 0, 7, 24, 157, 0, 7, 2, 27]
[3, 3, 24, 156, 3, 3, 3, 34, 3, 3, 13, 75, 3, 3, 0, 6]
[5, 6, 29, 205, 5, 6, 16, 86, 5, 6, 3, 34, 5, 6, 12, 67]
[0, 2, 29, 196, 0, 2, 13, 74, 0, 2, 11, 65, 0, 2, 24, 155]
[1, 3, 3, 34, 1, 3, 12, 67, 1, 3, 13, 75, 1, 3, 1, 16]
[3, 6, 30, 214, 3, 6, 30, 209, 3, 6, 24, 155, 3, 6, 3, 33]
[1, 4, 29, 205, 1, 4, 24, 155, 1, 4, 22, 139]
[6, 0, 13, 77, 6, 0, 1, 19, 6, 0, 29, 192, 6, 0, 29, 205]
[4, 7, 29, 205, 4, 7, 16, 94, 4, 7, 3, 34, 4, 7, 2, 30]
[2, 4, 29, 202, 2, 4, 29, 196, 2, 4, 16, 86, 2, 4, 16, 84]
[5, 2, 18, 106, 5, 2, 1, 16, 5, 2, 29, 196, 5, 2, 11, 61]
[0, 5, 18, 96, 0, 5, 29, 197, 0, 5, 16, 86]
[6, 5, 29, 196, 6, 5, 29, 205, 6, 5, 21, 132, 6, 5, 1, 16]
[3, 7, 13, 78, 3, 7, 16, 87, 3, 7, 26, 170, 3, 7, 29, 201]
[0, 3, 18, 96, 0, 3, 22, 139, 0, 3, 19, 125]
[3, 5, 12, 67, 3, 5, 26, 171, 3, 5, 29, 196, 3, 5, 29, 205]
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/night/trn.pkl`

- Type: `.pkl`
- Size: `1.1 KB`
- Useful for mapping: `unknown`
- Notes: Numeric flat 4-tuples; useful for layout and id-space overlap

```text
top-level: list
len: 18
[1, 7, 13, 78, 1, 7, 29, 205, 1, 7, 29, 192, 1, 7, 16, 86, 1, 7, 13, 75, 1, 7, 16, 87]
[4, 4, 2, 27, 4, 4, 0, 12, 4, 4, 30, 209, 4, 4, 13, 77, 4, 4, 20, 129, 4, 4, 25, 165]
[4, 7, 2, 27, 4, 7, 24, 155, 4, 7, 3, 34, 4, 7, 2, 28, 4, 7, 24, 156]
[0, 6, 2, 28, 0, 6, 24, 156, 0, 6, 3, 34, 0, 6, 2, 27, 0, 6, 24, 155]
[6, 0, 13, 78, 6, 0, 29, 196, 6, 0, 29, 205, 6, 0, 16, 86, 6, 0, 0, 12, 6, 0, 2, 28, 6, 0, 3, 33, 6, 0, 24, 155]
[2, 0, 13, 75, 2, 0, 29, 205, 2, 0, 29, 192, 2, 0, 16, 86, 2, 0, 16, 84]
[2, 0, 0, 12, 2, 0, 30, 209, 2, 0, 20, 129, 2, 0, 25, 165, 2, 0, 13, 77, 2, 0, 2, 27, 2, 0, 24, 156]
[1, 7, 2, 27, 1, 7, 24, 155, 1, 7, 3, 34, 1, 7, 2, 28, 1, 7, 24, 156]
[4, 0, 3, 34, 4, 0, 26, 170, 4, 0, 20, 129, 4, 0, 1, 16, 4, 0, 30, 209]
[6, 1, 13, 77, 6, 1, 29, 201, 6, 1, 0, 9, 6, 1, 2, 27, 6, 1, 26, 170, 6, 1, 3, 36]
[0, 7, 13, 78, 0, 7, 2, 27, 0, 7, 3, 34, 0, 7, 29, 202, 0, 7, 29, 196, 0, 7, 29, 205, 0, 7, 0, 10, 0, 7, 0, 6, 0, 7, 16, 86, 0, 7, 2, 28, 0, 7, 24, 156]
[1, 0, 13, 78, 1, 0, 2, 27, 1, 0, 3, 34, 1, 0, 29, 202, 1, 0, 29, 196, 1, 0, 29, 205, 1, 0, 0, 10, 1, 0, 0, 6, 1, 0, 16, 86, 1, 0, 2, 28, 1, 0, 24, 156]
[4, 0, 2, 28, 4, 0, 24, 156, 4, 0, 3, 34, 4, 0, 2, 27, 4, 0, 13, 78, 4, 0, 29, 197, 4, 0, 16, 83]
[4, 1, 24, 156, 4, 1, 2, 27, 4, 1, 3, 34, 4, 1, 13, 77, 4, 1, 29, 201]
[3, 7, 2, 28, 3, 7, 24, 156, 3, 7, 3, 34, 3, 7, 2, 27, 3, 7, 13, 78, 3, 7, 29, 205, 3, 7, 29, 192]
[2, 7, 2, 28, 2, 7, 3, 34, 2, 7, 2, 27, 2, 7, 24, 156, 2, 7, 24, 155, 2, 7, 13, 74, 2, 7, 29, 196, 2, 7, 29, 205, 2, 7, 16, 86, 2, 7, 0, 6, 2, 7, 1, 18, 2, 7, 26, 170]
[3, 0, 2, 28, 3, 0, 3, 34, 3, 0, 2, 27, 3, 0, 24, 156, 3, 0, 24, 155, 3, 0, 13, 75, 3, 0, 29, 198, 3, 0, 29, 205, 3, 0, 16, 84, 3, 0, 0, 12, 3, 0, 1, 16, 3, 0, 26, 171]
[5, 7, 2, 28, 5, 7, 24, 156, 5, 7, 3, 34, 5, 7, 2, 27, 5, 7, 13, 78, 5, 7, 29, 205, 5, 7, 29, 192]
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/single/trn.pkl`

- Type: `.pkl`
- Size: `26.7 KB`
- Useful for mapping: `unknown`
- Notes: Numeric flat 4-tuples; useful for layout and id-space overlap

```text
top-level: list
len: 964
[6, 1, 29, 192]
[4, 1, 29, 196, 4, 2, 29, 196, 6, 1, 29, 196]
[1, 1, 18, 96, 1, 2, 18, 96, 1, 3, 18, 96, 1, 5, 18, 96, 2, 1, 18, 96, 2, 2, 18, 96, 3, 1, 18, 96, 3, 2, 18, 96, 4, 0, 18, 96, 4, 1, 18, 96]
[3, 2, 18, 96, 4, 5, 18, 96]
[4, 0, 29, 196]
[0, 0, 29, 205, 0, 7, 29, 205]
[2, 6, 29, 196]
[2, 0, 29, 205]
[4, 6, 29, 192, 3, 7, 29, 205]
[3, 0, 29, 192]
[0, 7, 12, 67, 1, 0, 12, 67]
[6, 0, 29, 205, 0, 0, 29, 192, 0, 0, 29, 202]
[6, 1, 29, 196, 6, 6, 29, 196, 0, 5, 29, 196]
[1, 1, 18, 96, 1, 1, 18, 96, 1, 4, 18, 96, 2, 2, 18, 96, 2, 6, 18, 96, 3, 1, 18, 96]
[5, 0, 18, 96, 2, 0, 18, 96, 2, 2, 18, 96]
[1, 5, 18, 96, 2, 3, 18, 96, 2, 5, 18, 96, 3, 2, 18, 96, 3, 5, 18, 96, 4, 1, 18, 96, 4, 2, 18, 96, 5, 1, 18, 96]
[0, 0, 29, 205, 0, 7, 29, 205, 2, 1, 29, 205]
[4, 2, 29, 202]
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/spring/trn.pkl`

- Type: `.pkl`
- Size: `4.3 KB`
- Useful for mapping: `unknown`
- Notes: Numeric flat 4-tuples; useful for layout and id-space overlap

```text
top-level: list
len: 109
[3, 6, 2, 28, 3, 6, 2, 27, 3, 6, 29, 196, 3, 6, 24, 156, 3, 6, 2, 28, 3, 6, 11, 61]
[3, 1, 18, 115, 3, 2, 18, 115, 3, 3, 18, 115, 3, 4, 18, 115, 3, 4, 2, 28]
[6, 2, 2, 28, 6, 2, 3, 34, 6, 2, 2, 27, 6, 2, 11, 65, 6, 2, 13, 74, 6, 2, 16, 86]
[2, 0, 16, 83, 2, 0, 0, 12, 2, 0, 13, 78]
[0, 5, 22, 139, 0, 5, 11, 61, 0, 5, 29, 196, 0, 5, 29, 192]
[6, 6, 2, 28, 6, 6, 3, 34, 6, 6, 24, 156, 6, 6, 16, 84, 6, 6, 29, 196]
[1, 2, 2, 28, 1, 2, 3, 34, 1, 2, 13, 75, 1, 2, 1, 18, 1, 2, 29, 196]
[0, 2, 2, 28, 0, 2, 3, 34, 0, 2, 2, 27, 0, 2, 24, 156, 0, 2, 11, 65, 0, 2, 11, 61]
[0, 3, 24, 156, 0, 3, 3, 34, 0, 3, 2, 28, 0, 3, 2, 27]
[0, 6, 2, 28, 0, 6, 3, 34, 0, 6, 24, 156, 0, 6, 13, 75, 0, 6, 29, 196]
[4, 7, 18, 104, 4, 7, 16, 86]
[4, 0, 29, 196, 4, 0, 2, 27]
[2, 0, 18, 96, 2, 0, 13, 77, 2, 0, 24, 155, 2, 0, 3, 34, 2, 0, 2, 27]
[4, 1, 29, 202, 4, 1, 29, 205, 4, 1, 13, 76]
[0, 2, 1, 18, 0, 2, 11, 61, 0, 2, 22, 139, 0, 2, 3, 34, 0, 2, 2, 28, 0, 2, 29, 196]
[3, 2, 29, 205, 3, 2, 0, 6, 3, 2, 11, 61]
[4, 1, 21, 132, 4, 1, 11, 61, 4, 1, 13, 74, 4, 1, 29, 196]
[4, 7, 29, 205, 4, 7, 16, 84, 4, 7, 22, 141]
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/trn.pkl`

- Type: `.pkl`
- Size: `33.8 KB`
- Useful for mapping: `unknown`
- Notes: Numeric flat 4-tuples; useful for layout and id-space overlap

```text
top-level: list
len: 1125
[6, 1, 29, 192]
[4, 1, 29, 196, 4, 2, 29, 196, 6, 1, 29, 196]
[1, 1, 18, 96, 1, 2, 18, 96, 1, 3, 18, 96, 1, 5, 18, 96, 2, 1, 18, 96, 2, 2, 18, 96, 3, 1, 18, 96, 3, 2, 18, 96, 4, 0, 18, 96, 4, 1, 18, 96]
[3, 2, 18, 96, 4, 5, 18, 96]
[4, 0, 29, 196]
[0, 0, 29, 205, 0, 7, 29, 205]
[2, 6, 29, 196]
[2, 0, 29, 205]
[4, 6, 29, 192, 3, 7, 29, 205]
[3, 0, 29, 192]
[1, 7, 3, 34, 1, 7, 2, 27, 1, 7, 2, 28]
[0, 7, 12, 67, 1, 0, 12, 67]
[2, 2, 13, 75, 3, 1, 18, 96, 3, 1, 18, 96]
[6, 0, 29, 205, 0, 0, 29, 192, 0, 0, 29, 202]
[6, 1, 29, 196, 6, 6, 29, 196, 0, 5, 29, 196]
[1, 1, 18, 96, 1, 1, 18, 96, 1, 4, 18, 96, 2, 2, 18, 96, 2, 6, 18, 96, 3, 1, 18, 96]
[5, 0, 18, 96, 2, 0, 18, 96, 2, 2, 18, 96]
[6, 7, 2, 27, 6, 7, 3, 34, 6, 7, 9, 56, 6, 7, 16, 84, 6, 7, 24, 155, 6, 7, 24, 156]
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/winter/trn.pkl`

- Type: `.pkl`
- Size: `26.2 KB`
- Useful for mapping: `unknown`
- Notes: Numeric flat 4-tuples; useful for layout and id-space overlap

```text
top-level: list
len: 873
[5, 6, 2, 28, 5, 6, 3, 34, 5, 6, 2, 27, 5, 6, 24, 155, 5, 6, 29, 196]
[5, 1, 18, 96, 5, 1, 18, 96, 0, 1, 18, 96, 0, 2, 18, 96]
[3, 0, 18, 96, 3, 0, 18, 96, 3, 5, 18, 96, 4, 0, 18, 96]
[2, 0, 29, 192, 3, 5, 29, 192]
[6, 5, 29, 205, 6, 6, 29, 205]
[0, 1, 29, 205, 1, 0, 29, 205]
[5, 7, 29, 192]
[6, 1, 18, 96, 6, 3, 18, 96, 6, 4, 18, 96, 6, 7, 18, 96, 0, 0, 18, 96, 0, 2, 18, 96, 0, 3, 18, 96, 1, 1, 18, 96, 1, 2, 18, 96, 1, 3, 18, 96]
[0, 6, 29, 196, 2, 1, 29, 196]
[0, 5, 29, 196]
[1, 4, 18, 96, 1, 4, 18, 96, 1, 7, 18, 96, 3, 3, 18, 96, 3, 3, 18, 96, 4, 3, 18, 96, 4, 4, 18, 96]
[6, 7, 29, 196, 2, 0, 2, 28, 3, 1, 29, 196]
[3, 5, 18, 96, 4, 0, 18, 96]
[1, 1, 18, 96, 2, 7, 18, 96, 3, 2, 18, 96]
[0, 1, 29, 205]
[6, 1, 29, 196, 6, 6, 29, 196, 0, 5, 29, 196]
[2, 5, 18, 96, 3, 0, 18, 96, 3, 0, 18, 96]
[3, 0, 29, 192]
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/sp/daytime/trn.pkl`

- Type: `.pkl`
- Size: `16.7 KB`
- Useful for mapping: `unknown`
- Notes: Numeric flat 4-tuples; useful for layout and id-space overlap

```text
top-level: list
len: 872
[5, 5, 30, 204]
[2, 3, 30, 204]
[5, 5, 30, 204]
[5, 5, 21, 137, 0, 5, 21, 137, 2, 3, 21, 137, 2, 4, 21, 137, 3, 4, 21, 137, 3, 5, 21, 137, 4, 5, 21, 137, 5, 2, 21, 137]
[2, 2, 30, 217]
[5, 5, 18, 119]
[2, 2, 30, 217, 2, 2, 30, 208, 5, 2, 30, 208]
[1, 4, 30, 209, 1, 4, 30, 209]
[2, 3, 30, 217]
[1, 4, 18, 100, 3, 2, 18, 100, 3, 3, 18, 100, 3, 4, 18, 100]
[1, 3, 30, 217]
[0, 5, 30, 208]
[6, 3, 30, 217]
[1, 3, 30, 208]
[5, 3, 30, 208]
[4, 3, 12, 69, 4, 5, 27, 179, 5, 3, 12, 69, 5, 5, 18, 111, 6, 3, 0, 5]
[3, 4, 30, 217]
[3, 2, 30, 208]
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/sp/night/trn.pkl`

- Type: `.pkl`
- Size: `3.8 KB`
- Useful for mapping: `unknown`
- Notes: Numeric flat 4-tuples; useful for layout and id-space overlap

```text
top-level: list
len: 63
[4, 7, 13, 80, 4, 7, 13, 77, 4, 7, 30, 214, 4, 7, 30, 208, 4, 7, 30, 204, 4, 7, 0, 6, 4, 7, 0, 14, 4, 7, 16, 90, 4, 7, 16, 87]
[5, 7, 13, 80, 5, 7, 13, 76, 5, 7, 16, 90, 5, 7, 16, 96, 5, 7, 27, 181, 5, 7, 3, 34, 5, 7, 31, 226, 5, 7, 31, 221]
[0, 7, 13, 80, 0, 7, 30, 214, 0, 7, 30, 208, 0, 7, 30, 217, 0, 7, 0, 5, 0, 7, 11, 68, 0, 7, 24, 165, 0, 7, 3, 38, 0, 7, 3, 35, 0, 7, 27, 180]
[2, 1, 11, 67, 2, 1, 0, 11, 2, 1, 30, 213, 2, 1, 16, 91, 2, 1, 2, 29]
[2, 1, 13, 79, 2, 1, 30, 213, 2, 1, 0, 11, 2, 1, 2, 29, 2, 1, 24, 166, 2, 1, 3, 35]
[3, 3, 24, 165, 3, 3, 2, 28, 3, 3, 3, 35, 3, 3, 22, 144, 3, 3, 11, 68, 3, 3, 11, 64]
[0, 6, 30, 208, 0, 6, 16, 89, 0, 6, 13, 76, 0, 6, 3, 34, 0, 6, 0, 5, 0, 6, 24, 171]
[2, 7, 13, 80, 2, 7, 21, 137, 2, 7, 30, 208, 2, 7, 30, 217, 2, 7, 0, 14, 2, 7, 3, 35, 2, 7, 27, 180]
[1, 0, 13, 80, 1, 0, 30, 214, 1, 0, 30, 208, 1, 0, 30, 217, 1, 0, 0, 5, 1, 0, 11, 68, 1, 0, 24, 165, 1, 0, 3, 38, 1, 0, 3, 35, 1, 0, 27, 180]
[0, 0, 13, 77, 0, 0, 16, 90, 0, 0, 16, 87, 0, 0, 31, 226, 0, 0, 31, 221, 0, 0, 27, 180, 0, 0, 3, 35]
[4, 6, 24, 166, 4, 6, 2, 29, 4, 6, 11, 64, 4, 6, 16, 87, 4, 6, 30, 208, 4, 6, 30, 217]
[3, 0, 13, 77, 3, 0, 11, 68, 3, 0, 0, 14, 3, 0, 22, 144, 3, 0, 31, 221, 3, 0, 30, 217, 3, 0, 18, 111, 3, 0, 24, 166]
[5, 1, 13, 79, 5, 1, 0, 11, 5, 1, 30, 213, 5, 1, 27, 181, 5, 1, 24, 166]
[3, 7, 13, 77, 3, 7, 30, 208, 3, 7, 30, 217, 3, 7, 0, 6, 3, 7, 18, 115, 3, 7, 3, 35, 3, 7, 2, 28]
[1, 7, 13, 80, 1, 7, 30, 214, 1, 7, 30, 208, 1, 7, 3, 38, 1, 7, 3, 35, 1, 7, 2, 28, 1, 7, 0, 14, 1, 7, 16, 90]
[2, 2, 24, 165, 2, 2, 24, 166, 2, 2, 27, 180, 2, 2, 18, 111, 2, 2, 13, 79, 2, 2, 30, 213]
[4, 2, 24, 165, 4, 2, 2, 28, 4, 2, 18, 111, 4, 2, 13, 79, 4, 2, 11, 67, 4, 2, 3, 37]
[3, 6, 30, 214, 3, 6, 30, 204, 3, 6, 30, 217, 3, 6, 13, 80, 3, 6, 13, 77, 3, 6, 16, 87, 3, 6, 16, 90, 3, 6, 24, 166]
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/sp/single/trn.pkl`

- Type: `.pkl`
- Size: `74.5 KB`
- Useful for mapping: `unknown`
- Notes: Numeric flat 4-tuples; useful for layout and id-space overlap

```text
top-level: list
len: 2954
[5, 5, 30, 204]
[4, 7, 30, 208, 2, 1, 30, 208]
[2, 3, 30, 204]
[4, 5, 18, 100, 5, 0, 18, 100, 5, 1, 18, 100, 5, 5, 18, 100, 6, 0, 18, 100, 6, 2, 18, 100, 1, 1, 18, 100, 1, 2, 18, 100, 1, 5, 18, 100, 2, 0, 18, 100]
[5, 5, 30, 204]
[1, 0, 11, 68, 2, 0, 11, 68]
[1, 6, 30, 208, 2, 0, 30, 208, 0, 6, 30, 208, 1, 7, 30, 208]
[5, 5, 21, 137, 0, 5, 21, 137, 2, 3, 21, 137, 2, 4, 21, 137, 3, 4, 21, 137, 3, 5, 21, 137, 4, 5, 21, 137, 5, 2, 21, 137]
[2, 4, 30, 208, 2, 6, 30, 208, 3, 0, 30, 208, 3, 0, 30, 208]
[2, 2, 30, 217]
[5, 5, 18, 119]
[0, 3, 13, 80]
[0, 1, 30, 217, 0, 3, 30, 208, 1, 2, 30, 208]
[2, 2, 30, 217, 2, 2, 30, 208, 5, 2, 30, 208]
[5, 3, 13, 80]
[1, 4, 30, 209, 1, 4, 30, 209]
[3, 1, 30, 217, 6, 4, 30, 217]
[2, 3, 30, 217]
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_SPPC_th=0.915_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `1.9 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 52
[0, 0, 29, 205, 0, 0, 29, 196, 0, 0, 0, 6, 0, 0, 13, 75]
[0, 1, 1, 16, 0, 1, 24, 156, 0, 1, 3, 34, 0, 1, 2, 28]
[0, 2, 29, 196, 0, 2, 13, 74, 0, 2, 11, 65, 0, 2, 24, 155]
[0, 3, 18, 96, 0, 3, 22, 139, 0, 3, 19, 125]
[0, 4, 18, 96, 0, 4, 31, 218, 0, 4, 13, 75]
[0, 5, 18, 96, 0, 5, 29, 197, 0, 5, 16, 86]
[0, 6, 29, 196, 0, 6, 0, 6, 0, 6, 13, 74]
[0, 7, 29, 196, 0, 7, 26, 170, 0, 7, 24, 157, 0, 7, 2, 27]
[1, 2, 13, 78, 1, 2, 0, 7, 1, 2, 29, 196, 1, 2, 24, 156]
[1, 3, 3, 34, 1, 3, 12, 67, 1, 3, 13, 75, 1, 3, 1, 16]
[1, 4, 29, 205, 1, 4, 24, 155, 1, 4, 22, 139]
[1, 5, 13, 74, 1, 5, 11, 65, 1, 5, 29, 196, 1, 5, 18, 106]
[1, 6, 12, 66, 1, 6, 24, 156, 1, 6, 29, 205, 1, 6, 20, 129]
[1, 7, 2, 27, 1, 7, 0, 10, 1, 7, 29, 196, 1, 7, 26, 170]
[2, 0, 29, 192, 2, 0, 24, 155, 2, 0, 3, 34, 2, 0, 2, 28]
[2, 1, 13, 77, 2, 1, 0, 9, 2, 1, 29, 205, 2, 1, 22, 141]
[2, 2, 0, 10, 2, 2, 0, 6, 2, 2, 1, 20, 2, 2, 1, 16]
[2, 3, 13, 78, 2, 3, 13, 75, 2, 3, 11, 65, 2, 3, 11, 61]
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_SPPC_th=0.917_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `3.8 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 104
[0, 0, 29, 205, 0, 0, 13, 78, 0, 0, 1, 20, 0, 0, 24, 156]
[0, 2, 18, 96, 0, 2, 13, 75, 0, 2, 0, 6, 0, 2, 2, 28]
[0, 4, 18, 96, 0, 4, 29, 196, 0, 4, 16, 86, 0, 4, 3, 33]
[0, 5, 18, 96, 0, 5, 24, 155, 0, 5, 31, 218]
[0, 6, 29, 196, 0, 6, 13, 78, 0, 6, 0, 10, 0, 6, 22, 139]
[0, 7, 18, 96, 0, 7, 12, 66, 0, 7, 26, 170, 0, 7, 11, 65]
[1, 0, 29, 197, 1, 0, 13, 77, 1, 0, 1, 19]
[1, 2, 18, 96, 1, 2, 2, 27, 1, 2, 0, 9, 1, 2, 24, 156]
[1, 4, 18, 96, 1, 4, 29, 196, 1, 4, 16, 84, 1, 4, 3, 34]
[1, 5, 18, 96, 1, 5, 24, 155, 1, 5, 31, 219]
[1, 6, 29, 196, 1, 6, 13, 78, 1, 6, 0, 6, 1, 6, 22, 141]
[1, 7, 18, 96, 1, 7, 12, 67, 1, 7, 26, 171, 1, 7, 11, 64]
[1, 1, 29, 205, 1, 1, 29, 196, 1, 1, 0, 10, 1, 1, 13, 75, 1, 1, 24, 156]
[1, 2, 29, 205, 1, 2, 29, 196, 1, 2, 1, 20, 1, 2, 13, 78, 1, 2, 24, 155]
[1, 3, 29, 196, 1, 3, 29, 205, 1, 3, 2, 28, 1, 3, 22, 139, 1, 3, 31, 218]
[1, 4, 29, 196, 1, 4, 29, 205, 1, 4, 0, 13, 1, 4, 13, 74, 1, 4, 24, 156]
[1, 5, 29, 192, 1, 5, 29, 196, 1, 5, 1, 16, 1, 5, 11, 65, 1, 5, 24, 155]
[1, 6, 29, 196, 1, 6, 29, 205, 1, 6, 2, 27, 1, 6, 19, 125, 1, 6, 24, 156]
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_0_SPPC_th=0.915_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `1.7 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 8
['Monday', '(0~3)', 'Television', 'Television:volumeDown', 'Monday', '(0~3)', 'Television', 'Television:setChannel', 'Monday', '(0~3)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint', 'Monday', '(0~3)', 'Light', 'Light:setLevel']
['Monday', '(3~6)', 'AirPurifier', 'AirPurifier:setAirPurifierMode', 'Monday', '(3~6)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Monday', '(3~6)', 'Camera', 'Camera:notification', 'Monday', '(3~6)', 'Blind', 'Blind:windowShade open']
['Monday', '(6~9)', 'Television', 'Television:setChannel', 'Monday', '(6~9)', 'Light', 'Light:setColorTemperature', 'Monday', '(6~9)', 'Fan', 'Fan:switch on', 'Monday', '(6~9)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging']
['Monday', '(9~12)', 'Other', 'None:location', 'Monday', '(9~12)', 'Refrigerator', 'Refrigerator:samsungce.powerCool activate', 'Monday', '(9~12)', 'Oven', 'Oven:signalahead13665.startstopprogramv2 setStartstop']
['Monday', '(12~15)', 'Other', 'None:location', 'Monday', '(12~15)', 'Washer', 'Washer:washerOperatingState setMachineState run', 'Monday', '(12~15)', 'Light', 'Light:setLevel']
['Monday', '(15~18)', 'Other', 'None:location', 'Monday', '(15~18)', 'Television', 'Television:setInputSource', 'Monday', '(15~18)', 'NetworkAudio', 'NetworkAudio:mediaPlayback play']
['Monday', '(18~21)', 'Television', 'Television:setChannel', 'Monday', '(18~21)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint', 'Monday', '(18~21)', 'Light', 'Light:setColorTemperature']
['Monday', '(21~24)', 'Television', 'Television:setChannel', 'Monday', '(21~24)', 'SmartLock', 'SmartLock:lock lock', 'Monday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement homing', 'Monday', '(21~24)', 'Blind', 'Blind:windowShade close']
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_0_SPPC_th=0.917_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `2.6 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 12
['Monday', '(0~3)', 'Television', 'Television:volumeDown', 'Monday', '(0~3)', 'Light', 'Light:switch on', 'Monday', '(0~3)', 'AirPurifier', 'AirPurifier:switch on', 'Monday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning']
['Monday', '(6~9)', 'Other', 'None:location', 'Monday', '(6~9)', 'Light', 'Light:setLevel', 'Monday', '(6~9)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint', 'Monday', '(6~9)', 'Blind', 'Blind:windowShade open']
['Monday', '(12~15)', 'Other', 'None:location', 'Monday', '(12~15)', 'Television', 'Television:setChannel', 'Monday', '(12~15)', 'NetworkAudio', 'NetworkAudio:mediaPlayback play', 'Monday', '(12~15)', 'Camera', 'Camera:imageCapture take']
['Monday', '(15~18)', 'Other', 'None:location', 'Monday', '(15~18)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Monday', '(15~18)', 'Washer', 'Washer:washerOperatingState setMachineState run']
['Monday', '(18~21)', 'Television', 'Television:setChannel', 'Monday', '(18~21)', 'Light', 'Light:switch on', 'Monday', '(18~21)', 'AirConditioner', 'AirConditioner:switch on', 'Monday', '(18~21)', 'Refrigerator', 'Refrigerator:samsungce.powerCool activate']
['Monday', '(21~24)', 'Other', 'None:location', 'Monday', '(21~24)', 'GarageDoor', 'GarageDoor:doorControl close', 'Monday', '(21~24)', 'SmartLock', 'SmartLock:lock lock', 'Monday', '(21~24)', 'Fan', 'Fan:switch on']
['Tuesday', '(0~3)', 'Television', 'Television:setInputSource', 'Tuesday', '(0~3)', 'Light', 'Light:switch off', 'Tuesday', '(0~3)', 'AirPurifier', 'AirPurifier:switch off']
['Tuesday', '(6~9)', 'Other', 'None:location', 'Tuesday', '(6~9)', 'Blind', 'Blind:windowShade close', 'Tuesday', '(6~9)', 'AirConditioner', 'AirConditioner:switch off', 'Tuesday', '(6~9)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning']
['Tuesday', '(12~15)', 'Other', 'None:location', 'Tuesday', '(12~15)', 'Television', 'Television:setChannel', 'Tuesday', '(12~15)', 'NetworkAudio', 'NetworkAudio:audioVolume setVolume', 'Tuesday', '(12~15)', 'Camera', 'Camera:notification']
['Tuesday', '(15~18)', 'Other', 'None:location', 'Tuesday', '(15~18)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Tuesday', '(15~18)', 'Washer', 'Washer:washerOperatingState setMachineState stop']
['Tuesday', '(18~21)', 'Television', 'Television:setChannel', 'Tuesday', '(18~21)', 'Light', 'Light:switch on', 'Tuesday', '(18~21)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint', 'Tuesday', '(18~21)', 'Refrigerator', 'Refrigerator:samsungce.powerFree
['Tuesday', '(21~24)', 'Other', 'None:location', 'Tuesday', '(21~24)', 'GarageDoor', 'GarageDoor:doorControl open', 'Tuesday', '(21~24)', 'SmartLock', 'SmartLock:lock unlock', 'Tuesday', '(21~24)', 'Fan', 'Fan:switch off']
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_1_0_SPPC_th=0.917_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `2.7 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 9
['Tuesday', '(3~6)', 'Television', 'Television:volumeDown', 'Tuesday', '(3~6)', 'Television', 'Television:setChannel', 'Tuesday', '(3~6)', 'AirConditioner', 'AirConditioner:switch on', 'Tuesday', '(3~6)', 'Light', 'Light:setLevel', 'Tuesday', '(3~6)', 'RobotCl
['Tuesday', '(6~9)', 'Television', 'Television:volumeDown', 'Tuesday', '(6~9)', 'Television', 'Television:setChannel', 'Tuesday', '(6~9)', 'AirPurifier', 'AirPurifier:switch on', 'Tuesday', '(6~9)', 'Light', 'Light:switch on', 'Tuesday', '(6~9)', 'RobotCleaner
['Tuesday', '(9~12)', 'Television', 'Television:setChannel', 'Tuesday', '(9~12)', 'Television', 'Television:volumeDown', 'Tuesday', '(9~12)', 'Blind', 'Blind:windowShade open', 'Tuesday', '(9~12)', 'Refrigerator', 'Refrigerator:samsungce.powerCool activate', '
['Tuesday', '(12~15)', 'Television', 'Television:setChannel', 'Tuesday', '(12~15)', 'Television', 'Television:volumeDown', 'Tuesday', '(12~15)', 'AirConditioner', 'AirConditioner:temperatureUp', 'Tuesday', '(12~15)', 'Light', 'Light:setColorTemperature', 'Tues
['Tuesday', '(15~18)', 'Television', 'Television:audioMute unmute', 'Tuesday', '(15~18)', 'Television', 'Television:setChannel', 'Tuesday', '(15~18)', 'AirPurifier', 'AirPurifier:setAirPurifierMode', 'Tuesday', '(15~18)', 'Fan', 'Fan:switch on', 'Tuesday', '(1
['Tuesday', '(18~21)', 'Television', 'Television:setChannel', 'Tuesday', '(18~21)', 'Television', 'Television:volumeDown', 'Tuesday', '(18~21)', 'Blind', 'Blind:windowShade close', 'Tuesday', '(18~21)', 'Oven', 'Oven:signalahead13665.startstopprogramv2 setStar
['Tuesday', '(21~24)', 'Television', 'Television:setChannel', 'Tuesday', '(21~24)', 'Television', 'Television:volumeDown', 'Tuesday', '(21~24)', 'AirConditioner', 'AirConditioner:switch off', 'Tuesday', '(21~24)', 'Light', 'Light:switch off', 'Tuesday', '(21~2
['Wednesday', '(0~3)', 'Television', 'Television:volumeDown', 'Wednesday', '(0~3)', 'Television', 'Television:setChannel', 'Wednesday', '(0~3)', 'Camera', 'Camera:notification', 'Wednesday', '(0~3)', 'Blind', 'Blind:windowShade open', 'Wednesday', '(0~3)', 'Ro
['Wednesday', '(3~6)', 'Television', 'Television:setChannel', 'Wednesday', '(3~6)', 'Television', 'Television:volumeDown', 'Wednesday', '(3~6)', 'AirConditioner', 'AirConditioner:switch on', 'Wednesday', '(3~6)', 'Light', 'Light:setLevel', 'Wednesday', '(3~6)'
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_1_1_SPPC_th=0.917_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `2.2 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 8
['Tuesday', '(18~21)', 'GarageDoor', 'GarageDoor:doorControl open', 'Tuesday', '(18~21)', 'GarageDoor', 'GarageDoor:doorControl close', 'Tuesday', '(18~21)', 'Light', 'Light:switch on', 'Tuesday', '(18~21)', 'Television', 'Television:switch on', 'Tuesday', '(1
['Wednesday', '(0~3)', 'GarageDoor', 'GarageDoor:doorControl close', 'Wednesday', '(0~3)', 'Camera', 'Camera:notification', 'Wednesday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Wednesday', '(0~3)', 'Light', 'Light:switch off'
['Tuesday', '(12~15)', 'Other', 'Other:switch on', 'Tuesday', '(12~15)', 'AirPurifier', 'AirPurifier:switch on', 'Tuesday', '(12~15)', 'AirConditioner', 'AirConditioner:switch on', 'Tuesday', '(12~15)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint', 'T
['Tuesday', '(15~18)', 'Other', 'Other:switch on', 'Tuesday', '(15~18)', 'Fan', 'Fan:switch on', 'Tuesday', '(15~18)', 'Fan', 'Fan:fanSpeed setFanSpeed', 'Tuesday', '(15~18)', 'Light', 'Light:setLevel']
['Tuesday', '(21~24)', 'Other', 'Other:switch on', 'Tuesday', '(21~24)', 'Refrigerator', 'Refrigerator:samsungce.powerCool activate', 'Tuesday', '(21~24)', 'Refrigerator', 'Refrigerator:samsungce.powerFreeze activate', 'Tuesday', '(21~24)', 'Television', 'Tele
['Wednesday', '(0~3)', 'Other', 'Other:switch on', 'Wednesday', '(0~3)', 'Washer', 'Washer:washerOperatingState setMachineState run', 'Wednesday', '(0~3)', 'Washer', 'Washer:washerOperatingState setMachineState stop', 'Wednesday', '(0~3)', 'RobotCleaner', 'Rob
['Wednesday', '(3~6)', 'Other', 'Other:switch on', 'Wednesday', '(3~6)', 'PresenceSensor', 'PresenceSensor:switch on', 'Wednesday', '(3~6)', 'Camera', 'Camera:imageCapture take']
['Wednesday', '(6~9)', 'Other', 'Other:switch on', 'Wednesday', '(6~9)', 'Blind', 'Blind:windowShade open', 'Wednesday', '(6~9)', 'Blind', 'Blind:windowShade close', 'Wednesday', '(6~9)', 'AirConditioner', 'AirConditioner:temperatureUp']
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_1_SPPC_th=0.915_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `1.9 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 8
['Tuesday', '(6~9)', 'Light', 'Light:switch on', 'Tuesday', '(6~9)', 'AirConditioner', 'AirConditioner:setFanMode', 'Tuesday', '(6~9)', 'Television', 'Television:setChannel', 'Tuesday', '(6~9)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning']
['Tuesday', '(9~12)', 'Camera', 'Camera:notification', 'Tuesday', '(9~12)', 'GarageDoor', 'GarageDoor:doorControl open', 'Tuesday', '(9~12)', 'Light', 'Light:setLevel', 'Tuesday', '(9~12)', 'AirPurifier', 'AirPurifier:setAirPurifierMode']
['Tuesday', '(12~15)', 'Television', 'Television:volumeDown', 'Tuesday', '(12~15)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Tuesday', '(12~15)', 'Refrigerator', 'Refrigerator:samsungce.powerCool activate']
['Tuesday', '(15~18)', 'Light', 'Light:setColorTemperature', 'Tuesday', '(15~18)', 'Fan', 'Fan:switch on', 'Tuesday', '(15~18)', 'Television', 'Television:setChannel', 'Tuesday', '(15~18)', 'Other', 'Other:notification']
['Tuesday', '(18~21)', 'GarageDoor', 'GarageDoor:doorControl close', 'Tuesday', '(18~21)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Tuesday', '(18~21)', 'Television', 'Television:volumeDown', 'Tuesday', '(18~21)', 'PresenceSensor', 'Pr
['Tuesday', '(21~24)', 'Blind', 'Blind:windowShade close', 'Tuesday', '(21~24)', 'AirConditioner', 'AirConditioner:switch on', 'Tuesday', '(21~24)', 'Television', 'Television:setChannel', 'Tuesday', '(21~24)', 'SmartLock', 'SmartLock:lock lock']
['Wednesday', '(0~3)', 'Television', 'Television:audioMute unmute', 'Wednesday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Wednesday', '(0~3)', 'Camera', 'Camera:notification', 'Wednesday', '(0~3)', 'Blind', 'Blind:windowShade 
['Wednesday', '(3~6)', 'Light', 'Light:switch off', 'Wednesday', '(3~6)', 'AirConditioner', 'AirConditioner:switch off', 'Wednesday', '(3~6)', 'Television', 'Television:volumeDown', 'Wednesday', '(3~6)', 'Refrigerator', 'Refrigerator:samsungce.powerFreeze acti
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_2_0_SPPC_th=0.917_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `2.0 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 9
['Wednesday', '(3~6)', 'Other', 'None:location', 'Wednesday', '(3~6)', 'AirPurifier', 'AirPurifier:setAirPurifierMode', 'Wednesday', '(3~6)', 'Fan', 'Fan:switch on', 'Wednesday', '(3~6)', 'Light', 'Light:switch on']
['Wednesday', '(6~9)', 'Other', 'None:location', 'Wednesday', '(6~9)', 'Television', 'Television:setChannel', 'Wednesday', '(6~9)', 'Television', 'Television:volumeDown', 'Wednesday', '(6~9)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning']
['Wednesday', '(9~12)', 'Other', 'None:location', 'Wednesday', '(9~12)', 'Camera', 'Camera:notification', 'Wednesday', '(9~12)', 'Blind', 'Blind:windowShade open', 'Wednesday', '(9~12)', 'Blind', 'Blind:windowShade close']
['Wednesday', '(12~15)', 'Other', 'None:location', 'Wednesday', '(12~15)', 'AirConditioner', 'AirConditioner:switch on', 'Wednesday', '(12~15)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint', 'Wednesday', '(12~15)', 'Light', 'Light:setLevel']
['Wednesday', '(15~18)', 'Other', 'None:location', 'Wednesday', '(15~18)', 'Fan', 'Fan:switch on', 'Wednesday', '(15~18)', 'Fan', 'Fan:fanSpeed setFanSpeed', 'Wednesday', '(15~18)', 'Television', 'Television:setChannel']
['Wednesday', '(18~21)', 'Other', 'None:location', 'Wednesday', '(18~21)', 'Television', 'Television:setChannel', 'Wednesday', '(18~21)', 'Television', 'Television:audioMute unmute', 'Wednesday', '(18~21)', 'NetworkAudio', 'NetworkAudio:mediaPlayback play']
['Wednesday', '(21~24)', 'Other', 'None:location', 'Wednesday', '(21~24)', 'GarageDoor', 'GarageDoor:doorControl close', 'Wednesday', '(21~24)', 'SmartLock', 'SmartLock:lock lock', 'Wednesday', '(21~24)', 'Camera', 'Camera:imageCapture take']
['Thursday', '(0~3)', 'Other', 'None:location', 'Thursday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Thursday', '(0~3)', 'Refrigerator', 'Refrigerator:samsungce.powerCool activate', 'Thursday', '(0~3)', 'Heater', 'Heater:switc
['Thursday', '(3~6)', 'Other', 'None:location', 'Thursday', '(3~6)', 'Television', 'Television:setChannel', 'Thursday', '(3~6)', 'Television', 'Television:volumeDown', 'Thursday', '(3~6)', 'Light', 'Light:switch off']
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_2_1_SPPC_th=0.917_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `2.4 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 10
['Wednesday', '(0~3)', 'AirPurifier', 'AirPurifier:setAirPurifierMode', 'Wednesday', '(0~3)', 'AirConditioner', 'AirConditioner:switch on', 'Wednesday', '(0~3)', 'Light', 'Light:switch on', 'Wednesday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMov
['Wednesday', '(21~24)', 'Television', 'Television:volumeDown', 'Wednesday', '(21~24)', 'Television', 'Television:setChannel', 'Wednesday', '(21~24)', 'Light', 'Light:setLevel', 'Wednesday', '(21~24)', 'NetworkAudio', 'NetworkAudio:mediaPlayback play', 'Wednes
['Thursday', '(3~6)', 'Television', 'Television:volumeDown', 'Thursday', '(3~6)', 'Television', 'Television:setChannel', 'Thursday', '(3~6)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint', 'Thursday', '(3~6)', 'RobotCleaner', 'RobotCleaner:setRobotClea
['Wednesday', '(18~21)', 'Other', 'None:location', 'Wednesday', '(18~21)', 'Light', 'Light:switch on', 'Wednesday', '(18~21)', 'AirPurifier', 'AirPurifier:switch on', 'Wednesday', '(18~21)', 'Heater', 'Heater:switch on', 'Wednesday', '(18~21)', 'RobotCleaner',
['Wednesday', '(21~24)', 'Other', 'None:location', 'Wednesday', '(21~24)', 'Camera', 'Camera:imageCapture take', 'Wednesday', '(21~24)', 'SmartLock', 'SmartLock:lock lock', 'Wednesday', '(21~24)', 'GarageDoor', 'GarageDoor:doorControl close']
['Wednesday', '(6~9)', 'Other', 'None:location', 'Wednesday', '(6~9)', 'Light', 'Light:switch off', 'Wednesday', '(6~9)', 'AirConditioner', 'AirConditioner:switch off', 'Wednesday', '(6~9)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging']
['Wednesday', '(0~3)', 'Other', 'None:location', 'Wednesday', '(0~3)', 'Siren', 'Siren:alarm both', 'Wednesday', '(0~3)', 'PresenceSensor', 'PresenceSensor:switch on']
['Wednesday', '(3~6)', 'Other', 'Other:notification', 'Wednesday', '(3~6)', 'Television', 'Television:volumeDown', 'Wednesday', '(3~6)', 'NetworkAudio', 'NetworkAudio:audioVolume setVolume', 'Wednesday', '(3~6)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMo
['Wednesday', '(3~6)', 'Other', 'None:location', 'Wednesday', '(6~9)', 'Other', 'None:location', 'Wednesday', '(6~9)', 'Light', 'Light:switch on', 'Wednesday', '(6~9)', 'AirPurifier', 'AirPurifier:setAirPurifierMode']
['Wednesday', '(3~6)', 'Other', 'None:location', 'Wednesday', '(3~6)', 'Camera', 'Camera:notification', 'Wednesday', '(3~6)', 'SmartLock', 'SmartLock:lock unlock', 'Wednesday', '(3~6)', 'GarageDoor', 'GarageDoor:doorControl open']
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_2_SPPC_th=0.915_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `1.7 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 8
['Wednesday', '(6~9)', 'AirConditioner', 'AirConditioner:switch on', 'Wednesday', '(6~9)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint', 'Wednesday', '(6~9)', 'AirPurifier', 'AirPurifier:switch on', 'Wednesday', '(6~9)', 'AirPurifier', 'AirPurifier:se
['Wednesday', '(9~12)', 'Light', 'Light:switch on', 'Wednesday', '(9~12)', 'Light', 'Light:setLevel', 'Wednesday', '(9~12)', 'Fan', 'Fan:switch on', 'Wednesday', '(9~12)', 'Fan', 'Fan:fanSpeed setFanSpeed']
['Wednesday', '(12~15)', 'Television', 'Television:switch on', 'Wednesday', '(12~15)', 'Television', 'Television:setChannel', 'Wednesday', '(12~15)', 'NetworkAudio', 'NetworkAudio:mediaPlayback play', 'Wednesday', '(12~15)', 'NetworkAudio', 'NetworkAudio:audio
['Wednesday', '(15~18)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Wednesday', '(15~18)', 'Washer', 'Washer:washerOperatingState setMachineState run', 'Wednesday', '(15~18)', 'Refrigerator', 'Refrigerator:samsungce.powerCool activate']
['Wednesday', '(18~21)', 'Light', 'Light:switch on', 'Wednesday', '(18~21)', 'Light', 'Light:setColorTemperature', 'Wednesday', '(18~21)', 'Television', 'Television:setChannel', 'Wednesday', '(18~21)', 'NetworkAudio', 'NetworkAudio:mediaPlayback play']
['Wednesday', '(21~24)', 'SmartLock', 'SmartLock:lock lock', 'Wednesday', '(21~24)', 'Camera', 'Camera:notification', 'Wednesday', '(21~24)', 'GarageDoor', 'GarageDoor:doorControl close']
['Thursday', '(0~3)', 'AirPurifier', 'AirPurifier:setAirPurifierMode', 'Thursday', '(0~3)', 'Fan', 'Fan:switch on', 'Thursday', '(0~3)', 'Television', 'Television:volumeDown']
['Thursday', '(6~9)', 'PresenceSensor', 'PresenceSensor:switch on', 'Thursday', '(6~9)', 'Light', 'Light:switch on', 'Thursday', '(6~9)', 'AirConditioner', 'AirConditioner:switch on']
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_3_0_SPPC_th=0.917_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `1.9 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 10
['Thursday', '(3~6)', 'Other', 'None:location', 'Thursday', '(3~6)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint', 'Thursday', '(3~6)', 'Light', 'Light:switch on', 'Thursday', '(3~6)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning']
['Thursday', '(6~9)', 'Other', 'None:location', 'Thursday', '(6~9)', 'Television', 'Television:setChannel', 'Thursday', '(6~9)', 'AirPurifier', 'AirPurifier:setAirPurifierMode', 'Thursday', '(6~9)', 'Light', 'Light:setLevel']
['Thursday', '(9~12)', 'Television', 'Television:volumeDown', 'Thursday', '(9~12)', 'Fan', 'Fan:switch on', 'Thursday', '(9~12)', 'Blind', 'Blind:windowShade open']
['Thursday', '(12~15)', 'Other', 'None:location', 'Thursday', '(12~15)', 'Refrigerator', 'Refrigerator:samsungce.powerCool activate', 'Thursday', '(12~15)', 'Oven', 'Oven:signalahead13665.startstopprogramv2 setStartstop']
['Thursday', '(15~18)', 'Television', 'Television:audioMute unmute', 'Thursday', '(15~18)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Thursday', '(15~18)', 'Camera', 'Camera:notification']
['Thursday', '(18~21)', 'Other', 'None:location', 'Thursday', '(18~21)', 'SmartLock', 'SmartLock:lock lock', 'Thursday', '(18~21)', 'GarageDoor', 'GarageDoor:doorControl close']
['Thursday', '(21~24)', 'Television', 'Television:setInputSource', 'Thursday', '(21~24)', 'NetworkAudio', 'NetworkAudio:mediaPlayback play', 'Thursday', '(21~24)', 'Light', 'Light:switch off']
['Friday', '(0~3)', 'Other', 'None:location', 'Friday', '(0~3)', 'AirConditioner', 'AirConditioner:switch off', 'Friday', '(0~3)', 'Television', 'Television:volumeDown']
['Friday', '(6~9)', 'Other', 'None:location', 'Friday', '(6~9)', 'Washer', 'Washer:washerOperatingState setMachineState run', 'Friday', '(6~9)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning']
['Friday', '(9~12)', 'Television', 'Television:setChannel', 'Friday', '(9~12)', 'Fan', 'Fan:switch on', 'Friday', '(9~12)', 'Blind', 'Blind:windowShade close']
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_3_1_SPPC_th=0.917_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `2.3 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 10
['Thursday', '(15~18)', 'NetworkAudio', 'NetworkAudio:audioMute mute', 'Thursday', '(15~18)', 'Light', 'Light:setLevel', 'Thursday', '(15~18)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint', 'Thursday', '(15~18)', 'AirPurifier', 'AirPurifier:setAirPuri
['Thursday', '(6~9)', 'Light', 'Light:setLevel', 'Thursday', '(6~9)', 'Television', 'Television:switch on', 'Thursday', '(6~9)', 'Television', 'Television:setChannel', 'Thursday', '(6~9)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Thurs
['Thursday', '(3~6)', 'Television', 'Television:setInputSource', 'Thursday', '(3~6)', 'AirPurifier', 'AirPurifier:setAirPurifierMode', 'Thursday', '(3~6)', 'Light', 'Light:switch on', 'Thursday', '(3~6)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement c
['Thursday', '(21~24)', 'Other', 'None:location', 'Thursday', '(21~24)', 'Camera', 'Camera:notification', 'Thursday', '(21~24)', 'GarageDoor', 'GarageDoor:doorControl close', 'Thursday', '(21~24)', 'SmartLock', 'SmartLock:lock lock']
['Friday', '(6~9)', 'Other', 'None:location', 'Friday', '(6~9)', 'Light', 'Light:switch on', 'Friday', '(6~9)', 'AirConditioner', 'AirConditioner:setFanMode', 'Friday', '(6~9)', 'Television', 'Television:setChannel']
['Friday', '(9~12)', 'Other', 'None:location', 'Friday', '(9~12)', 'Refrigerator', 'Refrigerator:samsungce.powerCool activate', 'Friday', '(9~12)', 'Washer', 'Washer:washerOperatingState setMachineState run']
['Friday', '(12~15)', 'Other', 'None:location', 'Friday', '(12~15)', 'Oven', 'Oven:signalahead13665.startstopprogramv2 setStartstop', 'Friday', '(12~15)', 'Light', 'Light:setColorTemperature']
['Thursday', '(18~21)', 'Other', 'Other:notification', 'Thursday', '(18~21)', 'Fan', 'Fan:switch on', 'Thursday', '(18~21)', 'Fan', 'Fan:fanSpeed setFanSpeed', 'Thursday', '(18~21)', 'Television', 'Television:setInputSource']
['Thursday', '(3~6)', 'Other', 'None:location', 'Thursday', '(3~6)', 'Blind', 'Blind:windowShade open', 'Thursday', '(3~6)', 'Blind', 'Blind:windowShade close', 'Thursday', '(3~6)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerTurboMode on']
['Thursday', '(6~9)', 'Other', 'None:location', 'Thursday', '(6~9)', 'PresenceSensor', 'PresenceSensor:switch on', 'Thursday', '(6~9)', 'Camera', 'Camera:imageCapture take', 'Thursday', '(6~9)', 'GarageDoor', 'GarageDoor:doorControl open']
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_3_SPPC_th=0.915_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `1.5 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 6
['Thursday', '(6~9)', 'Light', 'Light:switch on', 'Thursday', '(6~9)', 'AirConditioner', 'AirConditioner:switch on', 'Thursday', '(6~9)', 'AirPurifier', 'AirPurifier:switch on', 'Thursday', '(6~9)', 'Television', 'Television:setChannel', 'Thursday', '(6~9)', '
['Thursday', '(9~12)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Thursday', '(9~12)', 'Camera', 'Camera:notification', 'Thursday', '(9~12)', 'Light', 'Light:setLevel', 'Thursday', '(9~12)', 'AirConditioner', 'AirConditioner:setCoolingSe
['Thursday', '(12~15)', 'NetworkAudio', 'NetworkAudio:switch on', 'Thursday', '(12~15)', 'NetworkAudio', 'NetworkAudio:mediaPlayback play', 'Thursday', '(12~15)', 'Light', 'Light:switch off', 'Thursday', '(12~15)', 'AirPurifier', 'AirPurifier:setAirPurifierMod
['Thursday', '(15~18)', 'GarageDoor', 'GarageDoor:doorControl open', 'Thursday', '(15~18)', 'SmartLock', 'SmartLock:lock unlock', 'Thursday', '(15~18)', 'Television', 'Television:setChannel', 'Thursday', '(15~18)', 'Television', 'Television:volumeDown']
['Thursday', '(18~21)', 'Heater', 'Heater:switch on', 'Thursday', '(18~21)', 'Heater', 'Heater:setHeatingSetpoint', 'Thursday', '(18~21)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Thursday', '(18~21)', 'Camera', 'Camera:imageCapture ta
['Thursday', '(21~24)', 'Light', 'Light:switch on', 'Thursday', '(21~24)', 'NetworkAudio', 'NetworkAudio:mediaPlayback stop', 'Thursday', '(21~24)', 'SmartLock', 'SmartLock:lock lock', 'Thursday', '(21~24)', 'Television', 'Television:switch off']
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_4_0_SPPC_th=0.917_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `2.1 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 7
['Friday', '(0~3)', 'AirConditioner', 'AirConditioner:switch on', 'Friday', '(0~3)', 'AirConditioner', 'AirConditioner:temperatureUp', 'Friday', '(0~3)', 'Television', 'Television:switch on', 'Friday', '(0~3)', 'Television', 'Television:setChannel', 'Friday', 
['Friday', '(6~9)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Friday', '(6~9)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Friday', '(6~9)', 'Light', 'Light:switch on', 'Friday', '(6~9)', 'Light', 'Light:setLevel'
['Friday', '(9~12)', 'AirPurifier', 'AirPurifier:switch on', 'Friday', '(9~12)', 'AirPurifier', 'AirPurifier:setAirPurifierMode', 'Friday', '(9~12)', 'Television', 'Television:switch on', 'Friday', '(9~12)', 'Television', 'Television:setChannel', 'Friday', '(9
['Friday', '(12~15)', 'Washer', 'Washer:washerOperatingState setMachineState run', 'Friday', '(12~15)', 'Washer', 'Washer:washerOperatingState setMachineState stop', 'Friday', '(12~15)', 'Refrigerator', 'Refrigerator:samsungce.powerCool activate', 'Friday', '(
['Friday', '(15~18)', 'Projector', 'Projector:switch on', 'Friday', '(15~18)', 'Projector', 'Projector:custom.soundmode setSoundMode', 'Friday', '(15~18)', 'NetworkAudio', 'NetworkAudio:mediaPlayback play', 'Friday', '(15~18)', 'NetworkAudio', 'NetworkAudio:au
['Friday', '(18~21)', 'Camera', 'Camera:notification', 'Friday', '(18~21)', 'SmartLock', 'SmartLock:lock lock', 'Friday', '(18~21)', 'GarageDoor', 'GarageDoor:doorControl close', 'Friday', '(18~21)', 'Television', 'Television:audioMute unmute', 'Friday', '(18~
['Friday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Friday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Friday', '(21~24)', 'Blind', 'Blind:windowShade close', 'Friday', '(21~24)', 'Blind', 
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_4_1_SPPC_th=0.917_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `1.3 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 5
['Friday', '(18~21)', 'Television', 'Television:setChannel', 'Friday', '(18~21)', 'Television', 'Television:volumeDown', 'Friday', '(18~21)', 'Light', 'Light:setLevel', 'Friday', '(18~21)', 'AirConditioner', 'AirConditioner:switch on', 'Friday', '(18~21)', 'Ai
['Saturday', '(9~12)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Saturday', '(9~12)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Saturday', '(9~12)', 'Washer', 'Washer:washerOperatingState setMachineState run', 'S
['Sunday', '(15~18)', 'Camera', 'Camera:notification', 'Sunday', '(15~18)', 'SmartLock', 'SmartLock:lock lock', 'Sunday', '(15~18)', 'Blind', 'Blind:windowShade open', 'Sunday', '(15~18)', 'Blind', 'Blind:windowShade close']
['Monday', '(6~9)', 'AirPurifier', 'AirPurifier:setAirPurifierMode', 'Monday', '(6~9)', 'Fan', 'Fan:switch on', 'Monday', '(6~9)', 'Fan', 'Fan:fanSpeed setFanSpeed', 'Monday', '(6~9)', 'Refrigerator', 'Refrigerator:samsungce.powerCool activate']
['Tuesday', '(12~15)', 'Oven', 'Oven:signalahead13665.startstopprogramv2 setStartstop', 'Tuesday', '(12~15)', 'Light', 'Light:setColorTemperature', 'Tuesday', '(12~15)', 'NetworkAudio', 'NetworkAudio:mediaPlayback play', 'Tuesday', '(12~15)', 'NetworkAudio', '
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_4_SPPC_th=0.915_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `2.3 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 9
['Friday', '(3~6)', 'Television', 'Television:setChannel', 'Friday', '(3~6)', 'Light', 'Light:switch on', 'Friday', '(3~6)', 'Blind', 'Blind:windowShade open', 'Friday', '(3~6)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Friday', '(3~6)
['Friday', '(6~9)', 'Television', 'Television:setChannel', 'Friday', '(6~9)', 'Light', 'Light:setLevel', 'Friday', '(6~9)', 'AirPurifier', 'AirPurifier:switch on', 'Friday', '(6~9)', 'NetworkAudio', 'NetworkAudio:mediaPlayback play', 'Friday', '(6~9)', 'Blind'
['Friday', '(18~21)', 'Television', 'Television:audioMute unmute', 'Friday', '(18~21)', 'Light', 'Light:switch off', 'Friday', '(18~21)', 'Refrigerator', 'Refrigerator:samsungce.powerCool activate', 'Friday', '(18~21)', 'RobotCleaner', 'RobotCleaner:setRobotCl
['Friday', '(0~3)', 'Television', 'Television:volumeDown', 'Friday', '(0~3)', 'Television', 'Television:audioMute unmute', 'Friday', '(0~3)', 'AirConditioner', 'AirConditioner:switch on', 'Friday', '(0~3)', 'SmartLock', 'SmartLock:lock lock']
['Friday', '(6~9)', 'Television', 'Television:volumeDown', 'Friday', '(6~9)', 'Projector', 'Projector:custom.soundmode setSoundMode', 'Friday', '(6~9)', 'Washer', 'Washer:washerOperatingState setMachineState run', 'Friday', '(6~9)', 'Light', 'Light:setColorTem
['Friday', '(21~24)', 'Television', 'Television:volumeDown', 'Friday', '(21~24)', 'NetworkAudio', 'NetworkAudio:switch on', 'Friday', '(21~24)', 'Camera', 'Camera:notification', 'Friday', '(21~24)', 'Blind', 'Blind:windowShadePreset presetPosition']
['Friday', '(12~15)', 'AirConditioner', 'AirConditioner:switch on', 'Friday', '(12~15)', 'Fan', 'Fan:switch on', 'Friday', '(12~15)', 'Refrigerator', 'Refrigerator:samsungce.powerFreeze activate', 'Friday', '(12~15)', 'Television', 'Television:setPictureMode']
['Saturday', '(3~6)', 'Other', 'Other:notification', 'Saturday', '(3~6)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Saturday', '(3~6)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Saturday', '(3~6)', 'Light', 'Ligh
['Saturday', '(6~9)', 'Other', 'Other:notification', 'Saturday', '(6~9)', 'AirPurifier', 'AirPurifier:setAirPurifierMode', 'Saturday', '(6~9)', 'Television', 'Television:setChannel', 'Saturday', '(6~9)', 'Fan', 'Fan:fanSpeed setFanSpeed']
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_5_SPPC_th=0.915_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `1.9 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 7
['Saturday', '(0~3)', 'Television', 'Television:volumeDown', 'Saturday', '(0~3)', 'AirConditioner', 'AirConditioner:switch on', 'Saturday', '(0~3)', 'AirPurifier', 'AirPurifier:switch on', 'Saturday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovem
['Saturday', '(6~9)', 'Light', 'Light:switch on', 'Saturday', '(6~9)', 'Television', 'Television:setChannel', 'Saturday', '(6~9)', 'Television', 'Television:audioMute unmute', 'Saturday', '(6~9)', 'Fan', 'Fan:switch on', 'Saturday', '(6~9)', 'Blind', 'Blind:wi
['Saturday', '(12~15)', 'Oven', 'Oven:switch on', 'Saturday', '(12~15)', 'Washer', 'Washer:washerOperatingState setMachineState run', 'Saturday', '(12~15)', 'Light', 'Light:setLevel', 'Saturday', '(12~15)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint'
['Saturday', '(18~21)', 'Television', 'Television:volumeDown', 'Saturday', '(18~21)', 'NetworkAudio', 'NetworkAudio:mediaPlayback play', 'Saturday', '(18~21)', 'Camera', 'Camera:notification', 'Saturday', '(18~21)', 'GarageDoor', 'GarageDoor:doorControl open']
['Saturday', '(21~24)', 'Heater', 'Heater:switch on', 'Saturday', '(21~24)', 'Television', 'Television:setChannel', 'Saturday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Saturday', '(21~24)', 'RobotCleaner', 'RobotCleaner:set
['Sunday', '(0~3)', 'Light', 'Light:switch off', 'Sunday', '(0~3)', 'AirPurifier', 'AirPurifier:switch off', 'Sunday', '(0~3)', 'Television', 'Television:audioMute unmute', 'Sunday', '(0~3)', 'Television', 'Television:volumeDown']
['Sunday', '(6~9)', 'Blind', 'Blind:windowShade close', 'Sunday', '(6~9)', 'Fan', 'Fan:switch off', 'Sunday', '(6~9)', 'AirConditioner', 'AirConditioner:switch off', 'Sunday', '(6~9)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Sunday', 
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_5_SPPC_th=0.917_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `2.3 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 10
['Saturday', '(0~3)', 'Television', 'Television:volumeDown', 'Saturday', '(0~3)', 'Light', 'Light:switch on', 'Saturday', '(0~3)', 'Camera', 'Camera:notification', 'Saturday', '(0~3)', 'Blind', 'Blind:windowShade close']
['Saturday', '(6~9)', 'Television', 'Television:volumeDown', 'Saturday', '(6~9)', 'AirConditioner', 'AirConditioner:switch on', 'Saturday', '(6~9)', 'AirPurifier', 'AirPurifier:switch on', 'Saturday', '(6~9)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovem
['Saturday', '(9~12)', 'Light', 'Light:switch on', 'Saturday', '(9~12)', 'GarageDoor', 'GarageDoor:doorControl open', 'Saturday', '(9~12)', 'GarageDoor', 'GarageDoor:doorControl close', 'Saturday', '(9~12)', 'Refrigerator', 'Refrigerator:samsungce.powerCool ac
['Saturday', '(12~15)', 'Television', 'Television:setChannel', 'Saturday', '(12~15)', 'Fan', 'Fan:switch on', 'Saturday', '(12~15)', 'Light', 'Light:setLevel', 'Saturday', '(12~15)', 'Camera', 'Camera:notification']
['Saturday', '(15~18)', 'Television', 'Television:audioMute unmute', 'Saturday', '(15~18)', 'Television', 'Television:volumeDown', 'Saturday', '(15~18)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint', 'Saturday', '(15~18)', 'Heater', 'Heater:switch on'
['Saturday', '(18~21)', 'Television', 'Television:setChannel', 'Saturday', '(18~21)', 'Light', 'Light:switch off', 'Saturday', '(18~21)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Saturday', '(18~21)', 'SmartLock', 'SmartLock:lock lock'
['Saturday', '(21~24)', 'Television', 'Television:volumeDown', 'Saturday', '(21~24)', 'AirPurifier', 'AirPurifier:switch off', 'Saturday', '(21~24)', 'Blind', 'Blind:windowShade open', 'Saturday', '(21~24)', 'Refrigerator', 'Refrigerator:samsungce.powerFreeze 
['Sunday', '(0~3)', 'Television', 'Television:audioMute unmute', 'Sunday', '(0~3)', 'Light', 'Light:switch on', 'Sunday', '(0~3)', 'Camera', 'Camera:notification', 'Sunday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning']
['Sunday', '(6~9)', 'Television', 'Television:setChannel', 'Sunday', '(6~9)', 'AirConditioner', 'AirConditioner:switch off', 'Sunday', '(6~9)', 'Fan', 'Fan:switch off', 'Sunday', '(6~9)', 'GarageDoor', 'GarageDoor:doorControl open']
['Sunday', '(9~12)', 'Television', 'Television:volumeDown', 'Sunday', '(9~12)', 'Light', 'Light:switch off', 'Sunday', '(9~12)', 'Refrigerator', 'Refrigerator:samsungce.powerCool activate', 'Sunday', '(9~12)', 'SmartLock', 'SmartLock:lock unlock']
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_6_0_SPPC_th=0.917_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `1.6 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 6
['Sunday', '(6~9)', 'Television', 'Television:switch on', 'Sunday', '(6~9)', 'Television', 'Television:setChannel', 'Sunday', '(6~9)', 'Television', 'Television:volumeDown', 'Sunday', '(6~9)', 'Light', 'Light:switch on', 'Sunday', '(6~9)', 'AirPurifier', 'AirP
['Sunday', '(9~12)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Sunday', '(9~12)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Sunday', '(9~12)', 'Camera', 'Camera:notification', 'Sunday', '(9~12)', 'Blind', 'Blind:
['Sunday', '(12~15)', 'AirConditioner', 'AirConditioner:switch on', 'Sunday', '(12~15)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint', 'Sunday', '(12~15)', 'Light', 'Light:setLevel', 'Sunday', '(12~15)', 'NetworkAudio', 'NetworkAudio:mediaPlayback pla
['Sunday', '(15~18)', 'SmartLock', 'SmartLock:lock lock', 'Sunday', '(15~18)', 'Camera', 'Camera:notification', 'Sunday', '(15~18)', 'Television', 'Television:setInputSource', 'Sunday', '(15~18)', 'Television', 'Television:setChannel']
['Sunday', '(18~21)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Sunday', '(18~21)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerTurboMode on', 'Sunday', '(18~21)', 'Light', 'Light:switch off', 'Sunday', '(18~21)', 'AirPurifier', 'AirPu
['Sunday', '(21~24)', 'Television', 'Television:switch on', 'Sunday', '(21~24)', 'Television', 'Television:audioMute unmute', 'Sunday', '(21~24)', 'NetworkAudio', 'NetworkAudio:setVolume', 'Sunday', '(21~24)', 'Blind', 'Blind:windowShade close', 'Sunday', '(21
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_6_1_SPPC_th=0.917_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `2.1 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 8
['Sunday', '(6~9)', 'Television', 'Television:switch on', 'Sunday', '(6~9)', 'Television', 'Television:audioMute unmute', 'Sunday', '(6~9)', 'Light', 'Light:switch on', 'Sunday', '(6~9)', 'AirConditioner', 'AirConditioner:switch on', 'Sunday', '(6~9)', 'AirPur
['Sunday', '(9~12)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Sunday', '(9~12)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Sunday', '(9~12)', 'Camera', 'Camera:notification', 'Sunday', '(9~12)', 'Blind', 'Blind:
['Sunday', '(12~15)', 'Television', 'Television:setChannel', 'Sunday', '(12~15)', 'Television', 'Television:volumeDown', 'Sunday', '(12~15)', 'NetworkAudio', 'NetworkAudio:mediaPlayback play', 'Sunday', '(12~15)', 'NetworkAudio', 'NetworkAudio:audioVolume setV
['Sunday', '(15~18)', 'GarageDoor', 'GarageDoor:doorControl open', 'Sunday', '(15~18)', 'GarageDoor', 'GarageDoor:doorControl close', 'Sunday', '(15~18)', 'SmartLock', 'SmartLock:lock lock', 'Sunday', '(15~18)', 'SmartLock', 'SmartLock:lock unlock']
['Sunday', '(18~21)', 'Refrigerator', 'Refrigerator:samsungce.powerCool activate', 'Sunday', '(18~21)', 'Refrigerator', 'Refrigerator:samsungce.powerFreeze activate', 'Sunday', '(18~21)', 'Heater', 'Heater:switch on', 'Sunday', '(18~21)', 'Heater', 'Heater:set
['Sunday', '(21~24)', 'Television', 'Television:setChannel', 'Sunday', '(21~24)', 'Television', 'Television:volumeDown', 'Sunday', '(21~24)', 'Light', 'Light:switch off', 'Sunday', '(21~24)', 'AirConditioner', 'AirConditioner:switch off']
['Monday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Monday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Monday', '(0~3)', 'Camera', 'Camera:imageCapture take', 'Monday', '(0~3)', 'Blind', 'Blind
['Monday', '(6~9)', 'Television', 'Television:switch on', 'Monday', '(6~9)', 'Television', 'Television:audioMute unmute', 'Monday', '(6~9)', 'Light', 'Light:switch on', 'Monday', '(6~9)', 'AirConditioner', 'AirConditioner:switch on', 'Monday', '(6~9)', 'AirPur
```

### `/home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_6_SPPC_th=0.915_gpt-4o_seq.pkl`

- Type: `.pkl`
- Size: `1.7 KB`
- Useful for mapping: `unknown`
- Notes: SmartGen offline synthetic numeric sequence candidate

```text
top-level: list
len: 6
['Sunday', '(6~9)', 'Television', 'Television:switch on', 'Sunday', '(6~9)', 'Television', 'Television:audioMute unmute', 'Sunday', '(6~9)', 'Television', 'Television:setChannel', 'Sunday', '(6~9)', 'Light', 'Light:switch on', 'Sunday', '(6~9)', 'AirConditione
['Sunday', '(9~12)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Sunday', '(9~12)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Sunday', '(9~12)', 'Camera', 'Camera:notification', 'Sunday', '(9~12)', 'Blind', 'Blind:
['Sunday', '(12~15)', 'NetworkAudio', 'NetworkAudio:mediaPlayback play', 'Sunday', '(12~15)', 'NetworkAudio', 'NetworkAudio:audioVolume setVolume', 'Sunday', '(12~15)', 'Light', 'Light:setLevel', 'Sunday', '(12~15)', 'Refrigerator', 'Refrigerator:samsungce.pow
['Sunday', '(15~18)', 'Television', 'Television:setChannel', 'Sunday', '(15~18)', 'Television', 'Television:volumeDown', 'Sunday', '(15~18)', 'Projector', 'Projector:custom.soundmode setSoundMode', 'Sunday', '(15~18)', 'AirPurifier', 'AirPurifier:setAirPurifie
['Sunday', '(18~21)', 'SmartLock', 'SmartLock:lock lock', 'Sunday', '(18~21)', 'GarageDoor', 'GarageDoor:doorControl close', 'Sunday', '(18~21)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Sunday', '(18~21)', 'RobotCleaner', 'RobotCleane
['Sunday', '(21~24)', 'Television', 'Television:switch on', 'Sunday', '(21~24)', 'Television', 'Television:setChannel', 'Sunday', '(21~24)', 'Light', 'Light:switch off', 'Sunday', '(21~24)', 'AirConditioner', 'AirConditioner:switch off', 'Sunday', '(21~24)', '
```

## SmartGuard Data Findings

### Original FR Data

- `train`: `/home/heyang/projects/SmartGuard/data/fr_data/fr_trn_instance_10.pkl`
  - samples: 2233; valid reshape: 2233; shapes: ['(10, 4)']
  - column ranges as `(min, max, unique_count)`: [(0, 6, 7), (0, 7, 8), (0, 31, 21), (3, 216, 56)]
  - first sample as 10x4: [[2, 2, 18, 109], [3, 1, 29, 193], [3, 5, 29, 193], [3, 6, 29, 193], [4, 5, 29, 193], [5, 5, 29, 193], [6, 1, 29, 192], [6, 1, 18, 109], [0, 5, 29, 193], [1, 2, 18, 109]]
- `val`: `/home/heyang/projects/SmartGuard/data/fr_data/fr_vld_instance_10.pkl`
  - samples: 322; valid reshape: 322; shapes: ['(10, 4)']
  - column ranges as `(min, max, unique_count)`: [(0, 6, 7), (0, 7, 8), (0, 31, 19), (3, 216, 46)]
  - first sample as 10x4: [[2, 6, 29, 193], [2, 7, 29, 193], [3, 3, 29, 193], [4, 3, 29, 193], [5, 1, 29, 193], [6, 7, 29, 193], [1, 0, 29, 193], [3, 3, 29, 193], [4, 0, 29, 193], [4, 3, 29, 193]]
- `test`: `/home/heyang/projects/SmartGuard/data/fr_data/fr_test_instance_10.pkl`
  - samples: 641; valid reshape: 641; shapes: ['(10, 4)']
  - column ranges as `(min, max, unique_count)`: [(0, 6, 7), (0, 7, 8), (0, 31, 20), (3, 216, 52)]
  - first sample as 10x4: [[3, 2, 29, 193], [3, 7, 29, 205], [3, 7, 29, 193], [4, 1, 29, 193], [4, 6, 29, 193], [0, 5, 29, 193], [1, 7, 29, 193], [3, 7, 29, 193], [4, 7, 13, 72], [4, 7, 29, 193]]

Combined FR original rows:
- events: 31960
- column ranges as `(min, max, unique_count)`: [(0, 6, 7), (0, 7, 8), (0, 31, 21), (3, 216, 59)]
- device-id candidate sample from column 3: [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 16, 18, 21, 24, 26, 27, 28, 29, 31]
- control/action-id sample from column 4: [3, 5, 9, 10, 14, 15, 16, 18, 22, 27, 28, 34, 35, 41, 42, 52, 56, 57, 60, 63, 65, 66, 67, 68, 69, 72, 75, 76, 77, 78, 83, 84, 86, 91, 94, 96, 104, 106, 109, 115, 132, 150, 155, 156, 164, 169, 170, 172, 173, 176, 180, 192, 193, 196, 197, 198, 202, 205, 216]

Inference: SmartGuard FR raw samples are 10x4. The strongest evidence from SmartGen generated data and `action_transitions.json` indicates the columns are `[day, hour_or_time_slot, device_id, action_or_control_id]`. The existing CausalGenGuard prepared JSONL preserves column 4 as `control_id`, but currently treats column 3 as `duration`/`persistence_or_device` rather than semantic `device_id`.

### Prepared CausalGenGuard FR JSONL

- `/home/heyang/projects/CausalGenGuard/outputs/processed/fr_sequences_smoke.jsonl`
  - sequences: 100; events: 1000
  - control stats: {'min': 9.0, 'max': 216.0, 'unique': 29, 'sample': [9, 10, 15, 27, 28, 34, 35, 41, 56, 57, 60, 67, 69, 72, 75, 84, 91, 96, 109, 155, 156, 176, 180, 192, 193, 196, 202, 205, 216]}
  - device_id null/non-null: 1000/0
  - duration stats: {'min': 0.0, 'max': 31.0, 'unique': 16, 'sample': [0, 1, 2, 3, 4, 9, 10, 12, 13, 16, 18, 24, 27, 28, 29, 31]}
  - day stats: {'min': 0.0, 'max': 6.0, 'unique': 7, 'sample': [0, 1, 2, 3, 4, 5, 6]}; hour stats: {'min': 0.0, 'max': 7.0, 'unique': 8, 'sample': [0, 1, 2, 3, 4, 5, 6, 7]}
  - raw field keys: ['control', 'day', 'hour', 'persistence_or_device', 'position', 'source_format']
- `/home/heyang/projects/CausalGenGuard/outputs/processed/fr_sequences_acceptance.jsonl`
  - sequences: 100; events: 1000
  - control stats: {'min': 9.0, 'max': 216.0, 'unique': 29, 'sample': [9, 10, 15, 27, 28, 34, 35, 41, 56, 57, 60, 67, 69, 72, 75, 84, 91, 96, 109, 155, 156, 176, 180, 192, 193, 196, 202, 205, 216]}
  - device_id null/non-null: 1000/0
  - duration stats: {'min': 0.0, 'max': 31.0, 'unique': 16, 'sample': [0, 1, 2, 3, 4, 9, 10, 12, 13, 16, 18, 24, 27, 28, 29, 31]}
  - day stats: {'min': 0.0, 'max': 6.0, 'unique': 7, 'sample': [0, 1, 2, 3, 4, 5, 6]}; hour stats: {'min': 0.0, 'max': 7.0, 'unique': 8, 'sample': [0, 1, 2, 3, 4, 5, 6, 7]}
  - raw field keys: ['control', 'day', 'hour', 'persistence_or_device', 'position', 'source_format']

Findings:
- Timestamp: no absolute timestamp found in SmartGuard FR raw samples; only day and hour/time-slot style values are present.
- Duration: no independent duration column was found. Column 3 is more likely numeric `device_id`, not duration.
- Device id: yes, likely numeric in column 3, but not named.
- Control id: yes, numeric in column 4. A semantic id-to-name mapping was not found.
- Reverse mapping from raw data alone: not reliable. Numeric ranges and transitions are insufficient to assign names such as camera/light/lock/window without a dictionary or curated mapping.

## SmartGen Data Findings

### Textual Catalogs

- `/home/heyang/projects/SmartGen/SmartGen/fr_keys_best.txt`
  - device entries: 31; action entries: 198
  - devices sample: ['AirConditioner', 'AirPurifier', 'Blind', 'Camera', 'ClothingCareMachine', 'Computer', 'ContactSensor', 'CurbPowerMeter', 'Dryer', 'Elevator', 'Fan', 'GarageDoor', 'Light', 'Microwave', 'MotionSensor', 'NetworkAudio', 'Other', 'Oven', 'PresenceSensor', 'Projector']
  - actions sample: [('AirConditioner', ['fanspeedDown', 'fanspeedUp', 'notification', 'setAcOptionalMode', 'setAirConditionerMode']), ('AirPurifier', ['notification', 'setAirPurifierMode', 'setFanMode', 'setFanSpeed', 'switch off']), ('Blind', ['refresh refresh', 'statelessCurtainPowerButton setButton', 'switch off', 'switch on', 'switchLevel setLevel']), ('Camera', ['alarm off', 'cameraPreset execute', 'imageCapture take', 'notification', 'switch off']), ('ClothingCareMachine', ['dryerOperatingState setMachineState stop', 'notification']), ('Computer', ['refresh refresh']), ('ContactSensor', ['doorControl close', 'lock lock', 'lock unlock', 'switch off', 'switch on']), ('CurbPowerMeter', ['energyMeter resetEnergyMeter'])]
- `/home/heyang/projects/SmartGen/SmartGen/sp_keys_best.txt`
  - device entries: 33; action entries: 211
  - devices sample: ['AirConditioner', 'AirPurifier', 'Blind', 'Camera', 'ClothingCareMachine', 'Computer', 'ContactSensor', 'CurbPowerMeter', 'Dishwasher', 'Dryer', 'Elevator', 'Fan', 'GarageDoor', 'Light', 'Microwave', 'MotionSensor', 'NetworkAudio', 'Other', 'Oven', 'PresenceSensor']
  - actions sample: [('AirConditioner', ['fanspeedDown', 'fanspeedUp', 'notification', 'setAcOptionalMode', 'setAirConditionerMode']), ('AirPurifier', ['notification', 'setAirPurifierMode', 'setFanMode', 'switch off', 'switch on']), ('Blind', ['refresh refresh', 'statelessCurtainPowerButton setButton', 'switch off', 'switch on', 'switchLevel setLevel']), ('Camera', ['alarm off', 'cameraPreset execute', 'imageCapture take', 'notification', 'switch off']), ('ClothingCareMachine', ['dryerOperatingState setMachineState run', 'notification', 'setting']), ('Computer', ['notification', 'switch off']), ('ContactSensor', ['doorControl close', 'lock lock', 'lock unlock', 'switch off', 'switch on']), ('CurbPowerMeter', ['energyMeter resetEnergyMeter'])]
- `/home/heyang/projects/SmartGen/SmartGen/us_keys_best.txt`
  - device entries: 39; action entries: 244
  - devices sample: ['AirConditioner', 'AirPurifier', 'Blind', 'Camera', 'ClothingCareMachine', 'Computer', 'ContactSensor', 'Dishwasher', 'Dryer', 'Elevator', 'Fan', 'GarageDoor', 'Humidifier', 'Irrigation', 'Light', 'LightSensor', 'Microwave', 'MotionSensor', 'MultiFunctionalSensor', 'NetworkAudio']
  - actions sample: [('AirConditioner', ['fanspeedDown', 'fanspeedUp', 'notification', 'setAcOptionalMode', 'setAirConditionerMode']), ('AirPurifier', ['notification', 'refresh', 'setAirPurifierMode', 'setCleaningOff', 'setFanMode']), ('Blind', ['refresh refresh', 'statelessCurtainPowerButton setButton', 'switch off', 'switch on', 'switchLevel setLevel']), ('Camera', ['cameraPreset execute', 'notification', 'switch off', 'switch on', 'videoCapture capture']), ('ClothingCareMachine', ['dryerOperatingState setMachineState run']), ('Computer', ['mute', 'notification', 'switch off', 'switch on', 'unmute']), ('ContactSensor', ['lock unlock', 'momentary push', 'switch off', 'switch on']), ('Dishwasher', ['notification', 'setRun', 'setStop', 'start'])]

The key files are useful for a canonical textual action vocabulary. They do not include numeric ids, so they cannot directly align SmartGuard numeric controls.

### Synthetic / Offline Candidate Files

| path | type | size | control_form | device_name | action_state | contexts | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 1.9 KB | numeric | no/unknown | no/unknown | fr,sp,multiple | [0, 0, 29, 205, 0, 0, 29, 196, 0, 0, 0, 6, 0, 0, 13, 75] |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 3.8 KB | numeric | no/unknown | no/unknown | fr,sp,multiple | [0, 0, 29, 205, 0, 0, 13, 78, 0, 0, 1, 20, 0, 0, 24, 156] |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_0_SPPC_th=0.915_gpt-4o.pkl | .pkl | 4.7 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the final user behavior sequence set, we will follow the steps outlined in the task:\n\n---\n\n### **Step 1: Select Possible New Device States**\nB |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_0_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 1.7 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Monday', '(0~3)', 'Television', 'Television:volumeDown', 'Monday', '(0~3)', 'Television', 'Television:setChannel', 'Monday', '(0~3)', 'AirConditioner', 'AirCo |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_0_SPPC_th=0.917_gpt-4o.pkl | .pkl | 5.3 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequence set, I will follow the steps outlined in the task:\n\n---\n\n### **Step 1: Select Possible New Device States**\nBase |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_0_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.6 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Monday', '(0~3)', 'Television', 'Television:volumeDown', 'Monday', '(0~3)', 'Light', 'Light:switch on', 'Monday', '(0~3)', 'AirPurifier', 'AirPurifier:switch  |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_1_0_SPPC_th=0.917_gpt-4o.pkl | .pkl | 5.4 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequences based on the provided information, I will follow the steps outlined in the task:\n\n---\n\n### Step 1: Select Possi |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_1_0_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.7 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Tuesday', '(3~6)', 'Television', 'Television:volumeDown', 'Tuesday', '(3~6)', 'Television', 'Television:setChannel', 'Tuesday', '(3~6)', 'AirConditioner', 'Ai |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_1_1_SPPC_th=0.917_gpt-4o.pkl | .pkl | 4.5 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequences based on the given information, I will follow the steps outlined in the task:\n\n---\n\n### Step 1: Select Possible |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_1_1_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.2 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Tuesday', '(18~21)', 'GarageDoor', 'GarageDoor:doorControl open', 'Tuesday', '(18~21)', 'GarageDoor', 'GarageDoor:doorControl close', 'Tuesday', '(18~21)', 'L |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_1_SPPC_th=0.915_gpt-4o.pkl | .pkl | 5.0 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequences based on the given information, I will follow the steps outlined in the task:\n\n---\n\n### **Step 1: Select Possib |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_1_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 1.9 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Tuesday', '(6~9)', 'Light', 'Light:switch on', 'Tuesday', '(6~9)', 'AirConditioner', 'AirConditioner:setFanMode', 'Tuesday', '(6~9)', 'Television', 'Televisio |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_2_0_SPPC_th=0.917_gpt-4o.pkl | .pkl | 4.4 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequence set, we will follow the steps outlined in the task:\n\n---\n\n### Step 1: Select Possible New Device States\nBased o |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_2_0_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.0 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Wednesday', '(3~6)', 'Other', 'None:location', 'Wednesday', '(3~6)', 'AirPurifier', 'AirPurifier:setAirPurifierMode', 'Wednesday', '(3~6)', 'Fan', 'Fan:switch |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_2_1_SPPC_th=0.917_gpt-4o.pkl | .pkl | 5.6 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequence set, I will follow the steps outlined in the task. Let's proceed step by step:\n\n---\n\n### **Step 1: Select Possib |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_2_1_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.4 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Wednesday', '(0~3)', 'AirPurifier', 'AirPurifier:setAirPurifierMode', 'Wednesday', '(0~3)', 'AirConditioner', 'AirConditioner:switch on', 'Wednesday', '(0~3)' |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_2_SPPC_th=0.915_gpt-4o.pkl | .pkl | 4.5 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequence set, we will follow the steps outlined in the task:\n\n---\n\n### **Step 1: Select Possible New Device States**\nCon |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_2_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 1.7 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Wednesday', '(6~9)', 'AirConditioner', 'AirConditioner:switch on', 'Wednesday', '(6~9)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint', 'Wednesday', ' |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_3_0_SPPC_th=0.917_gpt-4o.pkl | .pkl | 4.8 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequence set, we will follow the steps outlined in the task:\n\n---\n\n### **Step 1: Select Possible New Device States**\nBas |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_3_0_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 1.9 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Thursday', '(3~6)', 'Other', 'None:location', 'Thursday', '(3~6)', 'AirConditioner', 'AirConditioner:setCoolingSetpoint', 'Thursday', '(3~6)', 'Light', 'Light |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_3_1_SPPC_th=0.917_gpt-4o.pkl | .pkl | 5.6 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequence set, I will follow the steps outlined in the task. Let's proceed step by step:\n\n---\n\n### **Step 1: Select Possib |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_3_1_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.3 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Thursday', '(15~18)', 'NetworkAudio', 'NetworkAudio:audioMute mute', 'Thursday', '(15~18)', 'Light', 'Light:setLevel', 'Thursday', '(15~18)', 'AirConditioner' |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_3_SPPC_th=0.915_gpt-4o.pkl | .pkl | 5.0 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequences, we will follow the steps outlined in the task:\n\n---\n\n### **Step 1: Select Possible New Device States**\nBased  |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_3_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 1.5 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Thursday', '(6~9)', 'Light', 'Light:switch on', 'Thursday', '(6~9)', 'AirConditioner', 'AirConditioner:switch on', 'Thursday', '(6~9)', 'AirPurifier', 'AirPur |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_4_0_SPPC_th=0.917_gpt-4o.pkl | .pkl | 4.8 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequences, I will follow the steps outlined in the task:\n\n---\n\n### **Step 1: Select Possible New Device States**\nBased o |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_4_0_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.1 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Friday', '(0~3)', 'AirConditioner', 'AirConditioner:switch on', 'Friday', '(0~3)', 'AirConditioner', 'AirConditioner:temperatureUp', 'Friday', '(0~3)', 'Telev |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_4_1_SPPC_th=0.917_gpt-4o.pkl | .pkl | 4.7 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequence set, we will follow the steps outlined in the task:\n\n---\n\n### **Step 1: Select Possible New Device States**\nBas |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_4_1_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 1.3 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Friday', '(18~21)', 'Television', 'Television:setChannel', 'Friday', '(18~21)', 'Television', 'Television:volumeDown', 'Friday', '(18~21)', 'Light', 'Light:se |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_4_SPPC_th=0.915_gpt-4o.pkl | .pkl | 5.5 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequences, I will follow the steps outlined in the task. Let's proceed step by step:\n\n---\n\n### **Step 1: Select Possible  |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_4_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 2.3 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Friday', '(3~6)', 'Television', 'Television:setChannel', 'Friday', '(3~6)', 'Light', 'Light:switch on', 'Friday', '(3~6)', 'Blind', 'Blind:windowShade open',  |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_5_SPPC_th=0.915_gpt-4o.pkl | .pkl | 5.8 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequences, I will follow the steps outlined in the task:\n\n---\n\n### **Step 1: Select Possible New Device States**\nBased o |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_5_SPPC_th=0.915_gpt-4o_seq.pkl | .pkl | 1.9 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Saturday', '(0~3)', 'Television', 'Television:volumeDown', 'Saturday', '(0~3)', 'AirConditioner', 'AirConditioner:switch on', 'Saturday', '(0~3)', 'AirPurifie |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_5_SPPC_th=0.917_gpt-4o.pkl | .pkl | 5.3 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequences, we will follow the steps outlined in the task:\n\n---\n\n### **Step 1: Select Possible New Device States**\nWe wil |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_5_SPPC_th=0.917_gpt-4o_seq.pkl | .pkl | 2.3 KB | textual/mixed | yes | yes | fr,sp,multiple | ['Saturday', '(0~3)', 'Television', 'Television:volumeDown', 'Saturday', '(0~3)', 'Light', 'Light:switch on', 'Saturday', '(0~3)', 'Camera', 'Camera:notificatio |
| /home/heyang/projects/SmartGen/SmartGen/IoT_data/fr/multiple/fr_multiple_generation_day_6_0_SPPC_th=0.917_gpt-4o.pkl | .pkl | 5.7 KB | textual/mixed | yes | yes | fr,sp,multiple | "To generate the new user behavior sequence set, I will follow the steps outlined in the task:\n\n---\n\n### **Step 1: Select Possible New Device States**\nBase |

Summary: SmartGen has both textual hint files (`action_transitions.json`) and numeric generated sequence files (`*generation*seq.pkl`, `trn.pkl`). The numeric sequence files use flat 4-tuples compatible with CausalGenGuard `BehaviorSequence` conversion, but the semantic control name is absent from the sample itself. Context is mainly encoded by paths such as `fr/sp/us`, `spring/night/multiple/daytime/single`, and transition/generation filename conventions.

### SmartGen Code Search Notes

Searched SmartGen Python files for `keys_best`, `action_transitions`, `dictionary`, `device`, `action`, `mapping`, `id`, and `vocab`. Representative hits:

```text
/home/heyang/projects/SmartGen/SmartGen/SAS_main.py:9: vocab_dic = {"an": 141, "fr": 222, "us": 268, "sp": 234}
/home/heyang/projects/SmartGen/SmartGen/SAS_main.py:29: raise ValueError('Not a valid boolean string')
/home/heyang/projects/SmartGen/SmartGen/SAS_main.py:42: parser.add_argument('--hidden_units', default=50, type=int)
/home/heyang/projects/SmartGen/SmartGen/SAS_main.py:48: parser.add_argument('--device', default='cuda', type=str)
/home/heyang/projects/SmartGen/SmartGen/SAS_main.py:61: parser.add_argument('--hidden_units', default=50, type=int)
/home/heyang/projects/SmartGen/SmartGen/SAS_main.py:67: parser.add_argument('--device', default='cuda', type=str)
/home/heyang/projects/SmartGen/SmartGen/SAS_main.py:71: default=f'{nation}_{new_env}_generation_{method}_th={threshold}_{model}_seq_default/SASRec.epoch=500.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth',
/home/heyang/projects/SmartGen/SmartGen/SAS_main.py:86: [user_train, user_valid, user_test, usernum, _] = dataset
/home/heyang/projects/SmartGen/SmartGen/SAS_main.py:87: itemnum = vocab_dic[nation]
/home/heyang/projects/SmartGen/SmartGen/SAS_main.py:99: model = SASRec(usernum, itemnum, args).to(args.device)
/home/heyang/projects/SmartGen/SmartGen/SAS_main.py:112: epoch_start_idx = 1
/home/heyang/projects/SmartGen/SmartGen/SAS_main.py:115: model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
/home/heyang/projects/SmartGen/SmartGen/baseline1.py:15: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
/home/heyang/projects/SmartGen/SmartGen/baseline1.py:16: vocab_dic = {"an": 141, "fr": 223, "us": 269, "sp": 235}
/home/heyang/projects/SmartGen/SmartGen/baseline1.py:34: def pad(vocab_size, sequences):
/home/heyang/projects/SmartGen/SmartGen/baseline1.py:37: sequences[i].extend([vocab_size - 1] * (40 - len(sequences[i])))
/home/heyang/projects/SmartGen/SmartGen/baseline1.py:66: help='Mask strategy:random/top_k_loss/loss_guided')
/home/heyang/projects/SmartGen/SmartGen/baseline1.py:70: def make_data(new_env, vocab_size, data_file='reduced_flattened_useful_us_trn_instance_10.pkl', batch_size=32):
/home/heyang/projects/SmartGen/SmartGen/baseline1.py:73: data = pad(vocab_size, sequences)
/home/heyang/projects/SmartGen/SmartGen/baseline1.py:76: dataset = TimeSeriesDataset2(vocab_size, data)
/home/heyang/projects/SmartGen/SmartGen/baseline1.py:78: dataset = TimeSeriesDataset3(vocab_size, data)
/home/heyang/projects/SmartGen/SmartGen/baseline1.py:80: dataset = TimeSeriesDataset4(vocab_size, data)
/home/heyang/projects/SmartGen/SmartGen/baseline1.py:88: def train(new_env, vocab_size, epochs, train_file, model_name, seq_len):
/home/heyang/projects/SmartGen/SmartGen/baseline1.py:89: model = TransformerAutoencoder(vocab_size=vocab_size, d_model=512, nhead=8, num_encoder_layers=2,
/home/heyang/projects/SmartGen/SmartGen/baseline2.py:13: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
/home/heyang/projects/SmartGen/SmartGen/baseline2.py:31: def pad(vocab_size, sequences):
/home/heyang/projects/SmartGen/SmartGen/baseline2.py:34: sequence.extend([vocab_size - 1] * (40 - len(sequence)))
/home/heyang/projects/SmartGen/SmartGen/baseline2.py:38: def make_data(vocab_size, data_file='reduced_flattened_useful_us_trn_instance_10.pkl', batch_size=512):
/home/heyang/projects/SmartGen/SmartGen/baseline2.py:41: data = pad(vocab_size, sequence)
/home/heyang/projects/SmartGen/SmartGen/baseline2.py:45: dataset = TimeSeriesDataset1(vocab_size, data)
/home/heyang/projects/SmartGen/SmartGen/baseline2.py:52: def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=2):
/home/heyang/projects/SmartGen/SmartGen/baseline2.py:55: self.embedding = nn.Embedding(vocab_size, d_model)
/home/heyang/projects/SmartGen/SmartGen/baseline2.py:63: self.output_layer = nn.Linear(d_model, vocab_size)
/home/heyang/projects/SmartGen/SmartGen/baseline2.py:88: def Train(dataset, ori_env, vocab_size):
/home/heyang/projects/SmartGen/SmartGen/baseline2.py:94: model = TransformerAutoencoder(vocab_size, d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
/home/heyang/projects/SmartGen/SmartGen/baseline2.py:95: model = model.to(device)
/home/heyang/projects/SmartGen/SmartGen/dictionary.py:6: 'Friday': 4,
/home/heyang/projects/SmartGen/SmartGen/dictionary.py:22: fr_devices_dict = {
/home/heyang/projects/SmartGen/SmartGen/dictionary.py:58: fr_actions = {
/home/heyang/projects/SmartGen/SmartGen/dictionary.py:97: 'Camera:videoCapture capture': 38,
/home/heyang/projects/SmartGen/SmartGen/dictionary.py:166: 'Other:notification deviceNotification': 107,
/home/heyang/projects/SmartGen/SmartGen/dictionary.py:283: an_devices_dict = {
/home/heyang/projects/SmartGen/SmartGen/dictionary.py:286: 'dehumidifier': 2,
/home/heyang/projects/SmartGen/SmartGen/dictionary.py:287: 'deerma_humidifier': 3,
/home/heyang/projects/SmartGen/SmartGen/dictionary.py:291: 'mijia_humidifier': 7,
/home/heyang/projects/SmartGen/SmartGen/dictionary.py:323: an_actions = {
/home/heyang/projects/SmartGen/SmartGen/dictionary.py:330: 'ac_dehumidification_mode': 6,
/home/heyang/projects/SmartGen/SmartGen/dictionary.py:335: 'dehumidifier_on': 11,
/home/heyang/projects/SmartGen/SmartGen/find_categories.py:16: start_idx = i * 30
/home/heyang/projects/SmartGen/SmartGen/find_categories.py:17: end_idx = min((i + 1) * 30, len(data))
/home/heyang/projects/SmartGen/SmartGen/find_categories.py:18: subgroup = data[start_idx:end_idx]
/home/heyang/projects/SmartGen/SmartGen/find_categories.py:41: start_idx = i * 20
/home/heyang/projects/SmartGen/SmartGen/find_categories.py:42: end_idx = min((i + 1) * 20, len(data))
/home/heyang/projects/SmartGen/SmartGen/find_categories.py:43: subgroup = data[start_idx:end_idx]
/home/heyang/projects/SmartGen/SmartGen/find_categories.py:66: start_idx = i * 29
/home/heyang/projects/SmartGen/SmartGen/find_categories.py:67: end_idx = min((i + 1) * 30, len(data))
/home/heyang/projects/SmartGen/SmartGen/find_categories.py:68: subgroup = data[start_idx:end_idx]
/home/heyang/projects/SmartGen/SmartGen/main.py:7: from dictionary import dayofweek_dict, hour_dict, fr_devices_dict, fr_actions, sp_devices_dict, sp_actions, us_devices_dict, us_actions
/home/heyang/projects/SmartGen/SmartGen/main.py:15: from text_translation_matrix import ATM
/home/heyang/projects/SmartGen/SmartGen/main.py:21: vocab_dic = {"an": 141, "fr": 223, "us": 269, "sp": 235}
/home/heyang/projects/SmartGen/SmartGen/main.py:22: device_dic = {"us": us_devices_dict, "fr": fr_devices_dict, "sp": sp_devices_dict}
/home/heyang/projects/SmartGen/SmartGen/main.py:23: act_dic = {"us": us_actions, "fr": fr_actions, "sp": sp_actions}
/home/heyang/projects/SmartGen/SmartGen/main.py:91: Train(args.dataset, args.ori_env, vocab_dic[args.dataset])
/home/heyang/projects/SmartGen/SmartGen/main.py:92: SPPC_select(args.dataset, args.ori_env, vocab_dic[args.dataset], args.threshold)
/home/heyang/projects/SmartGen/SmartGen/main.py:97: device_dict = device_dic[args.dataset]
/home/heyang/projects/SmartGen/SmartGen/main.py:98: actions = act_dic[args.dataset]
/home/heyang/projects/SmartGen/SmartGen/main.py:99: dictionaries = [dayofweek_dict, hour_dict, device_dict, actions]
/home/heyang/projects/SmartGen/SmartGen/main.py:101: ATM(args.dataset, args.ori_env, actions)
/home/heyang/projects/SmartGen/SmartGen/main.py:111: with open(f'{args.dataset}_keys_best.txt', 'r') as file:
/home/heyang/projects/SmartGen/SmartGen/model.py:6: def __init__(self, hidden_units, dropout_rate):
/home/heyang/projects/SmartGen/SmartGen/model.py:9: self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
/home/heyang/projects/SmartGen/SmartGen/model.py:12: self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
/home/heyang/projects/SmartGen/SmartGen/model.py:29: self.dev = args.device
/home/heyang/projects/SmartGen/SmartGen/model.py:31: self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
/home/heyang/projects/SmartGen/SmartGen/model.py:32: self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
/home/heyang/projects/SmartGen/SmartGen/model.py:40: self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
/home/heyang/projects/SmartGen/SmartGen/model.py:43: new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
/home/heyang/projects/SmartGen/SmartGen/model.py:46: new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
/home/heyang/projects/SmartGen/SmartGen/model.py:51: new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
/home/heyang/projects/SmartGen/SmartGen/model.py:54: new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
/home/heyang/projects/SmartGen/SmartGen/model.py:67: attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
/home/heyang/projects/SmartGen/SmartGen/models1.py:8: def __init__(self, vocab_size, input_size, hidden_size1, hidden_size2):
/home/heyang/projects/SmartGen/SmartGen/models1.py:11: self.embedding = nn.Embedding(vocab_size, input_size)
/home/heyang/projects/SmartGen/SmartGen/models1.py:13: self.encoder_1 = nn.Linear(input_size, hidden_size1)
/home/heyang/projects/SmartGen/SmartGen/models1.py:14: self.encoder_2 = nn.Linear(hidden_size1, hidden_size2)
/home/heyang/projects/SmartGen/SmartGen/models1.py:16: self.decoder_1 = nn.Linear(hidden_size2, hidden_size1)
/home/heyang/projects/SmartGen/SmartGen/models1.py:17: self.decoder_2 = nn.Linear(hidden_size1, input_size)
/home/heyang/projects/SmartGen/SmartGen/models1.py:19: self.decoder_output = nn.Linear(input_size, vocab_size)
/home/heyang/projects/SmartGen/SmartGen/models1.py:33: def __init__(self, vocab_size, input_size, hidden_size1, hidden_size2, dropout_value):
/home/heyang/projects/SmartGen/SmartGen/models1.py:36: self.embedding = nn.Embedding(vocab_size, input_size)
/home/heyang/projects/SmartGen/SmartGen/models1.py:38: self.encoder_rnn1 = nn.GRU(input_size, hidden_size1, dropout=dropout_value, batch_first=True)
/home/heyang/projects/SmartGen/SmartGen/models1.py:39: self.encoder_rnn2 = nn.GRU(hidden_size1, hidden_size2, dropout=dropout_value, batch_first=True)
/home/heyang/projects/SmartGen/SmartGen/models1.py:41: self.decoder_rnn1 = nn.GRU(hidden_size2, hidden_size1, dropout=dropout_value, batch_first=True)
/home/heyang/projects/SmartGen/SmartGen/security_check.py:14: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
/home/heyang/projects/SmartGen/SmartGen/security_check.py:15: vocab_dic = {"an": 141, "fr": 223, "us": 269, "sp": 235}
/home/heyang/projects/SmartGen/SmartGen/security_check.py:33: def pad(vocab_size, sequences):
/home/heyang/projects/SmartGen/SmartGen/security_check.py:36: sequences[i].extend([vocab_size - 1] * (40 - len(sequences[i])))
/home/heyang/projects/SmartGen/SmartGen/security_check.py:112: help='Mask strategy:random/top_k_loss/loss_guided')
/home/heyang/projects/SmartGen/SmartGen/security_check.py:116: def make_data(new_env, vocab_size, data_file='reduced_flattened_useful_us_trn_instance_10.pkl', batch_size=32):
/home/heyang/projects/SmartGen/SmartGen/security_check.py:120: data = pad(vocab_size, sequences)
/home/heyang/projects/SmartGen/SmartGen/security_check.py:123: dataset = TimeSeriesDataset1(vocab_size, data)
/home/heyang/projects/SmartGen/SmartGen/security_check.py:129: def make_data_check(new_env, vocab_size, trn_file, add_file, batch_size=32):
/home/heyang/projects/SmartGen/SmartGen/security_check.py:136: data = pad(vocab_size, sequences)
/home/heyang/projects/SmartGen/SmartGen/security_check.py:139: dataset = TimeSeriesDataset1(vocab_size, data)
/home/heyang/projects/SmartGen/SmartGen/security_check.py:145: def train(new_env, vocab_size, epochs, train_file, model_name, seq_len):
/home/heyang/projects/SmartGen/SmartGen/split.py:5: from dictionary import fr_actions_off, us_actions_off, sp_actions_off
/home/heyang/projects/SmartGen/SmartGen/split.py:9: device_index = 2
/home/heyang/projects/SmartGen/SmartGen/split.py:10: action_index = 3
/home/heyang/projects/SmartGen/SmartGen/split.py:12: action_dic = {
/home/heyang/projects/SmartGen/SmartGen/split.py:13: "fr": fr_actions_off.values(),
/home/heyang/projects/SmartGen/SmartGen/split.py:14: "us": us_actions_off.values(),
/home/heyang/projects/SmartGen/SmartGen/split.py:15: "sp": sp_actions_off.values()
/home/heyang/projects/SmartGen/SmartGen/split.py:49: def semantic_judge(action, data_name):
/home/heyang/projects/SmartGen/SmartGen/split.py:50: return action[-1] not in np.array(action_dic[data_name])
/home/heyang/projects/SmartGen/SmartGen/split.py:77: devices = []
/home/heyang/projects/SmartGen/SmartGen/split.py:78: actions = []
/home/heyang/projects/SmartGen/SmartGen/split.py:83: devices.append(row[device_index])
/home/heyang/projects/SmartGen/SmartGen/sppc.py:13: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
/home/heyang/projects/SmartGen/SmartGen/sppc.py:16: def pad(vocab_size, sequences):
/home/heyang/projects/SmartGen/SmartGen/sppc.py:19: sequence.extend([vocab_size - 1] * (40 - len(sequence)))
```

No sampled code hit exposed a clean numeric id to textual `Device:action` dictionary. The code appears to operate on textual transition hints and numeric tuple data in separate places.

## Mapping Feasibility Decision

**Decision: B. 只能建立部分映射，需要人工补充.**

Rationale:
- A high-confidence automatic mapping would require explicit files such as `device_id -> device_name` and `action_id/control_id -> action_name`, or deterministic source code that constructs those ids in an inspectable order.
- The audit found strong partial evidence: SmartGuard and SmartGen share a compatible 4-column numeric event layout, and SmartGen exposes textual `Device:action` catalogs and textual transition hints.
- The missing piece is the numeric-to-text dictionary. Without it, named SmartGuard-style attacks can target textual concepts only after manual or recovered id alignment.

## Recommended Next Step

1. Implement a read-only parser for SmartGen `*_keys_best.txt` to build a canonical textual `Device:action` catalog.
2. Continue looking specifically for the original preprocessing dictionary that maps numeric device/action ids. The most valuable missing files would look like `device_dict`, `action_dict`, `capability_dict`, `id_to_action`, `id_to_device`, `vocab`, or SmartSense dictionary exports.
3. Inspect or request the SmartSense raw dictionary/routine files if available. SmartSense is the most likely source for country-specific device/action semantics.
4. After the numeric dictionary is found or curated, update CausalGenGuard adapters so SmartGuard column 3 becomes `device_id`, column 4 remains `control_id`, and `duration` is derived later only when available.
5. Only after approval, add a mapper under CausalGenGuard that aligns SmartGuard numeric ids and SmartGen textual controls. No mapping JSON was generated during this audit.

## Current Impact On Experiments

- SmartGuard named attacks: currently blocked for semantic targeting; numeric fallback remains usable.
- SmartGen/Causal-TOF kept=0: likely caused by vocabulary mismatch between SmartGen numeric controls and the CausalGenGuard vocab built from limited SmartGuard samples, plus missing canonical semantic alignment.
- Causal branch trend: should not be judged as ineffective until the semantic/device-action mapping and consistent vocabulary are repaired.
