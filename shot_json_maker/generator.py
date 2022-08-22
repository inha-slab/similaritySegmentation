import json

shot_count = 1
frame_list = []

with open("MiSang_Frame.json", 'r') as outfile:
    frame_list = json.load(outfile)
    shot_count = frame_list[-1]["shot"] + 1

while(True):
    frame = input()
    if(frame == 'q'):
        break
    else:
        frame_list.append(
            {
                "shot": shot_count,
                "frame": frame
            }
        )
        shot_count = shot_count + 1

with open("MiSang_Frame.json", 'w') as outfile:
    json.dump(frame_list, outfile, indent=4)