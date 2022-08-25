import json
import cv2

shot_count = 1
frame_list = []
frame = 1

with open("MiSang_Frame.json", 'r') as outfile:
    frame_list = json.load(outfile)
    shot_count = frame_list[-1]["shot"] + 1
    frame = frame_list[-1]["frame"] + 1

print(f'Current Shot : {shot_count}')
while(True):
    image = cv2.imread(f'C:\\Users\\user\\Desktop\\MiSang_Frame\\{frame}.jpg') #작업 이미지 폴더
    cv2.imshow('img', image)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    elif key == ord('s'):
        frame_list.append(
                {
                    "shot": shot_count,
                    "frame": frame
                }
            )
        print(f'Saved Frame : {frame}')
        frame = frame + 1
        shot_count = shot_count + 1
        print(f'Current Shot : {shot_count}')
    elif key == ord('a'):
        frame = frame - 1
    else :
        frame = frame + 1

with open("MiSang_Frame.json", 'w') as outfile:
    json.dump(frame_list, outfile, indent=4)

cv2.destroyAllWindows()