# %% imports
import os
folder_path = 'txt-files'

frame_width = 3840
frame_height = 2160

# %% action

out_path = os.path.join(folder_path, 'annotations.txt')
out_file = open(out_path, 'w')

for folder in os.walk(folder_path):
    folder, folders, files = folder
    for filename in files:
        if filename == 'classes.txt' or filename == 'annotations.txt': continue
        filepath = os.path.join(folder, filename)
        index, ext = os.path.splitext(filename)
        if ext != '.txt': continue
        image_path = os.path.join(folder, f'{index}.png')
        with open(filepath, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # todo: find correct alignment of values
                _, x, y, width, height = line.split(' ')
                x = float(x) * frame_width
                y = float(y) * frame_height
                width = float(width) * frame_width
                height = float(height) * frame_height
                half_width = width / 2
                half_height = height / 2
                
                left = round(x - half_width)
                right = round(x + half_width)
                top = round(y - half_height)
                bottom = round(y + half_height)
                
                output_str = f'{image_path} {left},{top},{right},{bottom},0\n'
                # print(output_str)
                # todo: write to file
                out_file.write(output_str)
                
    
out_file.close()