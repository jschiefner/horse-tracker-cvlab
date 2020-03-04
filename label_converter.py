# %% imports
import sys
import pandas as pd

# %% parse argument

usage = 'usage: python label_converter.py (pair|single)'
if len(sys.argv) != 2:
    print(usage)
    exit(1)
mode = sys.argv[1]
if mode != 'pair' and mode != 'single':
    print(usage)
    exit(1)

# %% data

data = pd.read_csv('rimondo_filtered.csv', usecols=['username', 'image', 'label', 'x', 'y', 'width', 'height'])
data.sort_values(by=['image', 'username', 'label'], inplace=True)
data.drop_duplicates(subset=['username', 'image', 'label'], keep=False, inplace=True)
data.drop_duplicates(subset=['image', 'label'], keep='first', inplace=True)
data.label = data.label.map(lambda e: e-1)
data = data.loc[data['label'] <= 1] # drop false labels
data.index = range(len(data))
data.index.rename('index', inplace=True)

print('{} entries to work with'.format(len(data)))

# %% write

image_width = 3840
image_height = 2160

out_filename = 'annotations.txt'

if mode == 'single':
    with open(out_filename, 'w') as file:
        before = None
        for row in data.itertuples():
            center_x = image_width * row.x
            distance_x = image_width * row.width / 2
            x_min = int(center_x - distance_x)
            x_max = int(center_x + distance_x)

            center_y = image_height * row.y
            distance_y = image_height * row.height / 2
            y_min = int(center_y - distance_y)
            y_max = int(center_y + distance_y)

            if row.label == 0:
                if before is not None and before.label == 0: continue
                file.write('{}png {},{},{},{},{} '.format(row.image[0:-3], x_min, y_min, x_max, y_max, 0))
            else:
                if before is not None and before.label == 1: continue
                file.write('{},{},{},{},{}\n'.format(x_min, y_min, x_max, y_max, 1))
            before = row
else:
    i = 0
    new = pd.DataFrame(columns=['index', 'username', 'image', 'label', 'x', 'y', 'width', 'height'])
    for index, row in enumerate(data.itertuples()):
        if i == 0:
            new.loc[i] = data.loc[i]
            i += 1
            continue
        # i > 0
        if row.label == 0:
            next = data.loc[index+1]
            if next.label == 1 and next.username == row.username and next.image == row.image:
                new.loc[i] = row
                i += 1
        else: # row.label == 1
            previous = data.loc[index-1]
            if previous.label == 0 and previous.username == row.username and previous.image == row.image:
                new.loc[i] = row
                i += 1

    new.drop('index', axis=1, inplace=True)
    new.index.rename('index', inplace=True)
    
    # check if new has the correct format:
    for index, row in enumerate(new.itertuples()):
        if index % 2 == 0: # index even
            next = new.loc[index+1]
            if not (row.username == next.username and row.image == next.image and row.label != next.label):
                # this should not be raised
                raise RuntimeError('incorrect format, every even line has to be of label 0 and the next line of label 1 but with the same image and username data')
    
    with open(out_filename, 'w') as file:
        before = None
        for index, row in enumerate(new.itertuples()):
            if index % 2 == 0: # index even
                horse = row
                person = new.loc[index+1]
                
                center_x_horse = image_width * horse.x
                distance_x_horse = image_width * horse.width / 2
                x_min_horse = int(center_x_horse - distance_x_horse)
                x_max_horse = int(center_x_horse + distance_x_horse)
                
                center_y_horse = image_height * horse.y
                distance_y_horse = image_height * horse.height / 2
                y_min_horse = int(center_y_horse - distance_y_horse)
                y_max_horse = int(center_y_horse + distance_y_horse)
                
                center_x_person = image_width * person.x
                distance_x_person = image_width * person.width / 2
                x_min_person = int(center_x_person - distance_x_person)
                x_max_person = int(center_x_person + distance_x_person)
                
                center_y_person = image_height * person.y
                distance_y_person = image_height * person.height / 2
                y_min_person = int(center_y_person - distance_y_person)
                y_max_person = int(center_y_person + distance_y_person)
                
                x_min = min([x_min_horse, x_min_person])
                y_min = min([y_min_horse, y_min_person])
                x_max = max([x_max_horse, x_max_person])
                y_max = max([y_max_horse, y_max_person])
                
                file.write('{}png {},{},{},{},{}\n'.format(horse.image[0:-3], x_min, y_min, x_max, y_max, 0))
    print('Written annotations to {}'.format(out_filename))