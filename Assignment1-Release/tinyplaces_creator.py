import numpy as np
import pickle
from PIL import Image


cat_to_id = {
    '16': 0,
    '18': 1,
    '20': 2,
    '33': 3,
    '45': 4,
    '48': 5,
    '59': 6,
    '63': 7,
    '62': 8,
    '70': 9,
    '15': 10,
    '24': 11,
    '27': 12,
    '29': 13,
    '35': 14,
    '49': 15,
    '53': 16,
    '74': 17,
    '68': 18,
    '77': 19
}

loc_nums = {
    '16': 0,
    '18': 0,
    '20': 0,
    '33': 0,
    '45': 0,
    '48': 0,
    '59': 0,
    '63': 0,
    '62': 0,
    '70': 0,
    '15': 0,
    '24': 0,
    '27': 0,
    '29': 0,
    '35': 0,
    '49': 0,
    '53': 0,
    '74': 0,
    '68': 0,
    '77': 0
}

indoor = {
    '16',
    '18',
    '20',
    '33',
    '45',
    '48',
    '59',
    '63',
    '62',
    '70'
}

outdoor = {
    '15',
    '24',
    '27',
    '29',
    '35',
    '49',
    '53',
    '74',
    '68',
    '77'
}

def im_to_np(im):
    im = im.resize((32, 32))
    im = (np.array(im))
    r = im[:,:,0].flatten()
    g = im[:,:,1].flatten()
    b = im[:,:,2].flatten()
    return np.concatenate((r, g, b), axis = 0)


TRAIN_SIZE = 10000
TRAIN_CAT = 500

VAL_SIZE = 1000
VAL_CAT = 50

if __name__ == '__main__':

    
    prepend = 'images/'
    
    
    # generate training set
    data = np.empty((TRAIN_SIZE, 3072), dtype=np.uint8)
    label = np.empty(TRAIN_SIZE, dtype=np.int32)
    with open('train.txt', 'r') as train_list:
        Lines = train_list.readlines()
        count = 0
        for line in Lines:
            path, cat = line.split()
            if cat in loc_nums and loc_nums[cat] < TRAIN_CAT: 
                loc_nums[cat] += 1
                path = prepend + path
                with Image.open(path) as im:
                    data[count] = im_to_np(im)
                    label[count] = 0 if cat in indoor else 1
                count += 1
                if count >= TRAIN_SIZE:
                    break
            
    
    shuffler = np.random.permutation(len(data))
    data_s = data[shuffler]
    label_s = label[shuffler]
            
    tinyplaces_train = {
        'data': data_s,
        'label': label_s
    }
    
    with open('data/tinyplaces-train', 'wb') as f:
        pickle.dump(tinyplaces_train, f)
    

    # clear counter
    for key in loc_nums:
        loc_nums[key] = 0
    
    # generate validation set
    v_data = np.empty((VAL_SIZE, 3072), dtype=np.uint8)
    v_label = np.empty(VAL_SIZE, dtype=np.int32)
    with open('val.txt', 'r') as train_list:
        Lines = train_list.readlines()
        count = 0
        for line in Lines:
            path, cat = line.split()
            if cat in loc_nums and loc_nums[cat] < VAL_CAT: 
                loc_nums[cat] += 1
                path = prepend + path
                with Image.open(path) as im:
                    v_data[count] = im_to_np(im)
                    v_label[count] = 0 if cat in indoor else 1
                count += 1
                if count >= VAL_SIZE:
                    break
            
    shuffler_v = np.random.permutation(len(v_data))
    v_data_s = v_data[shuffler_v]
    v_label_s = v_label[shuffler_v]
            
    tinyplaces_val = {
        'data': v_data_s,
        'label': v_label_s
    }
    
    with open('data/tinyplaces-val', 'wb') as f:
        pickle.dump(tinyplaces_val, f)
      
    # clear counter
    for key in loc_nums:
        loc_nums[key] = 0
    
    # generate multiclass training set
    m_t_data = np.empty((TRAIN_SIZE, 3072), dtype=np.uint8)
    m_t_label = np.empty(TRAIN_SIZE, dtype=np.int32)
    with open('train.txt', 'r') as train_list:
        Lines = train_list.readlines()
        count = 0
        for line in Lines:
            path, cat = line.split()
            if cat in loc_nums and loc_nums[cat] < TRAIN_CAT: 
                loc_nums[cat] += 1
                path = prepend + path
                with Image.open(path) as im:
                    m_t_data[count] = im_to_np(im)
                    m_t_label[count] = cat_to_id[cat]
                count += 1
                if count >= TRAIN_SIZE:
                    break
            
    shuffler_v = np.random.permutation(len(m_t_data))
    m_t_data_s = m_t_data[shuffler_v]
    m_t_label_s = m_t_label[shuffler_v]
            
    tinyplaces_train_multiclass = {
        'data': m_t_data_s,
        'label': m_t_label_s
    }
    
    with open('data/tinyplaces-train-multiclass', 'wb') as f:
        pickle.dump(tinyplaces_train_multiclass, f)
        
# clear counter
    for key in loc_nums:
        loc_nums[key] = 0
    
    # generate multiclass training set
    m_v_data = np.empty((VAL_SIZE, 3072), dtype=np.uint8)
    m_v_label = np.empty(VAL_SIZE, dtype=np.int32)
    with open('val.txt', 'r') as train_list:
        Lines = train_list.readlines()
        count = 0
        for line in Lines:
            path, cat = line.split()
            if cat in loc_nums and loc_nums[cat] < VAL_CAT: 
                loc_nums[cat] += 1
                path = prepend + path
                with Image.open(path) as im:
                    m_v_data[count] = im_to_np(im)
                    m_v_label[count] = cat_to_id[cat]
                count += 1
                if count >= VAL_SIZE:
                    break
            
    shuffler_v = np.random.permutation(len(m_v_data))
    m_v_data_s = m_v_data[shuffler_v]
    m_v_label_s = m_v_label[shuffler_v]
            
    tinyplaces_val_multiclass = {
        'data': m_v_data_s,
        'label': m_v_label_s
    }
    
    with open('data/tinyplaces-val-multiclass', 'wb') as f:
        pickle.dump(tinyplaces_val_multiclass, f)