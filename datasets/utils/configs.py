
SEMSEG_NUM_OUTPUT = {
    'nyud': 40,
    'pascalcontext': 21,
}

HUMAN_PARTS_NUM_OUTPUT = {
    'pascalcontext': 7,
}

SAL_NUM_OUTPUT = {
    'pascalcontext': 1,
}

NORMALS_NUM_OUTPUT = {
    'pascalcontext': 3,
    'nyud': 3,
}

EDGE_NUM_OUTPUT = {
    'pascalcontext': 1,
    'nyud': 1,
}

DEPTH_NUM_OUTPUT = {
    'nyud': 1,
}

def get_output_num(task, dataname):
    if task.lower() == 'semseg':
        return SEMSEG_NUM_OUTPUT[dataname]
    elif task.lower() == 'human_parts':
        return HUMAN_PARTS_NUM_OUTPUT[dataname]
    elif task.lower() == 'sal':
        return SAL_NUM_OUTPUT[dataname]
    elif task.lower() == 'normals':
        return NORMALS_NUM_OUTPUT[dataname]
    elif task.lower() == 'edge':
        return EDGE_NUM_OUTPUT[dataname]
    elif task.lower() == 'depth':
        return DEPTH_NUM_OUTPUT[dataname]

TRAIN_SCALE = {
    'pascalcontext': (512, 512),
    'nyud': (480, 640)
}

TEST_SCALE = {
    'pascalcontext': (512, 512),
    'nyud': (480, 640)
}

NUM_TRAIN_IMAGES = {
    'pascalcontext': 4998,
    'nyud': 795
}

NUM_TEST_IMAGES = {
    'pascalcontext': 5105,
    'nyud': 654
}