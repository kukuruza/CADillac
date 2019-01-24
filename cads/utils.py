import logging
import collections
from math import cos, sin, pi, sqrt
import bpy
from mathutils import Vector


def get_origin_and_dims (car_group_name, height_ratio = 0.33, min_count = 10):
    '''Find car origin and dimensions by taking max dimensions below mirrors.
    Args:
    Returns:
      origin:  Dict with 'x', 'y', 'z' entries. 
               'x' and 'y' = center of the car body (no mirrors), 'z' = min(z)
      dims:    Dict with 'x', 'y', 'z'
               'x' and 'y' = dimensions of the car body (no mirrors)
    '''
    # find the z dimensions of the car
    z_min = 1000
    z_max = -1000

    for obj in bpy.data.groups[car_group_name].objects:
        roi = bounds(obj)
        z_min = min(roi.z.min, z_min)
        z_max = max(roi.z.max, z_max)
    logging.info ('z_min = %f, z_max = %f' % (z_min, z_max))

    # find x and y dimensions for everything below third height
    y_min_body = y_min_mirrors = 1000
    y_max_body = y_max_mirrors = -1000
    x_min_body = x_min_mirrors = 1000
    x_max_body = x_max_mirrors = -1000
    count_below_height_ratio = 0
    for obj in bpy.data.groups[car_group_name].objects:
        roi = bounds(obj)
        # update total (with mirrors and other expanding stuff) dimensions
        y_min_mirrors = min(roi.y.min, y_min_mirrors)
        y_max_mirrors = max(roi.y.max, y_max_mirrors)
        x_min_mirrors = min(roi.x.min, x_min_mirrors)
        x_max_mirrors = max(roi.x.max, x_max_mirrors)
        # update body dimensions
        if (roi.z.min + roi.z.max) / 2 < z_min + (z_max - z_min) * height_ratio:
            y_min_body = min(roi.y.min, y_min_body)
            y_max_body = max(roi.y.max, y_max_body)
            x_min_body = min(roi.x.min, x_min_body)
            x_max_body = max(roi.x.max, x_max_body)
            count_below_height_ratio += 1
    # check the number of objects that are low enough
    count_all = len(bpy.data.groups[car_group_name].objects)
    logging.debug ('out of %d objects, %d are below height_ratio %f' % \
        (count_all, count_below_height_ratio, height_ratio))
    if count_below_height_ratio < min_count:
        raise MinHeightException('not enough objects below %.2f of height' % height_ratio)

    # verbose output
    length_body    = x_max_body - x_min_body
    width_body     = y_max_body - y_min_body
    width_mirrors  = y_max_mirrors - y_min_mirrors    
    logging.debug ('mirrors dims: %.2f < y < %.2f' % (y_min_mirrors, y_max_mirrors))
    logging.debug ('body dims:    %.2f < y < %.2f' % (y_min_body, y_max_body))
    logging.info ('body/mirrors width ratio = %.3f' % (width_body / width_mirrors))
    if width_body == width_mirrors:
        logging.warning ('mirror and body widths are equal: %.2f' % width_body)

    # form the result  (note origin.z=z_min)
    origin = [(x_min_body+x_max_body)/2, (y_min_body+y_max_body)/2, z_min]
    origin = dict(zip(['x', 'y', 'z'], origin))
    dims = [x_max_body-x_min_body, y_max_body-y_min_body, z_max-z_min]
    dims = dict(zip(['x', 'y', 'z'], dims))

    return origin, dims


def get_origin_and_dims_adjusted (car_group_name, min_count = 10):
    '''Run get_origin_and_dims with different parameters for adjusted result
    '''
    # first compensate for mirrors and save adjusted width
    for height_ratio in [0.1, 0.2, 0.33, 0.5, 0.7, 1]:
        try:
            logging.debug ('get_origin_and_dims_adjusted: trying %f' % height_ratio)
            _, dims = get_origin_and_dims (car_group_name, height_ratio, min_count)
            adjusted_width = dims['y']
            break
        except MinHeightException:
            logging.debug ('height_ratio %f failed' % height_ratio)

    # then do not compensate for anything for correct length and height
    origin, dims, _ = get_origin_and_dims (car_group_name, height_ratio = 1,
                                        min_count = min_count)
    dims['y'] = adjusted_width
    return origin, dims


def get_origin_and_dims_single (car_group_name, mirrors=False):
    '''
    Args:
      mirrors:   if True, adjust 'y' for mirrors
    '''

    obj = bpy.data.groups[car_group_name].objects[0]
    roi = bounds(obj)
    origin = [(roi.x.min+roi.x.max)/2, (roi.y.min+roi.y.max)/2, roi.z.min]
    origin = dict(zip(['x', 'y', 'z'], origin))
    dims = [roi.x.max-roi.x.min, roi.y.max-roi.y.min, roi.z.max-roi.z.min]
    dims = dict(zip(['x', 'y', 'z'], dims))
    if mirrors:
        dims['y'] *= 0.87
    return origin, dims
    



def set_origin_to_zero (model_id):

    # give 3dcursor new coordinates
    bpy.context.scene.cursor_location = Vector((0.0,0.0,0.0))

    # set the origin on the current object to the 3dcursor location
    obj = bpy.data.objects[model_id]
    obj.origin_set(type='ORIGIN_CURSOR')



def bounds(obj, local=False):
    '''Usage:
        object_details = bounds(obj)
        a = object_details.z.max
        b = object_details.z.min
        c = object_details.z.distance
    '''

    local_coords = obj.bound_box[:]
    om = obj.matrix_world

    if not local:    
        worldify = lambda p: om * Vector(p[:]) 
        coords = [worldify(p).to_tuple() for p in local_coords]
    else:
        coords = [p[:] for p in local_coords]

    rotated = zip(*coords[::-1])

    push_axis = []
    for (axis, _list) in zip('xyz', rotated):
        info = lambda: None
        info.max = max(_list)
        info.min = min(_list)
        info.distance = info.max - info.min
        push_axis.append(info)

    originals = dict(zip(['x', 'y', 'z'], push_axis))

    o_details = collections.namedtuple('object_details', 'x y z')
    return o_details(**originals)


def get_x_wheels(obj):
    ''' Get x coordinates of supposeably wheels.
    Assumes car is on the ground (z_min = 0)
    Will get all the low vertices.
    '''

    om = obj.matrix_world
    global_coords = [om * v.co for v in obj.data.vertices]

    roi = bounds(obj)
    low_coords = [c for c in global_coords if c.z < roi.z.max * 0.05]

    x_coords = [c.x for c in low_coords]
    return x_coords
