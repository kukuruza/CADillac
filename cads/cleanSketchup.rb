require 'sketchup.rb'

def setOriginToCenter 
    bb = Geom::BoundingBox.new ;   # Outter Shell 
    Sketchup.active_model.entities.each {|e| bb.add(e.bounds) ; } 

    vector = Geom::Vector3d.new -bb.center[0], -bb.center[1], 0
    transformation = Geom::Transformation.translation vector

    entities = Sketchup.active_model.entities
    entities.transform_entities(transformation, entities.to_a)
end

def orientAlongX
    bb = Sketchup.active_model.bounds 
    center = bb.center

    if bb.width < bb.height
        puts 'bb.width < bb.height: will rotate'
        vectorUp = Geom::Vector3d.new 0,0,1
        origin   = Geom::Point3d.new 0,0,0
        rot = Geom::Transformation.rotation origin, vectorUp, Math::PI / 2

        entities = Sketchup.active_model.entities
        entities.transform_entities(rot, entities.to_a)
    else
        puts 'bb.width >= bb.height: leave as it is'
    end
end

def rotate180
    vectorUp = Geom::Vector3d.new 0,0,1
    origin   = Geom::Point3d.new 0,0,0
    rot = Geom::Transformation.rotation origin, vectorUp, Math::PI

    entities = Sketchup.active_model.entities
    entities.transform_entities(rot, entities.to_a)
end

def exportObj (filepath)
    Sketchup.active_model.export filepath
end

def scaleTo (target_dims_meters)
    # that's what we have
    bb = Sketchup.active_model.bounds 
    actual_x = bb.width.to_m
    actual_y = bb.height.to_m
    actual_z = bb.depth.to_m
    # here's what we want to have
    target_x = target_dims_meters[0]
    target_y = target_dims_meters[1]
    target_z = target_dims_meters[2]
    # assume mirrors take 10% of car width
    actual_y = actual_y / 1.1
    # we only use car width (y) as the most reliable
    scale = target_y / actual_y
    origin   = Geom::Point3d.new 0,0,0
    transform = Geom::Transformation.scaling origin, scale
    entities = Sketchup.active_model.entities
    entities.transform_entities(transform, entities.to_a)
end


## does not work. At all
def scaleTo2 (target_dims_meters)
    bb = Sketchup.active_model.bounds 
    car_height = bb.depth
    min_mirror_z = 0.50 * car_height  # mirrors are definitely higher than this

    adjusted_car_width = 0    # init with large number
    adjusted_car_min = 0
    for part in Sketchup.active_model.definitions
        max_corner = part.bounds.max
        min_corner = part.bounds.min
        # "max_corner.z > 0" -- a hack to remove some exhilary shit
        if max_corner.z > 0 and max_corner.z < bb.depth * min_mirror_z
            adjusted_car_width = [max_corner.y.to_m, adjusted_car_width].max
            #puts part.name, max_corner, adjusted_car_width
            adjusted_car_min   = [min_corner.y.to_m, adjusted_car_min].min
            #puts part.name, max_corner, adjusted_car_width
        end
    end
    raise 'adjusted_car_width < 0' if adjusted_car_width == 0
    puts "adjusted max: #{adjusted_car_width}, min: #{adjusted_car_min}"
    #puts "car_width: #{bb.height.to_m}, adjusted: #{adjusted_car_width}"

    # here's what we want to have
    target_car_width = target_dims_meters[1]
    # we only use car width (y) as the most reliable (no extra stuff on sides)
    scale = target_car_width / adjusted_car_width
    origin   = Geom::Point3d.new 0,0,0
    transform = Geom::Transformation.scaling origin, scale
    entities = Sketchup.active_model.entities
    #entities.transform_entities(transform, entities.to_a)

end


def printoutDims
    bb = Sketchup.active_model.bounds 
    bb_width  = bb.width.to_m.round(2)
    bb_height = bb.height.to_m.round(2)
    bb_depth  = bb.depth.to_m.round(2)
    puts "\"dims\": [#{bb_width}, #{bb_height}, #{bb_depth}]"
end

#=============================================================================

if( not file_loaded?("centerpoint.rb") )
    UI.menu("Plugins").add_item("Set Origin To Center") { setOriginToCenter }
    UI.menu("Plugins").add_item("Orient Along X") { orientAlongX }
    UI.menu("Plugins").add_item("Rotate 180") { rotate180 }
    UI.menu("Plugins").add_item("Print dimensions") { printoutDims }
end
#-----------------------------------------------------------------------------
file_loaded("centerpoint.rb")

