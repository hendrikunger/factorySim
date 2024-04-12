#%%
from factorySim.creation import FactoryCreator
import factorySim.baseConfigs as baseConfigs
from factorySim.factorySimClass import FactorySim
from tqdm.auto import tqdm
import os
import ifcopenshell
from ifcopenshell.api import run
import ifcopenshell.geom
import ifcopenshell.util.shape
from shapely.geometry import box, MultiPoint, Polygon, MultiPolygon
from shapely.affinity import rotate, scale, translate
from shapely.ops import unary_union
from shapely.prepared import prep
from factorySim.factoryObject import FactoryObject
import math
import numpy as np

factoryConfig = baseConfigs.SMALLSQUARE



if __name__ == "__main__":

    print("Creating Factory")
    basePath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    outputPath = os.path.join(basePath, "Output")
    #Create directory if it does not exist
    ifcPath = os.path.join(basePath, "..","Input", "2")
    print(ifcPath)

    factory = FactorySim(path_to_ifc_file=ifcPath,factoryConfig=factoryConfig, randSeed=0, createMachines=False)
    print(factory.machine_dict)


    # #save line to json
    # with open("factory.pkl", "wb") as f:
    #     pickle.dump(factory.machine_dict, f)
    


ifc_file = ifcopenshell.open(os.path.join(ifcPath, "Basic.ifc"))
# %%

# Create a blank model
model = ifcopenshell.file()

# All projects must have one IFC Project element
project = run("root.create_entity", model, ifc_class="IfcProject", name="My Project")

# Geometry is optional in IFC, but because we want to use geometry in this example, let's define units
# Assigning without arguments defaults to metric units
length = run("unit.add_si_unit", model, unit_type="LENGTHUNIT", prefix="MILLI")
run("unit.assign_unit", model, units=[length])

# Let's create a modeling geometry context, so we can store 3D geometry (note: IFC supports 2D too!)
context = run("context.add_context", model, context_type="Model")

# In particular, in this example we want to store the 3D "body" geometry of objects, i.e. the body shape
body = run("context.add_context", model, context_type="Model",
    context_identifier="Body", target_view="MODEL_VIEW", parent=context)

# Create a site, building, and storey. Many hierarchies are possible.
site = run("root.create_entity", model, ifc_class="IfcSite", name="My Site")
building = run("root.create_entity", model, ifc_class="IfcBuilding", name="Building A")
storey = run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="Ground Floor")

# Since the site is our top level location, assign it to the project
# Then place our building on the site, and our storey in the building
run("aggregate.assign_object", model, relating_object=project, product=site)
run("aggregate.assign_object", model, relating_object=site, product=building)
run("aggregate.assign_object", model, relating_object=building, product=storey)

# Let's create a new wall
wall = run("root.create_entity", model, ifc_class="IfcWall")


# Create a 4x4 identity matrix. This matrix is at the origin with no rotation.
matrix = np.eye(4)

# Rotate the matix 90 degrees anti-clockwise around the Z axis (i.e. in plan).
# Anti-clockwise is positive. Clockwise is negative.
matrix = ifcopenshell.util.placement.rotation(90, "Z") @ matrix

# Set the X, Y, Z coordinates. Notice how we rotate first then translate.
# This is because the rotation origin is always at 0, 0, 0.
matrix[:,3][0:3] = (2, 3, 5)

# Give our wall a local origin at (0, 0, 0)
run("geometry.edit_object_placement", model, product=wall, matrix=matrix, is_si=True)

# Add a new wall-like body geometry, 5 meters long, 3 meters high, and 200mm thick
representation = run("geometry.add_wall_representation", model, context=body, length=5, height=3, thickness=0.2)
# Assign our new body geometry back to our wall
run("geometry.assign_representation", model, product=wall, representation=representation)


pyramide = run("root.create_entity", model, ifc_class="IfcBuildingElementProxy")
run("geometry.edit_object_placement", model, product=pyramide)
# These vertices and faces represent a 2m square 1m high pyramid in SI units.
# Note how they are nested lists. Each nested list represents a "mesh". There may be multiple meshes.
vertices = [[(0.,0.,0.), (0.,2.,0.), (2.,2.,0.), (2.,0.,0.), (1.,1.,1.)]]
faces = [[(0,1,2,3), (0,4,1), (1,4,2), (2,4,3), (3,4,0)]]
representation = run("geometry.add_mesh_representation", model, context=body, vertices=vertices, faces=faces)
run("geometry.assign_representation", model, product=pyramide, representation=representation)

# Place our wall in the ground floor
run("spatial.assign_container", model, relating_structure=storey, products=[wall, pyramide],)


# Write out to a file
model.write("model.ifc")





# %%

elementName = "IFCBUILDINGELEMENTPROXY"
model = ifcopenshell.open(os.path.join(ifcPath, "Basic.ifc"))
element_dict = {}
elements = model.by_type(elementName)

for index, element in enumerate(elements):
    #get origin
    origin = element.ObjectPlacement.RelativePlacement.Location.Coordinates
    #element.ObjectPlacement.RelativePlacement.Axis.DirectionRatios[0]
    #element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[0]

    #get rotation
    x = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[0]
    y = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[1]
    rotation = math.atan2(y,x)
    my_uuid = ""

    #points = element.Representation.Representations[0].Items[0].Outer.CfsFaces[0].Bounds[0].Bound.Polygon
    #Always choose Representation 0
    items = element.Representation.Representations[0].Items

    #Parse imported data into Factory Object (Machine, Wall, ...)

    result = []

    for item in items:
        faces = item.Outer.CfsFaces
        facelist = []

        for face in faces:
            exterior = []
            interior = []
            bounds = face.Bounds
            for bound in bounds:
                type = bound.get_info()['type']
                points = bound.Bound.Polygon
                #Remove z Dimension of Polygon Coordinates
                pointlist = [(point.Coordinates[0], point.Coordinates[1]) for point in points ]
                #Check if face is a surface or a hole
                if (type == "IfcFaceBound"):
                    interior.append(pointlist)
                else:
                    #Case surface
                    exterior = pointlist

            facelist.append(Polygon(exterior, interior))
        
        result.append(MultiPolygon(facelist))


    singleElement = unary_union(result)
    if singleElement.type != "MultiPolygon":
        singleElement = MultiPolygon([singleElement])
    #Fix coordinates, since ifc does not save geometry at the correct position
    singleElement = translate(singleElement, origin[0], origin[1])
    singleElement = rotate(singleElement, rotation, origin=(origin[0], origin[1]), use_radians=True)
    #create Factory Object       
    
    element_dict[element.GlobalId] = FactoryObject(gid=element.GlobalId, 
                                                    name=element.Name + my_uuid,
                                                    origin=(origin[0], origin[1]),
                                                    poly=singleElement)
del(ifc_file)  #Hopefully fixes memory leak

if True:
    bbox = unary_union([x.poly for x in element_dict.values()])
    #Prevent error due to single element in IFC File
    if bbox.type == "MultiPolygon":
        bbox = bbox.bounds
    else:
        bbox = MultiPolygon([bbox]).bounds
    bb = box(bbox[0], bbox[1], bbox[2], bbox[3])
    factoryWidth = bbox[2] - bbox[0]
    factoryHeight = bbox[3] - bbox[1]

for element in element_dict.values():
    element.poly = scale(element.poly, yfact=-1, origin=bb.centroid)
    polybbox = element.poly.bounds
    element.origin = (polybbox[0], polybbox[1])
    element.center = element.poly.representative_point()

print(element_dict)



# %%

element = model.by_type(elementName)[5]
settings = ifcopenshell.geom.settings()
shape = ifcopenshell.geom.create_shape(settings, element)
print(shape.name)
dir(shape)

# %%
print(ifcopenshell.util.shape.get_shape_matrix(shape))
#
# %%
element.ObjectPlacement.RelativePlacement.Location.Coordinates
# %%
x = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[0]
y = element.ObjectPlacement.RelativePlacement.RefDirection.DirectionRatios[1]
rotation = math.atan2(y,x)
print(x,y,rotation)
matrix= (ifcopenshell.util.shape.get_shape_matrix(shape))
print(matrix[0,0], matrix[1,0])
# %%
print(shape.geometry.faces)
print(ifcopenshell.util.shape.get_vertices(shape.geometry))
print(ifcopenshell.util.shape.get_shape_vertices(shape, shape.geometry))
print(ifcopenshell.util.shape.get_edges(shape.geometry))
poly = Polygon(ifcopenshell.util.shape.get_shape_vertices(shape, shape.geometry)[0:3])
# %%
items = element.Representation.Representations[0].Items
for item in items:
    faces = item.Outer.CfsFaces
    for face in faces:
        exterior = []
        interior = []
        bounds = face.Bounds
        for bound in bounds:
            print(bound.get_info()['type'])
            points = bound.Bound.Polygon
            pointlist = [(point.Coordinates[0], point.Coordinates[1]) for point in points ]
            print(pointlist)
# %%
poly
# %%
print(shape.geometry.edges)
print(shape.geometry.verts)

# %%
ifcopenshell.util.unit.calculate_unit_scale(model)
# %%
