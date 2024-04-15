#%%
import factorySim.baseConfigs as baseConfigs
from factorySim.factorySimClass import FactorySim
from tqdm.auto import tqdm
import os
import ifcopenshell
from ifcopenshell.api import run
from ifcopenshell.util.shape_builder import ShapeBuilder
from mathutils import Vector
from shapely.geometry import box, MultiPoint, Polygon, MultiPolygon
from shapely.affinity import  scale

import math
import numpy as np

factoryConfig = baseConfigs.SMALLSQUARE



if __name__ == "__main__":

    print("Creating Factory")
    basePath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    outputPath = os.path.join(basePath, "Output")
    #Create directory if it does not exist
    ifcPath = os.path.join(basePath, "..","Input", "2", "Basic.ifc")
    print(ifcPath)

    factory = FactorySim(path_to_ifc_file=ifcPath,factoryConfig=factoryConfig, randSeed=0, createMachines=False, randomPos=False)
    print(factory.machine_dict)




# %%


def write_ifc_class(ifc_file, ifc_context, ifc_class, element_dict):
    builder = ShapeBuilder(ifc_file)
    elements = []

    for element in element_dict.values():
        print(element.name)

        ifc_element = run("root.create_entity", ifc_file, ifc_class=ifc_class, name=element.name)
        #matrix[:,3][0:3] = (element.origin[0]/1000, element.origin[1]/1000, 0)

        run("geometry.edit_object_placement", model, product=ifc_element)
        #run("geometry.edit_object_placement", ifc_file, product=ifc_element, matrix=matrix, is_si=True)


        vertices = []
        edges = []
        for poly in element.poly.geoms:
            outer_curve = builder.polyline(poly.exterior.coords)

            inner_curves = []
            for ring in poly.interiors:
                inner_curves.append(builder.polyline(ring.coords))

        profile = builder.profile(outer_curve, inner_curves=inner_curves, name=element.name)

        representation = run("geometry.add_profile_representation", ifc_file, context=ifc_context, profile=profile, depth=0)
        run("geometry.assign_representation", ifc_file, product=ifc_element, representation=representation)
        elements.append(ifc_element)
    return elements




# Create a blank model
model = ifcopenshell.file()

# All projects must have one IFC Project element
project = run("root.create_entity", model, ifc_class="IfcProject", name="FactorySim Project")

# Geometry is optional in IFC, but because we want to use geometry in this example, let's define units
# Assigning without arguments defaults to metric units
length = run("unit.add_si_unit", model, unit_type="LENGTHUNIT", prefix="MILLI")
run("unit.assign_unit", model, units=[length])

# Let's create a modeling geometry context, so we can store 3D geometry (note: IFC supports 2D too!)
model3d = run("context.add_context", model, context_type="Model")
plan  = run("context.add_context", model, context_type="Plan")

# In particular, in this example we want to store the 3D "body" geometry of objects, i.e. the body shape
body = run("context.add_context", model, context_type="Model",
    context_identifier="Body", target_view="MODEL_VIEW", parent=model3d)

drawing = run("context.add_context", model,
    context_type="Plan", context_identifier="Annotation", target_view="PLAN_VIEW", parent=plan)

# Create a site, building, and storey. Many hierarchies are possible.
site = run("root.create_entity", model, ifc_class="IfcSite", name="IfcSite")
building = run("root.create_entity", model, ifc_class="IfcBuilding", name="IfcBuilding")
storey = run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="IfcBuildingStorey")

# Since the site is our top level location, assign it to the project
# Then place our building on the site, and our storey in the building
run("aggregate.assign_object", model, relating_object=project, product=site)
run("aggregate.assign_object", model, relating_object=site, product=building)
run("aggregate.assign_object", model, relating_object=building, product=storey)


elements = write_ifc_class(model, drawing, "IfcBuildingElementProxy", factory.machine_dict)
run("spatial.assign_container", model, relating_structure=storey, products=elements)

elements = write_ifc_class(model, drawing, "IfcWall", factory.wall_dict)
run("spatial.assign_container", model, relating_structure=storey, products=elements)


# Write out to a file
model.write("model.ifc")





# %%
elementName = "IFCBUILDINGELEMENTPROXY"
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
first_poly= list(factory.machine_dict.values())[0]
vertices = [list(poly.exterior.coords) for poly in first_poly.poly.geoms]

vertices = []

for poly in first_poly.poly.geoms:
    lineString = [tuple([point[0], point[1], 0]) for point in poly.exterior.coords]
    vertices.append(lineString)

print(vertices)

edges = []
num_vertices = len(vertices)
print(num_vertices)
for i in range(num_vertices - 1):
    print(i)
    edges.append((i, i + 1))
edges.append((num_vertices - 1, 0))  # Connect the last vertex to the first one to close the loop

print(edges)
# %%
